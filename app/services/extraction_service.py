"""End-to-end BOQ extraction.

    PDF -> Docling -> evidence -> GPT-OSS 120B -> Pydantic -> rules -> BOQ JSON

`extract_boq_from_pdf(path)` runs the whole pipeline in one call. It is composed
of two halves that callers may also use separately:

    run_document_stage(path)             PDF  -> (DoclingResult, EvidencePackage)
    extract_boq_from_evidence(evidence)  evidence -> BOQResult

The CLI uses the halves, because it also reports on the document stage and
writes Docling's own exports; the API layer only needs the single call.

**What raises, and what reports.** The service raises when it could not form a
judgement about the document at all, and returns a failed report when it could:

    raises     DocumentError        the input is not a usable PDF
               ExtractionError      Docling could not process it
               ConfigurationError   GROQ_API_KEY is not set
               AIRequestError       the model was never reached

    reports    NO_BOQ_DETECTED      the document holds nothing to interpret
               LLM_INVALID_OUTPUT   the model answered and the answer was refused
               rule violations      items were extracted but do not hold up

A request error means no answer ever arrived; a response error means one did and
was rejected. Only the second is a statement about the document, so only the
second belongs in an extraction report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.agent.boq_agent import AgentResult, BOQAgent
from app.config import Settings, get_settings
from app.document.docling_processor import DoclingProcessor, DoclingResult
from app.document.evidence_builder import build_evidence_from_result
from app.errors import AIResponseError
from app.schemas.boq import BOQDocument, BOQItem
from app.schemas.evidence import EvidencePackage, EvidenceStatistics
from app.schemas.report import (
    ExtractionIssue,
    ExtractionReport,
    ExtractionStatistics,
    IssueType,
)
from app.validation.boq_validator import derive_status, failed_report, validate_boq

logger = logging.getLogger(__name__)


class EvidenceMetadata(BaseModel):
    """Provenance of the evidence a result was derived from.

    Carries the counts and document identity, never the evidence itself — a
    caller that wants the text and tables back should keep the
    `EvidencePackage`.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str
    page_count: int
    extracted_at: datetime
    evidence_version: str
    statistics: EvidenceStatistics
    docling_status: str | None = None
    docling_schema_version: str | None = None

    @classmethod
    def from_evidence(cls, evidence: EvidencePackage) -> EvidenceMetadata:
        document = evidence.document
        return cls(
            filename=document.filename,
            page_count=document.page_count,
            extracted_at=document.extracted_at,
            evidence_version=evidence.evidence_version,
            statistics=evidence.statistics,
            docling_status=document.docling_status,
            docling_schema_version=document.docling_schema_version,
        )


class LLMMetadata(BaseModel):
    """How the model was used. Diagnostics only — never part of the BOQ."""

    model_config = ConfigDict(extra="forbid")

    model: str
    attempts: int
    duration_seconds: float
    used_json_schema: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    unresolved_rows: int = 0

    @classmethod
    def from_agent_result(cls, result: AgentResult) -> LLMMetadata:
        return cls(
            model=result.model,
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            used_json_schema=result.used_json_schema,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            unresolved_rows=len(result.unresolved),
        )


@dataclass(frozen=True)
class BOQResult:
    """The outcome of one extraction run.

    `boq` is `None` when no canonical document could be assembled at all — the
    document held no BOQ, or the model never produced usable output. In every
    other case a `BOQDocument` is returned *together with* the report that says
    how much it can be trusted.

    `llm` is `None` when the model was never called (there was nothing to send).
    """

    boq: BOQDocument | None
    report: ExtractionReport
    evidence_metadata: EvidenceMetadata
    llm: LLMMetadata | None = None

    @property
    def items(self) -> list[BOQItem]:
        return self.boq.items if self.boq is not None else []

    @property
    def succeeded(self) -> bool:
        """True when a BOQ was produced and nothing blocking was found."""
        return self.boq is not None and self.report.is_valid


def run_document_stage(
    path: str | Path,
    *,
    processor: DoclingProcessor | None = None,
    settings: Settings | None = None,
) -> tuple[DoclingResult, EvidencePackage]:
    """PDF -> Docling -> evidence package.

    Raises `DocumentError` for unusable input and `ExtractionError` when Docling
    cannot produce a document.
    """
    processor = processor or DoclingProcessor(settings=settings or get_settings())
    result = processor.process(path)
    evidence = build_evidence_from_result(result)
    logger.info(
        "Evidence built for %s: %d pages, %d text blocks, %d tables, %d rows",
        evidence.document.filename,
        evidence.statistics.pages,
        evidence.statistics.text_blocks,
        evidence.statistics.tables,
        evidence.statistics.table_rows,
    )
    return result, evidence


def extract_boq_from_evidence(
    evidence: EvidencePackage,
    *,
    agent: BOQAgent | None = None,
    settings: Settings | None = None,
) -> BOQResult:
    """Evidence -> LLM restructuring -> canonical BOQ + extraction report."""
    settings = settings or get_settings()
    metadata = EvidenceMetadata.from_evidence(evidence)
    statistics = ExtractionStatistics(
        pages_processed=evidence.statistics.pages,
        tables_detected=evidence.statistics.tables,
    )

    agent = agent or BOQAgent(settings=settings)
    try:
        agent_result = agent.extract(evidence)
    except AIResponseError as exc:
        logger.error("Extraction failed for %s: %s", metadata.filename, exc)
        return BOQResult(
            boq=None,
            report=failed_report(
                [
                    ExtractionIssue(type=IssueType.LLM_INVALID_OUTPUT, message=str(exc)),
                    *exc.issues,
                ],
                statistics,
            ),
            evidence_metadata=metadata,
        )

    llm = LLMMetadata.from_agent_result(agent_result) if agent_result.called_llm else None

    # The agent reports document-level errors (no BOQ present) rather than
    # raising, so there is nothing to assemble but plenty to report.
    if agent_result.errors:
        return BOQResult(
            boq=None,
            report=failed_report(agent_result.errors, statistics),
            evidence_metadata=metadata,
            llm=llm,
        )

    boq = BOQDocument.assemble(agent_result.items, filename=evidence.document.filename)
    report = _validate(boq, statistics=statistics, agent_result=agent_result)

    logger.info("Extraction finished for %s — %s", metadata.filename, report.summary_line())
    return BOQResult(boq=boq, report=report, evidence_metadata=metadata, llm=llm)


def extract_boq_from_pdf(
    path: str | Path,
    *,
    processor: DoclingProcessor | None = None,
    agent: BOQAgent | None = None,
    settings: Settings | None = None,
) -> BOQResult:
    """Run the full pipeline on a PDF and return the canonical BOQ + report."""
    settings = settings or get_settings()
    _, evidence = run_document_stage(path, processor=processor, settings=settings)
    return extract_boq_from_evidence(evidence, agent=agent, settings=settings)


def _validate(
    boq: BOQDocument,
    *,
    statistics: ExtractionStatistics,
    agent_result: AgentResult,
) -> ExtractionReport:
    """Deterministic rules, plus the rows the agent refused to guess.

    An unresolved row is not a rule violation — the agent behaved correctly by
    not inventing it — but it still means the output is incomplete, so it lands
    in the report as a warning and can move a clean run to `partial`.
    """
    report = validate_boq(boq, statistics)
    if not agent_result.warnings:
        return report

    warnings = [*report.warnings, *agent_result.warnings]
    return report.model_copy(
        update={"warnings": warnings, "status": derive_status(report.errors, warnings)}
    )
