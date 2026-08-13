"""End-to-end pipeline tests with a stubbed agent — no network, no models.

Docling is exercised for real in `tests/test_docling.py` (`-m docling`); here the
document stage is fed a `DoclingResult` built from captured Docling output, so
the orchestration itself can be tested in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.boq_agent import AgentResult
from app.config import Settings
from app.document.docling_processor import DoclingResult, DoclingSummary
from app.errors import AIRequestError, AIResponseError, ConfigurationError, DocumentError
from app.schemas.boq import BOQItem, UnresolvedItem
from app.schemas.evidence import EvidencePackage
from app.schemas.report import ExtractionIssue, ExtractionStatus, IssueType
from app.services.extraction_service import (
    BOQResult,
    EvidenceMetadata,
    LLMMetadata,
    extract_boq_from_evidence,
    extract_boq_from_pdf,
    run_document_stage,
)
from tests.factories import load_sample_boq_document

FIXTURES = Path(__file__).parent / "fixtures"

SETTINGS = Settings(groq_api_key="test-key")


@pytest.fixture(scope="module")
def evidence() -> EvidencePackage:
    return EvidencePackage.model_validate_json(
        (FIXTURES / "evidence_sample_boq.json").read_text()
    )


def make_item(**overrides) -> BOQItem:
    payload = {
        "level_path": ["EARTHWORKS"],
        "boq_item_code": "1.1",
        "boq_item_name": "Excavation for foundations",
        "quantity": 125.5,
        "unit": "m3",
    } | overrides
    return BOQItem.model_validate(payload)


def agent_result(items=None, **overrides) -> AgentResult:
    defaults = {
        "items": items if items is not None else [make_item()],
        "model": "openai/gpt-oss-120b",
        "attempts": 1,
        "duration_seconds": 1.5,
        "prompt_tokens": 1800,
        "completion_tokens": 1400,
        "used_json_schema": True,
    }
    return AgentResult(**(defaults | overrides))


class StubAgent:
    """Stands in for `BOQAgent`: returns a canned result or raises."""

    def __init__(self, result: AgentResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[EvidencePackage] = []

    def extract(self, evidence: EvidencePackage) -> AgentResult:
        self.calls.append(evidence)
        if self.error is not None:
            raise self.error
        return self.result or agent_result()


def extract(evidence: EvidencePackage, agent: StubAgent) -> BOQResult:
    return extract_boq_from_evidence(evidence, agent=agent, settings=SETTINGS)


# --- happy path -------------------------------------------------------


def test_pipeline_produces_a_canonical_boq(evidence):
    result = extract(evidence, StubAgent())

    assert result.succeeded is True
    assert result.report.status is ExtractionStatus.SUCCESS
    assert [item.boq_item_code for item in result.items] == ["1.1"]


def test_document_metadata_comes_from_python_not_the_model(evidence):
    result = extract(evidence, StubAgent())

    assert result.boq.document.filename == evidence.document.filename
    assert result.boq.document.extracted_at is not None
    assert result.boq.boq_source_version == "1.0"


def test_evidence_metadata_describes_the_source_document(evidence):
    result = extract(evidence, StubAgent())
    metadata = result.evidence_metadata

    assert metadata.filename == evidence.document.filename
    assert metadata.page_count == evidence.document.page_count
    assert metadata.evidence_version == evidence.evidence_version
    assert metadata.statistics == evidence.statistics


def test_llm_metadata_is_reported(evidence):
    result = extract(evidence, StubAgent())

    assert result.llm == LLMMetadata(
        model="openai/gpt-oss-120b",
        attempts=1,
        duration_seconds=1.5,
        used_json_schema=True,
        prompt_tokens=1800,
        completion_tokens=1400,
        unresolved_rows=0,
    )


def test_statistics_combine_evidence_and_validation_counts(evidence):
    items = [
        make_item(),
        make_item(boq_item_code="2.1", level_path=["CONCRETE WORKS"]),
    ]
    result = extract(evidence, StubAgent(agent_result(items)))
    statistics = result.report.statistics

    assert statistics.pages_processed == evidence.statistics.pages
    assert statistics.tables_detected == evidence.statistics.tables
    assert statistics.items_extracted == 2
    assert statistics.sections_detected == 2


def test_the_evidence_package_is_what_reaches_the_agent(evidence):
    agent = StubAgent()

    extract(evidence, agent)

    assert agent.calls == [evidence]


# --- unresolved rows --------------------------------------------------


def test_unresolved_rows_become_warnings_and_downgrade_the_status(evidence):
    unresolved = [UnresolvedItem(boq_item_code="2.2", missing_fields=["unit"], reason="column blank")]
    result = extract(
        evidence,
        StubAgent(
            agent_result(
                unresolved=unresolved,
                warnings=[
                    ExtractionIssue(
                        type=IssueType.MISSING_UNIT,
                        message="unit could not be confidently identified for '2.2'",
                        item_code="2.2",
                        field="unit",
                    )
                ],
            )
        ),
    )

    assert result.report.status is ExtractionStatus.PARTIAL
    assert result.report.errors == []
    assert [issue.type for issue in result.report.warnings] == [IssueType.MISSING_UNIT]
    # A partial result is still a result: the items that were extracted survive.
    assert result.boq is not None
    assert result.llm.unresolved_rows == 1


def test_agent_warnings_do_not_displace_rule_warnings(evidence):
    result = extract(
        evidence,
        StubAgent(
            agent_result(
                [make_item(quantity=0)],
                warnings=[
                    ExtractionIssue(type=IssueType.UNRESOLVED_ROW, message="row 9 unreadable")
                ],
            )
        ),
    )

    assert {issue.type for issue in result.report.warnings} == {
        IssueType.INVALID_QUANTITY,
        IssueType.UNRESOLVED_ROW,
    }


# --- failures ---------------------------------------------------------


def test_no_boq_detected_is_reported_without_a_document(evidence):
    errors = [
        ExtractionIssue(type=IssueType.NO_BOQ_DETECTED, message="nothing to interpret")
    ]
    result = extract(evidence, StubAgent(agent_result(errors=errors, attempts=0, model="")))

    assert result.boq is None
    assert result.succeeded is False
    assert result.report.status is ExtractionStatus.FAILED
    assert [issue.type for issue in result.report.errors] == [IssueType.NO_BOQ_DETECTED]
    # The model was never called, so there is nothing to report about it.
    assert result.llm is None


def test_invalid_model_output_is_reported_not_raised(evidence):
    detail = ExtractionIssue(type=IssueType.MISSING_UNIT, message="items.0.unit: field required")
    agent = StubAgent(error=AIResponseError("no schema-valid output after 2 attempts", [detail]))

    result = extract(evidence, agent)

    assert result.boq is None
    assert result.report.status is ExtractionStatus.FAILED
    assert result.report.errors[0].type is IssueType.LLM_INVALID_OUTPUT
    assert "2 attempts" in result.report.errors[0].message
    # The per-field detail is kept, so the report says *what* was wrong.
    assert result.report.errors[1] == detail


def test_an_unreachable_model_raises_rather_than_reporting(evidence):
    """No answer arrived, so there is no judgement to make about the document."""
    agent = StubAgent(error=AIRequestError("connection refused"))

    with pytest.raises(AIRequestError):
        extract(evidence, agent)


def test_a_missing_api_key_raises(evidence):
    agent = StubAgent(error=ConfigurationError("GROQ_API_KEY is not set"))

    with pytest.raises(ConfigurationError):
        extract(evidence, agent)


def test_rule_violations_fail_the_report_but_keep_the_items(evidence):
    items = [make_item(), make_item(boq_item_name="Total")]
    result = extract(evidence, StubAgent(agent_result(items)))

    assert result.report.status is ExtractionStatus.FAILED
    assert result.succeeded is False
    assert IssueType.TOTAL_ROW_AS_ITEM in {issue.type for issue in result.report.errors}
    # Kept deliberately: a human needs to see what was extracted to fix it.
    assert len(result.items) == 2


def test_zero_items_is_a_failure(evidence):
    result = extract(evidence, StubAgent(agent_result([])))

    assert result.report.status is ExtractionStatus.FAILED
    assert [issue.type for issue in result.report.errors] == [IssueType.NO_ITEMS_EXTRACTED]
    assert result.items == []


# --- document stage ---------------------------------------------------


class StubProcessor:
    """Returns a `DoclingResult` built from captured Docling output."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.paths: list[object] = []

    def process(self, path) -> DoclingResult:
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        document = load_sample_boq_document()
        return DoclingResult(
            source_path=Path(path),
            filename=Path(path).name,
            status="SUCCESS",
            markdown=document.export_to_markdown(),
            document_dict=document.export_to_dict(),
            summary=DoclingSummary(
                page_count=1, table_count=1, text_block_count=4, picture_count=0
            ),
            duration_seconds=0.4,
            document=document,
        )


def test_document_stage_builds_evidence_from_the_pdf():
    docling_result, evidence = run_document_stage(
        "data/input/sample_boq.pdf", processor=StubProcessor(), settings=SETTINGS
    )

    assert docling_result.filename == "sample_boq.pdf"
    assert evidence.document.filename == "sample_boq.pdf"
    assert evidence.has_tables is True


def test_document_errors_propagate():
    with pytest.raises(DocumentError):
        run_document_stage(
            "missing.pdf",
            processor=StubProcessor(error=DocumentError("not a PDF")),
            settings=SETTINGS,
        )


def test_extract_boq_from_pdf_runs_both_halves():
    result = extract_boq_from_pdf(
        "data/input/sample_boq.pdf",
        processor=StubProcessor(),
        agent=StubAgent(),
        settings=SETTINGS,
    )

    assert result.succeeded is True
    assert result.boq.document.filename == "sample_boq.pdf"
    assert result.evidence_metadata.filename == "sample_boq.pdf"


# --- output contract --------------------------------------------------


def test_the_boq_json_carries_no_pms_or_derived_fields(evidence):
    result = extract(evidence, StubAgent())
    payload = json.loads(result.boq.model_dump_json())

    assert set(payload) == {"boq_source_version", "document", "items"}
    assert set(payload["document"]) == {"filename", "extracted_at"}
    for forbidden in ("id", "temp_id", "boq_id", "amount", "total", "parent_id"):
        assert forbidden not in payload["items"][0]


def test_the_report_is_not_embedded_in_the_boq(evidence):
    result = extract(evidence, StubAgent())

    assert "report" not in json.loads(result.boq.model_dump_json())


def test_evidence_metadata_rejects_unknown_fields():
    with pytest.raises(ValueError):
        EvidenceMetadata(
            filename="x.pdf",
            page_count=1,
            extracted_at="2026-08-12T09:14:00Z",
            evidence_version="1.0",
            statistics={},
            confidence=0.94,
        )
