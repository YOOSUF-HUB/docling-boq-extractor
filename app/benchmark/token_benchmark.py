"""Measuring where an extraction request's tokens go.

    PDF ─┐
         ├─> EvidencePackage ─> build_user_prompt() ─> request ─> Groq usage
  evidence.json ─┘

Two modes:

* **static** (default) — everything that can be known without spending tokens:
  document shape, evidence size, the exact prompt the agent would send, and the
  budget that request would reserve. No API key needed.
* **live** (`live=True`) — additionally makes the real call through a recording
  client, so Groq's own `usage` figures, the finish reason, and the cost of any
  repair round are measured rather than predicted.

Nothing here changes extraction. The prompt is obtained by calling the agent's
own `build_user_prompt`, and a live run reaches the model through the agent
itself; the benchmark only watches.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.agent.boq_agent import BOQAgent
from app.agent.groq_client import GroqClient, LLMResponse
from app.agent.prompts import BOQ_RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.benchmark.metrics import (
    AttemptMetrics,
    BudgetMetrics,
    CompletionMetrics,
    CostCentre,
    DocumentMetrics,
    EvidenceMetrics,
    Measurement,
    PromptComponent,
    PromptMetrics,
    TokenBenchmark,
    TokenizerReport,
)
from app.benchmark.recorder import RecordingGroqClient
from app.benchmark.segments import SegmentKind, segment_texts, split_user_prompt
from app.benchmark.tokenizer import TextMeasurement, TokenCounter
from app.config import Settings, get_settings
from app.document.docling_processor import DoclingProcessor, DoclingResult
from app.errors import AIRequestError, BOQExtractionError
from app.schemas.evidence import EvidencePackage
from app.services.extraction_service import BOQResult, extract_boq_from_evidence, run_document_stage

logger = logging.getLogger(__name__)


class PromptDriftError(BOQExtractionError):
    """The benchmark's view of the prompt no longer matches the real one.

    Raised rather than worked around: attributing tokens to components that are
    not what the agent sends would produce a confident, wrong answer.
    """


# --- entry points -----------------------------------------------------


def load_evidence(path: str | Path) -> EvidencePackage:
    """Read an `evidence.json` written by the CLI."""
    return EvidencePackage.model_validate_json(Path(path).read_text(encoding="utf-8"))


def benchmark_pdf(
    path: str | Path,
    *,
    settings: Settings | None = None,
    counter: TokenCounter | None = None,
    processor: DoclingProcessor | None = None,
    live: bool = False,
    client: GroqClient | None = None,
    envelope_tokens: int | None = None,
) -> tuple[TokenBenchmark, EvidencePackage]:
    """Run the document stage, then measure. Returns the evidence for caching."""
    settings = settings or get_settings()
    path = Path(path)
    docling, evidence = run_document_stage(path, processor=processor, settings=settings)
    benchmark = benchmark_evidence(
        evidence,
        settings=settings,
        counter=counter,
        source=str(path),
        docling=docling,
        file_size_bytes=path.stat().st_size if path.exists() else None,
        live=live,
        client=client,
        envelope_tokens=envelope_tokens,
    )
    return benchmark, evidence


def benchmark_evidence(
    evidence: EvidencePackage,
    *,
    settings: Settings | None = None,
    counter: TokenCounter | None = None,
    source: str = "",
    docling: DoclingResult | None = None,
    file_size_bytes: int | None = None,
    live: bool = False,
    client: GroqClient | None = None,
    envelope_tokens: int | None = None,
) -> TokenBenchmark:
    """Measure the request this evidence package would produce."""
    settings = settings or get_settings()
    counter = counter or TokenCounter(settings.groq_model)

    user_prompt = build_user_prompt(evidence)
    user = counter.measure(user_prompt)
    system = counter.measure(SYSTEM_PROMPT)
    schema = counter.measure(_serialize_schema())
    components = _components(evidence, user_prompt, counter)

    completion = None
    if live:
        completion = _measure_live(
            evidence,
            settings=settings,
            counter=counter,
            client=client,
            user_prompt=user_prompt,
            counted_input_tokens=system.tokens + user.tokens + schema.tokens,
        )

    prompt = _prompt_metrics(
        system=system,
        user=user,
        schema=schema,
        completion=completion,
        supplied_envelope=envelope_tokens,
    )
    budget = _budget_metrics(prompt, completion, settings)

    return TokenBenchmark(
        measured_at=datetime.now(UTC),
        source=source or evidence.document.filename,
        mode="live" if live else "static",
        model=settings.groq_model,
        tokenizer=TokenizerReport.of(counter.info),
        document=_document_metrics(evidence, docling, file_size_bytes),
        evidence=_evidence_metrics(evidence, user, docling, counter),
        prompt=prompt,
        components=components,
        cost_centres=_cost_centres(prompt, components, budget),
        budget=budget,
        completion=completion,
    )


# --- static measurement ----------------------------------------------


def _serialize_schema() -> str:
    """The response schema as Groq receives it in the request body."""
    return json.dumps(BOQ_RESPONSE_SCHEMA, separators=(",", ":"))


def _document_metrics(
    evidence: EvidencePackage,
    docling: DoclingResult | None,
    file_size_bytes: int | None,
) -> DocumentMetrics:
    statistics = evidence.statistics
    return DocumentMetrics(
        filename=evidence.document.filename,
        file_size_bytes=file_size_bytes,
        page_count=evidence.document.page_count,
        table_count=statistics.tables,
        text_block_count=statistics.text_blocks,
        table_row_count=statistics.table_rows,
        picture_count=statistics.pictures,
        docling_duration_seconds=docling.duration_seconds if docling else None,
    )


def _evidence_metrics(
    evidence: EvidencePackage,
    rendered: TextMeasurement,
    docling: DoclingResult | None,
    counter: TokenCounter,
) -> EvidenceMetrics:
    compact = counter.measure(evidence.model_dump_json())
    indented = counter.measure(evidence.model_dump_json(indent=2))

    docling_json = None
    if docling is not None and docling.document_dict:
        docling_json = counter.measure(
            json.dumps(docling.document_dict, ensure_ascii=False, default=str)
        )

    saving = (1 - rendered.tokens / compact.tokens) * 100 if compact.tokens else 0.0
    return EvidenceMetrics(
        docling_document_json=Measurement.of(docling_json) if docling_json else None,
        serialized_compact=Measurement.of(compact),
        serialized_indented=Measurement.of(indented),
        rendered_prompt=Measurement.of(rendered),
        rendering_saving_percent=round(saving, 2),
    )


def _components(
    evidence: EvidencePackage,
    user_prompt: str,
    counter: TokenCounter,
) -> list[PromptComponent]:
    """Attribute the user prompt's tokens to its labelled parts."""
    segments = split_user_prompt(evidence)
    texts = segment_texts(segments)

    rebuilt = "".join(texts)
    if rebuilt != user_prompt:
        raise PromptDriftError(
            "The benchmark's prompt segmentation no longer reproduces "
            "build_user_prompt(). app/benchmark/segments.py must be updated to "
            "match app/agent/prompts.py before these numbers can be trusted."
        )

    token_counts = counter.attribute(user_prompt, texts)
    total = sum(token_counts)

    components: list[PromptComponent] = []
    for segment, text, tokens in zip(segments, texts, token_counts, strict=True):
        per_row = None
        if segment.row_count:
            per_row = round(tokens / segment.row_count, 2)
        components.append(
            PromptComponent(
                kind=segment.kind.value,
                label=segment.label,
                sequence=segment.sequence,
                page_number=segment.page_number,
                row_count=segment.row_count,
                characters=len(text),
                tokens=tokens,
                share_of_user_prompt_percent=round(tokens / total * 100, 2) if total else 0.0,
                tokens_per_row=per_row,
                scales_with_document=segment.kind.is_document_content,
            )
        )
    return components


def _prompt_metrics(
    *,
    system: TextMeasurement,
    user: TextMeasurement,
    schema: TextMeasurement,
    completion: CompletionMetrics | None,
    supplied_envelope: int | None,
) -> PromptMetrics:
    counted = system.tokens + user.tokens + schema.tokens

    envelope: int | None = None
    envelope_source = "not measured"
    if completion is not None and completion.prompt_tokens is not None:
        envelope = completion.prompt_tokens - counted
        envelope_source = "measured from this run's Groq usage"
    elif supplied_envelope is not None:
        envelope = supplied_envelope
        envelope_source = "supplied from an earlier live run"

    return PromptMetrics(
        system_prompt=Measurement.of(system),
        user_prompt=Measurement.of(user),
        response_schema=Measurement.of(schema),
        counted_input_tokens=counted,
        envelope_tokens=envelope,
        envelope_source=envelope_source,
        input_tokens=counted + (envelope or 0),
        input_is_lower_bound=envelope is None,
    )


def _budget_metrics(
    prompt: PromptMetrics,
    completion: CompletionMetrics | None,
    settings: Settings,
) -> BudgetMetrics:
    """What this request reserves against the tokens-per-minute limit."""
    charged = completion.prompt_tokens if completion else None
    input_tokens = charged if charged is not None else prompt.input_tokens

    reserved = settings.groq_max_completion_tokens
    requested = input_tokens + reserved
    limit = settings.groq_tpm_limit

    return BudgetMetrics(
        max_completion_tokens=reserved,
        tpm_limit=limit,
        requested_budget_tokens=requested,
        headroom_tokens=limit - requested,
        utilisation_percent=round(requested / limit * 100, 1),
        exceeds_limit=requested > limit,
        budget_is_lower_bound=charged is None and prompt.input_is_lower_bound,
    )


def _cost_centres(
    prompt: PromptMetrics,
    components: list[PromptComponent],
    budget: BudgetMetrics,
) -> list[CostCentre]:
    """Group tokens by what would have to change to reduce them."""

    def total(*kinds: SegmentKind) -> int:
        wanted = {kind.value for kind in kinds}
        return sum(part.tokens for part in components if part.kind in wanted)

    entries: list[tuple[str, int, bool, str]] = [
        (
            "system prompt",
            prompt.system_prompt.tokens,
            False,
            "Extraction rules. Identical on every request, of any size.",
        ),
        (
            "response schema",
            prompt.response_schema.tokens,
            False,
            "The strict json_schema sent as response_format.",
        ),
        (
            "prompt header + footer",
            total(SegmentKind.HEADER, SegmentKind.FOOTER),
            False,
            "Document facts and the closing instruction.",
        ),
        (
            "page markers",
            total(SegmentKind.PAGE_MARKER),
            True,
            "One `--- PAGE n ---` line per page transition.",
        ),
        (
            "text blocks",
            total(SegmentKind.TEXT_BLOCK),
            True,
            "Non-table text: headings, notes, page furniture.",
        ),
        (
            "table rows",
            total(SegmentKind.TABLE),
            True,
            "The BOQ itself — the only content the items come from.",
        ),
    ]

    if prompt.envelope_tokens:
        entries.append(
            (
                "chat envelope",
                prompt.envelope_tokens,
                False,
                f"Role markers and server-side preamble ({prompt.envelope_source}).",
            )
        )

    entries.append(
        (
            "reserved completion budget",
            budget.max_completion_tokens,
            False,
            "GROQ_MAX_COMPLETION_TOKENS, held against TPM whether used or not.",
        )
    )

    denominator = budget.requested_budget_tokens or 1
    return [
        CostCentre(
            name=name,
            tokens=tokens,
            share_percent=round(tokens / denominator * 100, 2),
            scales_with_document=scales,
            note=note,
        )
        for name, tokens, scales, note in entries
    ]


# --- live measurement -------------------------------------------------


def _measure_live(
    evidence: EvidencePackage,
    *,
    settings: Settings,
    counter: TokenCounter,
    client: GroqClient | None,
    user_prompt: str,
    counted_input_tokens: int,
) -> CompletionMetrics:
    """Run the real extraction through a recording client and report its usage."""
    recorder = RecordingGroqClient(inner=client or GroqClient(settings=settings))
    agent = BOQAgent(client=recorder, settings=settings)  # type: ignore[arg-type]

    result: BOQResult | None = None
    error: str | None = None
    try:
        result = extract_boq_from_evidence(evidence, agent=agent, settings=settings)
    except AIRequestError as exc:
        # The request failing *is* the measurement when it fails on size.
        logger.warning("Live benchmark request failed: %s", exc)
        error = str(exc)

    attempts = [
        AttemptMetrics(
            index=index,
            message_count=len(call.messages),
            sent_characters=call.sent_characters,
            sent_tokens=counter.count(call.sent_text),
            prompt_tokens=call.response.prompt_tokens if call.response else None,
            completion_tokens=call.response.completion_tokens if call.response else None,
            total_tokens=_sum_usage(call.response),
            finish_reason=call.response.finish_reason if call.response else None,
            duration_seconds=round(call.response.duration_seconds, 3) if call.response else 0.0,
            used_json_schema=call.response.used_json_schema if call.response else False,
        )
        for index, call in enumerate(recorder.calls, start=1)
    ]

    first = attempts[0] if attempts else None
    charged_totals = [attempt.total_tokens for attempt in attempts if attempt.total_tokens]

    estimate_error = None
    estimate_error_percent = None
    if first is not None and first.prompt_tokens:
        estimate_error = counted_input_tokens - first.prompt_tokens
        estimate_error_percent = round(estimate_error / first.prompt_tokens * 100, 2)

    llm = result.llm if result is not None else None
    return CompletionMetrics(
        model=(llm.model if llm else "") or recorder.model,
        attempts=len(attempts),
        finish_reason=first.finish_reason if first else None,
        prompt_tokens=first.prompt_tokens if first else None,
        completion_tokens=first.completion_tokens if first else None,
        total_tokens=first.total_tokens if first else None,
        total_tokens_all_attempts=sum(charged_totals) if charged_totals else None,
        llm_duration_seconds=round(llm.duration_seconds, 3) if llm else 0.0,
        items_extracted=len(result.items) if result else 0,
        unresolved_items=llm.unresolved_rows if llm else 0,
        validation_status=result.report.status.value if result else None,
        per_attempt=attempts,
        error=error or _recorded_error(recorder),
        request_matches_measurement=_request_matches(recorder, user_prompt),
        estimate_error_tokens=estimate_error,
        estimate_error_percent=estimate_error_percent,
    )


def _sum_usage(response: LLMResponse | None) -> int | None:
    if response is None or response.prompt_tokens is None or response.completion_tokens is None:
        return None
    return response.prompt_tokens + response.completion_tokens


def _recorded_error(recorder: RecordingGroqClient) -> str | None:
    return next((call.error for call in recorder.calls if call.error), None)


def _request_matches(recorder: RecordingGroqClient, user_prompt: str) -> bool:
    """Did the agent send exactly the prompt this benchmark measured?"""
    call = recorder.first_call
    if call is None:
        return False
    contents = [message.get("content", "") for message in call.messages]
    return contents == [SYSTEM_PROMPT, user_prompt]
