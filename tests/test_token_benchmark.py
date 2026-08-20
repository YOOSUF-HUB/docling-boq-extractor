"""Tests for the token-usage benchmark.

No network and no models: the tokenizer runs locally, evidence comes from the
captured fixture, and live-mode tests use a fake Groq client.

Two properties matter more than any individual number here:

* **Fidelity** — the benchmark must measure the prompt the agent actually
  sends, not a copy that has drifted from it.
* **Additivity** — the per-component counts must sum to the whole, or the
  "where do the tokens go" breakdown is decoration rather than measurement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent.groq_client import LLMResponse
from app.agent.prompts import BOQ_RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.benchmark import __main__ as cli
from app.benchmark import report as report_renderer
from app.benchmark.metrics import TokenBenchmark
from app.benchmark.recorder import RecordingGroqClient
from app.benchmark.segments import SegmentKind, segment_texts, split_user_prompt
from app.benchmark.token_benchmark import PromptDriftError, benchmark_evidence, load_evidence
from app.benchmark.tokenizer import (
    TokenCounter,
    TokenizerUnavailableError,
    encoding_for_model,
)
from app.config import Settings
from app.errors import AIRequestError
from app.schemas.evidence import (
    DocumentEvidence,
    EvidencePackage,
    EvidenceStatistics,
    PageEvidence,
    TableEvidence,
    TableRowEvidence,
    TextBlockEvidence,
)

FIXTURES = Path(__file__).parent / "fixtures"
MODEL = "openai/gpt-oss-120b"
SETTINGS = Settings(groq_api_key="test-key", groq_model=MODEL, groq_tpm_limit=8000)


@pytest.fixture(scope="module")
def counter() -> TokenCounter:
    return TokenCounter(MODEL)


@pytest.fixture(scope="module")
def evidence() -> EvidencePackage:
    return load_evidence(FIXTURES / "evidence_sample_boq.json")


# --- evidence builders ------------------------------------------------


def make_evidence(
    *,
    text_blocks: list[TextBlockEvidence] | None = None,
    tables: list[TableEvidence] | None = None,
    page_count: int = 1,
    filename: str = "synthetic.pdf",
) -> EvidencePackage:
    text_blocks = text_blocks or []
    tables = tables or []
    return EvidencePackage(
        document=DocumentEvidence(
            filename=filename,
            page_count=page_count,
            extracted_at=datetime(2026, 8, 20, tzinfo=UTC),
        ),
        pages=[PageEvidence(page_number=number) for number in range(1, page_count + 1)],
        text_blocks=text_blocks,
        tables=tables,
        statistics=EvidenceStatistics(
            pages=page_count,
            text_blocks=len(text_blocks),
            tables=len(tables),
            table_rows=sum(len(table.rows) for table in tables),
        ),
    )


def make_text_block(sequence: int, text: str, page: int | None = 1) -> TextBlockEvidence:
    return TextBlockEvidence(
        sequence=sequence,
        ref=f"#/texts/{sequence}",
        page_number=page,
        label="section_header",
        content_layer="body",
        text=text,
    )


def make_table(
    sequence: int,
    *,
    index: int = 0,
    page: int | None = 1,
    rows: int = 3,
    caption: str | None = None,
) -> TableEvidence:
    return TableEvidence(
        sequence=sequence,
        ref=f"#/tables/{index}",
        table_index=index,
        page_number=page,
        num_rows=rows,
        num_cols=4,
        caption=caption,
        rows=[
            TableRowEvidence(
                row_index=row,
                is_header=row == 0,
                cells=[f"{row}.1", f"Item description {row}", "m3", f"{row * 12.5}"],
            )
            for row in range(rows)
        ],
    )


EVIDENCE_CASES = {
    "empty": make_evidence(),
    "text_only": make_evidence(text_blocks=[make_text_block(0, "EARTHWORKS")]),
    "table_only": make_evidence(tables=[make_table(0)]),
    "interleaved": make_evidence(
        text_blocks=[make_text_block(0, "EARTHWORKS"), make_text_block(2, "CONCRETE", page=2)],
        tables=[make_table(1), make_table(3, index=1, page=2, caption="Bill 2")],
        page_count=2,
    ),
    "unknown_page": make_evidence(
        text_blocks=[make_text_block(0, "No provenance", page=None)],
        tables=[make_table(1, page=None)],
    ),
    "unicode": make_evidence(text_blocks=[make_text_block(0, "Ærø — 25 m² × 3 ✓")]),
}


# --- tokenizer --------------------------------------------------------


def test_gpt_oss_uses_its_own_encoding():
    encoding, is_native = encoding_for_model("openai/gpt-oss-120b")

    assert encoding == "o200k_harmony"
    assert is_native is True


def test_an_unknown_model_falls_back_and_says_so():
    encoding, is_native = encoding_for_model("some-other/model-v9")

    assert encoding == "o200k_base"
    assert is_native is False


def test_the_fallback_is_flagged_in_the_report(counter):
    proxy = TokenCounter("some-other/model-v9")

    assert proxy.info.is_native is False
    assert "proxy" in proxy.info.note
    assert counter.info.is_native is True


def test_token_counts_are_not_a_character_heuristic(counter):
    """A BOQ table row tokenizes far denser than the len/4 rule of thumb."""
    row = "  row 3 | 2.1 | Reinforced concrete foundations | m3 | 50.00 | 12,500.00"

    tokens = counter.count(row)

    assert tokens > len(row) / 4
    assert counter.measure(row).chars_per_token < 4.0


def test_document_text_can_contain_tokenizer_control_strings(counter):
    """PDF text is untrusted input; `<|endoftext|>` must count, not explode."""
    assert counter.count("Item 1 <|endoftext|> Item 2") > 0


def test_attribution_sums_exactly_to_the_whole(counter, evidence):
    prompt = build_user_prompt(evidence)
    texts = segment_texts(split_user_prompt(evidence))

    assert sum(counter.attribute(prompt, texts)) == counter.count(prompt)


def test_attribution_rejects_segments_that_are_not_the_text(counter):
    with pytest.raises(ValueError):
        counter.attribute("hello world", ["hello", "planet"])


def test_a_missing_tokenizer_is_an_error_not_an_estimate():
    with pytest.raises(TokenizerUnavailableError):
        TokenCounter(MODEL, encoding="no_such_encoding")


# --- prompt segmentation ----------------------------------------------


@pytest.mark.parametrize("name", sorted(EVIDENCE_CASES))
def test_segments_reproduce_the_real_prompt_exactly(name):
    """The drift guard: segmentation must rebuild `build_user_prompt` byte for byte."""
    package = EVIDENCE_CASES[name]

    rebuilt = "".join(segment_texts(split_user_prompt(package)))

    assert rebuilt == build_user_prompt(package)


def test_segments_reproduce_the_prompt_for_captured_docling_output(evidence):
    assert "".join(segment_texts(split_user_prompt(evidence))) == build_user_prompt(evidence)


def test_segment_kinds_describe_the_prompt():
    kinds = [segment.kind for segment in split_user_prompt(EVIDENCE_CASES["interleaved"])]

    assert kinds[0] is SegmentKind.HEADER
    assert kinds[-1] is SegmentKind.FOOTER
    assert kinds.count(SegmentKind.TABLE) == 2
    assert kinds.count(SegmentKind.PAGE_MARKER) == 2


def test_only_document_content_scales():
    assert SegmentKind.TABLE.is_document_content is True
    assert SegmentKind.TEXT_BLOCK.is_document_content is True
    assert SegmentKind.HEADER.is_document_content is False
    assert SegmentKind.FOOTER.is_document_content is False


# --- static benchmark -------------------------------------------------


def measure(package: EvidencePackage, counter: TokenCounter, **kwargs) -> TokenBenchmark:
    return benchmark_evidence(package, settings=SETTINGS, counter=counter, **kwargs)


def test_document_metrics_come_from_the_evidence(counter, evidence):
    result = measure(evidence, counter)

    assert result.document.filename == evidence.document.filename
    assert result.document.page_count == evidence.document.page_count
    assert result.document.table_count == evidence.statistics.tables
    assert result.document.table_row_count == evidence.statistics.table_rows


def test_the_prompt_components_are_measured_separately(counter, evidence):
    result = measure(evidence, counter)

    assert result.prompt.system_prompt.tokens == counter.count(SYSTEM_PROMPT)
    assert result.prompt.user_prompt.tokens == counter.count(build_user_prompt(evidence))
    assert result.prompt.response_schema.tokens == counter.count(
        json.dumps(BOQ_RESPONSE_SCHEMA, separators=(",", ":"))
    )


def test_counted_input_is_the_sum_of_its_parts(counter, evidence):
    prompt = measure(evidence, counter).prompt

    assert prompt.counted_input_tokens == (
        prompt.system_prompt.tokens + prompt.user_prompt.tokens + prompt.response_schema.tokens
    )


def test_static_mode_reports_a_lower_bound(counter, evidence):
    result = measure(evidence, counter)

    assert result.mode == "static"
    assert result.completion is None
    assert result.prompt.envelope_tokens is None
    assert result.prompt.input_is_lower_bound is True
    assert result.budget.budget_is_lower_bound is True


def test_components_account_for_every_prompt_token(counter, evidence):
    result = measure(evidence, counter)

    assert sum(part.tokens for part in result.components) == result.prompt.user_prompt.tokens


def test_cost_centres_account_for_the_whole_request(counter, evidence):
    result = measure(evidence, counter)

    assert sum(centre.tokens for centre in result.cost_centres) == (
        result.budget.requested_budget_tokens
    )


def test_the_system_prompt_and_schema_are_fixed_costs(counter, evidence):
    result = measure(evidence, counter)
    fixed = {centre.name for centre in result.cost_centres if not centre.scales_with_document}

    assert {"system prompt", "response schema", "reserved completion budget"} <= fixed


def test_the_completion_budget_is_reserved_whether_used_or_not(counter, evidence):
    result = measure(evidence, counter)

    assert result.budget.requested_budget_tokens == (
        result.prompt.input_tokens + SETTINGS.groq_max_completion_tokens
    )


def test_a_small_document_fits_the_limit(counter, evidence):
    result = measure(evidence, counter)

    assert result.budget.exceeds_limit is False
    assert result.budget.headroom_tokens > 0


def test_a_large_document_exceeds_the_limit(counter):
    large = make_evidence(tables=[make_table(0, rows=400)], page_count=20)

    result = measure(large, counter)

    assert result.budget.exceeds_limit is True
    assert result.budget.headroom_tokens < 0


def test_tables_dominate_a_boq_prompt(counter):
    package = make_evidence(
        text_blocks=[make_text_block(0, "EARTHWORKS")],
        tables=[make_table(1, rows=60)],
    )

    result = measure(package, counter)
    tables = sum(p.tokens for p in result.components if p.kind == "table")

    assert tables / result.prompt.user_prompt.tokens > 0.8


def test_per_row_cost_is_reported(counter):
    result = measure(make_evidence(tables=[make_table(0, rows=20)]), counter)
    table = next(part for part in result.components if part.kind == "table")

    assert table.row_count == 20
    assert table.tokens_per_row == pytest.approx(table.tokens / 20, abs=0.01)


def test_the_renderer_saving_against_raw_evidence_json_is_reported(counter, evidence):
    result = measure(evidence, counter)

    assert result.evidence.rendering_saving_percent > 0
    assert result.evidence.rendered_prompt.tokens < result.evidence.serialized_compact.tokens


def test_the_tokenizer_used_is_recorded(counter, evidence):
    tokenizer = measure(evidence, counter).tokenizer

    assert tokenizer.encoding == "o200k_harmony"
    assert tokenizer.library == "tiktoken"
    assert tokenizer.is_native is True


def test_an_envelope_from_an_earlier_run_can_be_applied(counter, evidence):
    result = measure(evidence, counter, envelope_tokens=120)

    assert result.prompt.envelope_tokens == 120
    assert result.prompt.input_is_lower_bound is False
    assert result.prompt.input_tokens == result.prompt.counted_input_tokens + 120
    assert "earlier live run" in result.prompt.envelope_source


def test_prompt_drift_is_refused_rather_than_guessed(counter, evidence, monkeypatch):
    """If the real prompt changes and segmentation does not, say so loudly."""
    monkeypatch.setattr(
        "app.benchmark.token_benchmark.build_user_prompt",
        lambda package: "a completely different prompt",
    )

    with pytest.raises(PromptDriftError):
        measure(evidence, counter)


# --- live benchmark ---------------------------------------------------


VALID_ITEM = {
    "level_path": ["EARTHWORKS"],
    "boq_item_code": "1.1",
    "boq_item_name": "Excavation for foundations",
    "quantity": 125.5,
    "unit": "m3",
}
VALID_RESPONSE = json.dumps({"items": [VALID_ITEM], "unresolved": []})


def llm_response(content: str = VALID_RESPONSE, **overrides) -> LLMResponse:
    defaults = {
        "content": content,
        "model": MODEL,
        "finish_reason": "stop",
        "prompt_tokens": 2400,
        "completion_tokens": 310,
        "duration_seconds": 1.25,
        "used_json_schema": True,
    }
    return LLMResponse(**(defaults | overrides))


class FakeGroqClient:
    """Stands in for `GroqClient`, returning canned responses in order."""

    model = MODEL

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses) or [llm_response()]
        self.received: list[list[dict[str, str]]] = []

    def complete_json(self, messages, *, json_schema=None, schema_name="response"):
        self.received.append(messages)
        answer = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_live_mode_reports_what_groq_charged(counter, evidence):
    result = measure(evidence, counter, live=True, client=FakeGroqClient())

    assert result.mode == "live"
    assert result.completion is not None
    assert result.completion.prompt_tokens == 2400
    assert result.completion.completion_tokens == 310
    assert result.completion.total_tokens == 2710
    assert result.completion.finish_reason == "stop"
    assert result.completion.model == MODEL


def test_live_mode_measures_the_chat_envelope(counter, evidence):
    result = measure(evidence, counter, live=True, client=FakeGroqClient())

    assert result.prompt.envelope_tokens == 2400 - result.prompt.counted_input_tokens
    assert result.prompt.input_is_lower_bound is False
    assert result.prompt.input_tokens == 2400
    assert "measured" in result.prompt.envelope_source


def test_the_budget_uses_charged_tokens_when_they_exist(counter, evidence):
    result = measure(evidence, counter, live=True, client=FakeGroqClient())

    assert result.budget.budget_is_lower_bound is False
    assert result.budget.requested_budget_tokens == 2400 + SETTINGS.groq_max_completion_tokens


def test_the_benchmark_measures_the_request_that_was_actually_sent(counter, evidence):
    """Fidelity: the agent's messages must be the ones the report describes."""
    client = FakeGroqClient()

    result = measure(evidence, counter, live=True, client=client)

    sent = [message["content"] for message in client.received[0]]
    assert sent == [SYSTEM_PROMPT, build_user_prompt(evidence)]
    assert result.completion.request_matches_measurement is True


def test_extraction_results_are_reported_alongside_the_tokens(counter, evidence):
    result = measure(evidence, counter, live=True, client=FakeGroqClient())

    assert result.completion.items_extracted == 1
    assert result.completion.validation_status == "success"
    assert result.completion.unresolved_items == 0


def test_a_repair_round_is_measured_as_a_second_attempt(counter, evidence):
    client = FakeGroqClient(llm_response("not json at all"), llm_response())

    result = measure(evidence, counter, live=True, client=client)
    completion = result.completion

    assert completion.attempts == 2
    assert len(completion.per_attempt) == 2
    # The repair round resends the conversation plus the rejected output.
    assert completion.per_attempt[1].message_count == 4
    assert completion.per_attempt[1].sent_tokens > completion.per_attempt[0].sent_tokens
    assert completion.total_tokens_all_attempts == 2710 * 2


def test_the_baseline_figures_describe_the_first_attempt(counter, evidence):
    client = FakeGroqClient(llm_response("not json at all"), llm_response(prompt_tokens=9000))

    completion = measure(evidence, counter, live=True, client=client).completion

    assert completion.prompt_tokens == 2400
    assert completion.total_tokens_all_attempts == 2710 + 9310


def test_a_rejected_request_is_recorded_not_raised(counter, evidence):
    """A 413 for an oversized request is the measurement, not an interruption."""
    client = FakeGroqClient(AIRequestError("Groq API error (413): Request too large"))

    result = measure(evidence, counter, live=True, client=client)

    assert result.completion.error is not None
    assert "413" in result.completion.error
    assert result.completion.prompt_tokens is None
    assert result.budget.budget_is_lower_bound is True


def test_counted_input_is_reconciled_against_charged_tokens(counter, evidence):
    completion = measure(evidence, counter, live=True, client=FakeGroqClient()).completion

    assert completion.estimate_error_tokens is not None
    assert completion.estimate_error_percent is not None


# --- recorder ---------------------------------------------------------


def test_the_recorder_does_not_alter_the_request():
    client = FakeGroqClient()
    recorder = RecordingGroqClient(inner=client)
    messages = [{"role": "user", "content": "hello"}]

    response = recorder.complete_json(messages, json_schema={"type": "object"})

    assert response.content == VALID_RESPONSE
    assert client.received == [messages]
    assert recorder.calls[0].messages == messages
    assert recorder.calls[0].used_json_schema_request is True


def test_the_recorder_records_failures_and_re_raises():
    recorder = RecordingGroqClient(inner=FakeGroqClient(AIRequestError("boom")))

    with pytest.raises(AIRequestError):
        recorder.complete_json([{"role": "user", "content": "hi"}])

    assert recorder.calls[0].error is not None
    assert "boom" in recorder.calls[0].error


def test_recorded_messages_are_a_snapshot():
    """Mutating the conversation afterwards must not rewrite history."""
    recorder = RecordingGroqClient(inner=FakeGroqClient())
    messages = [{"role": "user", "content": "original"}]

    recorder.complete_json(messages)
    messages[0]["content"] = "changed"

    assert recorder.calls[0].messages[0]["content"] == "original"


# --- report rendering -------------------------------------------------


def test_the_report_shows_every_section(counter, evidence):
    text = report_renderer.render(measure(evidence, counter))

    for section in (
        "TOKEN BENCHMARK",
        "DOCUMENT",
        "EVIDENCE FUNNEL",
        "REQUEST INPUT",
        "WHERE THE TOKENS GO",
        "HEAVIEST PROMPT COMPONENTS",
        "RATE-LIMIT BUDGET",
    ):
        assert f"========== {section}" in text


def test_a_static_report_never_presents_a_floor_as_a_fact(counter, evidence):
    text = report_renderer.render(measure(evidence, counter))

    assert "at least" in text
    assert "LIVE USAGE" not in text


def test_a_live_report_adds_the_charged_figures(counter, evidence):
    text = report_renderer.render(measure(evidence, counter, live=True, client=FakeGroqClient()))

    assert "LIVE USAGE (charged by Groq)" in text
    assert "at least" not in text


def test_the_comparison_table_needs_more_than_one_document(counter, evidence):
    single = [measure(evidence, counter)]

    assert report_renderer.render_comparison(single) == ""


def test_the_comparison_table_reports_marginal_cost(counter):
    small = measure(make_evidence(tables=[make_table(0, rows=10)]), counter)
    large = measure(make_evidence(tables=[make_table(0, rows=200)], page_count=8), counter)

    text = report_renderer.render_comparison([small, large])

    assert "CORPUS COMPARISON" in text
    assert "tokens per table row" in text
    assert "OVER" in text


# --- the record itself ------------------------------------------------


def test_the_record_round_trips_as_json(counter, evidence):
    result = measure(evidence, counter, live=True, client=FakeGroqClient())

    restored = TokenBenchmark.model_validate_json(result.model_dump_json())

    assert restored == result


def test_the_record_rejects_unknown_fields(counter, evidence):
    payload = json.loads(measure(evidence, counter).model_dump_json())
    payload["confidence"] = 0.94

    with pytest.raises(ValueError):
        TokenBenchmark.model_validate(payload)


def test_the_benchmark_makes_no_call_in_static_mode(counter, evidence, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("static mode must not construct a Groq client")

    monkeypatch.setattr("app.benchmark.token_benchmark.GroqClient", fail)

    assert measure(evidence, counter).completion is None


# --- CLI --------------------------------------------------------------


def write_evidence(directory: Path, package: EvidencePackage, name: str) -> Path:
    path = directory / name
    path.write_text(package.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_cli_measures_an_evidence_file_and_writes_the_record(tmp_path, capsys):
    source = write_evidence(tmp_path, EVIDENCE_CASES["interleaved"], "evidence.json")

    code = cli.main(["--evidence", str(source), "--output-dir", str(tmp_path / "out")])
    printed = capsys.readouterr().out

    assert code == cli.EXIT_OK
    assert "========== WHERE THE TOKENS GO ==========" in printed
    record = json.loads((tmp_path / "out" / "evidence.benchmark.json").read_text())
    assert record["mode"] == "static"
    assert record["tokenizer"]["encoding"] == "o200k_harmony"


def test_cli_no_write_leaves_nothing_behind(tmp_path, capsys):
    source = write_evidence(tmp_path, EVIDENCE_CASES["table_only"], "evidence.json")
    output = tmp_path / "out"

    code = cli.main(["--evidence", str(source), "--output-dir", str(output), "--no-write"])

    assert code == cli.EXIT_OK
    assert not output.exists()


def test_cli_can_fail_the_run_when_a_request_would_not_fit(tmp_path, capsys):
    big = make_evidence(tables=[make_table(0, rows=400)], page_count=20)
    source = write_evidence(tmp_path, big, "big.json")

    code = cli.main(
        ["--evidence", str(source), "--no-write", "--fail-on-exceed", "--top", "0"]
    )

    assert code == cli.EXIT_BUDGET_EXCEEDED
    assert "EXCEEDS the limit" in capsys.readouterr().out


def test_cli_reports_the_same_run_as_ok_without_the_flag(tmp_path):
    big = make_evidence(tables=[make_table(0, rows=400)], page_count=20)
    source = write_evidence(tmp_path, big, "big.json")

    assert cli.main(["--evidence", str(source), "--no-write"]) == cli.EXIT_OK


def test_cli_overrides_the_limits_it_checks_against(tmp_path, capsys):
    source = write_evidence(tmp_path, EVIDENCE_CASES["table_only"], "evidence.json")

    cli.main(
        [
            "--evidence",
            str(source),
            "--no-write",
            "--tpm-limit",
            "30000",
            "--max-completion-tokens",
            "1000",
        ]
    )
    printed = capsys.readouterr().out

    assert "30,000 tokens" in printed
    assert "Reserved for completion       : 1,000 tokens" in printed


def test_cli_compares_multiple_documents(tmp_path, capsys):
    small = write_evidence(tmp_path, make_evidence(tables=[make_table(0, rows=8)]), "small.json")
    large = write_evidence(
        tmp_path, make_evidence(tables=[make_table(0, rows=250)], page_count=9), "large.json"
    )

    code = cli.main(
        ["--evidence", str(small), "--evidence", str(large), "--no-write", "--top", "0"]
    )
    printed = capsys.readouterr().out

    assert code == cli.EXIT_OK
    assert "========== CORPUS COMPARISON ==========" in printed
    assert "tokens per table row" in printed


def test_cli_reports_an_unreadable_input(tmp_path, capsys):
    assert cli.main(["--evidence", str(tmp_path / "nope.json"), "--no-write"]) == (
        cli.EXIT_DOCUMENT_ERROR
    )
