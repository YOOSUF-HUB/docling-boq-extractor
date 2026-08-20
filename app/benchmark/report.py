"""Rendering a `TokenBenchmark` for humans.

The console layout follows the pipeline CLI's `========== SECTION ==========`
convention so both tools read the same way.

Every printed figure says where it came from. A count this project made is
plain; a figure Groq charged is labelled `charged`; a number that can only be a
floor until a live run happens is labelled `at least`. The point of the
benchmark is to be trusted as a baseline, and a baseline that blurs measured and
inferred values cannot be.
"""

from __future__ import annotations

from app.benchmark.metrics import TokenBenchmark

_LABEL_WIDTH = 30


def render(benchmark: TokenBenchmark, *, top: int = 10) -> str:
    """The full single-document report."""
    lines: list[str] = []
    lines += _header(benchmark)
    lines += _document(benchmark)
    lines += _funnel(benchmark)
    lines += _prompt(benchmark)
    lines += _cost_centres(benchmark)
    lines += _components(benchmark, top)
    lines += _budget(benchmark)
    if benchmark.completion is not None:
        lines += _live(benchmark)
    return "\n".join(lines)


def _row(label: str, value: object) -> str:
    return f"{label:<{_LABEL_WIDTH}}: {value}"


def _tokens(count: int) -> str:
    return f"{count:,} tokens"


def _header(benchmark: TokenBenchmark) -> list[str]:
    tokenizer = benchmark.tokenizer
    return [
        "",
        "========== TOKEN BENCHMARK ==========",
        _row("Source", benchmark.source),
        _row("Mode", benchmark.mode),
        _row("Model", benchmark.model),
        _row("Tokenizer", f"{tokenizer.encoding} ({tokenizer.library} {tokenizer.library_version})"),
        _row("Tokenizer fidelity", tokenizer.note),
        _row("Measured at", benchmark.measured_at.isoformat(timespec="seconds")),
    ]


def _document(benchmark: TokenBenchmark) -> list[str]:
    document = benchmark.document
    lines = [
        "",
        "========== DOCUMENT ==========",
        _row("File", document.filename),
        _row("Pages", document.page_count),
        _row("Tables", document.table_count),
        _row("Table rows", document.table_row_count),
        _row("Text blocks", document.text_block_count),
        _row("Pictures", document.picture_count),
    ]
    if document.file_size_bytes is not None:
        lines.append(_row("File size", f"{document.file_size_bytes / 1024:,.1f} KiB"))
    if document.docling_duration_seconds is not None:
        lines.append(_row("Docling duration", f"{document.docling_duration_seconds:.2f}s"))
    return lines


def _funnel(benchmark: TokenBenchmark) -> list[str]:
    evidence = benchmark.evidence
    lines = ["", "========== EVIDENCE FUNNEL =========="]

    if evidence.docling_document_json is not None:
        lines.append(
            _row("Docling document JSON", _measure(evidence.docling_document_json))
        )
    lines += [
        _row("Evidence JSON (compact)", _measure(evidence.serialized_compact)),
        _row("Evidence JSON (indent=2)", _measure(evidence.serialized_indented)),
        _row("Rendered prompt", _measure(evidence.rendered_prompt)),
        _row(
            "Renderer saving",
            f"{evidence.rendering_saving_percent:.1f}% vs sending the evidence JSON",
        ),
    ]
    return lines


def _measure(measurement) -> str:
    return (
        f"{measurement.tokens:>7,} tokens  "
        f"{measurement.characters:>8,} chars  "
        f"({measurement.chars_per_token:.2f} chars/token)"
    )


def _prompt(benchmark: TokenBenchmark) -> list[str]:
    prompt = benchmark.prompt
    lines = [
        "",
        "========== REQUEST INPUT ==========",
        _row("System prompt", _measure(prompt.system_prompt)),
        _row("User prompt (evidence)", _measure(prompt.user_prompt)),
        _row("Response schema", _measure(prompt.response_schema)),
        _row("Counted input", _tokens(prompt.counted_input_tokens)),
    ]

    if prompt.envelope_tokens is None:
        lines.append(_row("Chat envelope", "not measured (needs a live run)"))
        lines.append(_row("Input total", f"at least {_tokens(prompt.input_tokens)}"))
    else:
        lines.append(
            _row("Chat envelope", f"{prompt.envelope_tokens:+,} tokens — {prompt.envelope_source}")
        )
        lines.append(_row("Input total", _tokens(prompt.input_tokens)))
    return lines


def _cost_centres(benchmark: TokenBenchmark) -> list[str]:
    lines = [
        "",
        "========== WHERE THE TOKENS GO ==========",
        f"{'centre':<30}{'tokens':>9}{'share':>8}  {'scales with doc':<16}",
    ]
    for centre in benchmark.cost_centres:
        scales = "yes" if centre.scales_with_document else "no — fixed cost"
        lines.append(
            f"{centre.name:<30}{centre.tokens:>9,}{centre.share_percent:>7.1f}%  {scales:<16}"
        )

    fixed = sum(c.tokens for c in benchmark.cost_centres if not c.scales_with_document)
    scaling = sum(c.tokens for c in benchmark.cost_centres if c.scales_with_document)
    lines += [
        "",
        _row("Fixed per request", _tokens(fixed)),
        _row("Scales with the document", _tokens(scaling)),
    ]
    return lines


def _components(benchmark: TokenBenchmark, top: int) -> list[str]:
    if top <= 0 or not benchmark.components:
        return []

    heaviest = benchmark.heaviest_components(top)
    lines = [
        "",
        f"========== HEAVIEST PROMPT COMPONENTS (top {len(heaviest)}) ==========",
        f"{'tokens':>7}{'share':>8}  {'per row':>8}  {'page':>5}  component",
    ]
    for part in heaviest:
        per_row = f"{part.tokens_per_row:.1f}" if part.tokens_per_row is not None else "—"
        page = part.page_number if part.page_number is not None else "?"
        lines.append(
            f"{part.tokens:>7,}{part.share_of_user_prompt_percent:>7.1f}%  "
            f"{per_row:>8}  {page:>5}  {part.label}"
        )

    tables = [part for part in benchmark.components if part.kind == "table"]
    if tables:
        rows = sum(part.row_count or 0 for part in tables)
        tokens = sum(part.tokens for part in tables)
        if rows:
            lines += ["", _row("Average table row", f"{tokens / rows:.1f} tokens")]
    return lines


def _budget(benchmark: TokenBenchmark) -> list[str]:
    budget = benchmark.budget
    qualifier = "at least " if budget.budget_is_lower_bound else ""
    verdict = "EXCEEDS the limit" if budget.exceeds_limit else "fits"

    return [
        "",
        "========== RATE-LIMIT BUDGET ==========",
        _row("Reserved for completion", _tokens(budget.max_completion_tokens)),
        _row("Requested budget", f"{qualifier}{_tokens(budget.requested_budget_tokens)}"),
        _row("Configured TPM limit", _tokens(budget.tpm_limit)),
        _row("Headroom", f"{budget.headroom_tokens:+,} tokens"),
        _row("Utilisation", f"{budget.utilisation_percent:.1f}% — {verdict}"),
        "",
        "Groq reserves prompt + max_completion_tokens against TPM before the call,",
        "so the completion budget is spent whether the model uses it or not.",
    ]


def _live(benchmark: TokenBenchmark) -> list[str]:
    completion = benchmark.completion
    assert completion is not None

    lines = [
        "",
        "========== LIVE USAGE (charged by Groq) ==========",
        _row("Model", completion.model or "—"),
        _row("Attempts", completion.attempts),
        _row("Finish reason", completion.finish_reason or "—"),
        _row("Prompt tokens", _charged(completion.prompt_tokens)),
        _row("Completion tokens", _charged(completion.completion_tokens)),
        _row("Total tokens", _charged(completion.total_tokens)),
    ]

    if completion.attempts > 1:
        lines.append(
            _row("Total across attempts", _charged(completion.total_tokens_all_attempts))
        )
    lines += [
        _row("Items extracted", completion.items_extracted),
        _row("Unresolved rows", completion.unresolved_items),
        _row("Validation status", completion.validation_status or "—"),
        _row("LLM duration", f"{completion.llm_duration_seconds:.2f}s"),
    ]

    if completion.estimate_error_tokens is not None:
        lines.append(
            _row(
                "Counted vs charged",
                f"{completion.estimate_error_tokens:+,} tokens "
                f"({completion.estimate_error_percent:+.1f}%)",
            )
        )
    if not completion.request_matches_measurement:
        lines.append(
            _row("WARNING", "the request sent did not match the prompt measured above")
        )
    if completion.error:
        lines.append(_row("Error", completion.error))

    if completion.attempts > 1:
        lines += ["", "Per attempt:"]
        for attempt in completion.per_attempt:
            lines.append(
                f"  #{attempt.index}  {attempt.message_count} messages  "
                f"sent {attempt.sent_tokens:,} counted / "
                f"{_charged(attempt.prompt_tokens)} charged in, "
                f"{_charged(attempt.completion_tokens)} out  "
                f"({attempt.finish_reason or 'no response'})"
            )
    return lines


def _charged(value: int | None) -> str:
    return f"{value:,}" if value is not None else "—"


# --- multi-document comparison ---------------------------------------


def render_comparison(benchmarks: list[TokenBenchmark]) -> str:
    """A one-line-per-document table, for spotting how usage scales."""
    if len(benchmarks) < 2:
        return ""

    lines = [
        "",
        "========== CORPUS COMPARISON ==========",
        f"{'document':<34}{'pages':>6}{'rows':>6}{'prompt':>9}{'input':>9}"
        f"{'budget':>9}{'TPM':>7}",
    ]
    for benchmark in sorted(benchmarks, key=lambda b: b.budget.requested_budget_tokens):
        flag = "OVER" if benchmark.budget.exceeds_limit else "ok"
        lines.append(
            f"{_truncate(benchmark.document.filename, 33):<34}"
            f"{benchmark.document.page_count:>6}"
            f"{benchmark.document.table_row_count:>6}"
            f"{benchmark.prompt.user_prompt.tokens:>9,}"
            f"{benchmark.prompt.input_tokens:>9,}"
            f"{benchmark.budget.requested_budget_tokens:>9,}"
            f"{flag:>7}"
        )

    scaling = _rows_per_token(benchmarks)
    if scaling is not None:
        lines += [
            "",
            f"Marginal cost of document content: ~{scaling:.1f} tokens per table row",
            "(slope between the smallest and largest document measured).",
        ]
    return "\n".join(lines)


def _rows_per_token(benchmarks: list[TokenBenchmark]) -> float | None:
    """Slope of prompt tokens against table rows, across the corpus."""
    points = [
        (b.document.table_row_count, b.prompt.user_prompt.tokens)
        for b in benchmarks
        if b.document.table_row_count
    ]
    if len(points) < 2:
        return None

    smallest = min(points)
    largest = max(points)
    if largest[0] == smallest[0]:
        return None
    return (largest[1] - smallest[1]) / (largest[0] - smallest[0])


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
