# Token baseline

Measured with `python -m app.benchmark`, tokenizer `o200k_harmony` (the encoding
`openai/gpt-oss-120b` itself uses), against `data/input/` on 2026-08-20.

This document records **what the pipeline currently costs**. Nothing was changed
to produce it: the benchmark calls the agent's own `build_user_prompt()` and, in
live mode, reaches Groq through `BOQAgent` itself.

## Headline

| | tokens |
|---|---|
| Fixed cost of any request, before a single line of document | **5,606** |
| Configured TPM limit (`GROQ_TPM_LIMIT`, free tier) | 8,000 |
| Left for document content | **2,394** |
| Marginal cost of one BOQ table row | ~25–32 |

**A request has room for roughly 75–95 table rows.** Every BOQ larger than that
exceeds the limit regardless of how efficiently its rows are rendered.

## Where the fixed 5,606 goes

| Component | Tokens | Why it is fixed |
|---|---|---|
| Reserved completion budget | 4,000 | `GROQ_MAX_COMPLETION_TOKENS`, reserved against TPM before the call runs |
| System prompt | 907 | The extraction rules, identical on every request |
| Response schema | 498 | `BOQ_RESPONSE_SCHEMA`, serialized into `response_format` |
| Chat envelope | 141 | Harmony role markers and server-side preamble (measured) |
| Prompt header + footer | 60 | Document facts and the closing instruction |

The completion budget is the single largest line item in the entire request —
larger than the system prompt, the schema and every small document in the corpus
combined. Groq reserves `prompt_tokens + max_completion_tokens` against
tokens-per-minute *before* the call runs, so it is spent whether the model emits
one token or four thousand.

(The floor moves by a token or two between documents: the prompt header states
the filename and page count, so a longer filename costs marginally more. It is
5,606 for `sample_boq.pdf` and 5,608 for `Contract_BOQ_179.pdf`.)

### This explains the 10,717-token request

The 3-page BOQ that produced a 10,717-token request measures **2,717 input
tokens**. With an 8,000-token completion reserve, `2,717 + 8,000 = 10,717` —
exact to the token.

**75% of that request was the reserved completion budget, not the document.**
The document content itself was 1,171 tokens.

## Corpus

`input` = counted prompt + measured envelope. `budget` = what the request
reserves against TPM, at `GROQ_MAX_COMPLETION_TOKENS=4000`.

| document | pages | rows | prompt | input | budget | vs 8k |
|---|---:|---:|---:|---:|---:|---|
| 09_non_boq_document.pdf | 1 | 0 | 207 | 1,753 | 5,753 | ok |
| 04_missing_fields_boq.pdf | 1 | 9 | 292 | 1,838 | 5,838 | ok |
| 07_scanned_boq.pdf | 1 | 0 | 294 | 1,840 | 5,840 | ok |
| sample_boq.pdf | 1 | 10 | 307 | 1,853 | 5,853 | ok |
| 01_structured_boq.pdf | 1 | 12 | 344 | 1,890 | 5,890 | ok |
| 03_reordered_columns_boq.pdf | 1 | 12 | 344 | 1,890 | 5,890 | ok |
| 08_rates_costs_boq.pdf | 1 | 11 | 387 | 1,933 | 5,933 | ok |
| 05_totals_subtotals_boq.pdf | 1 | 14 | 391 | 1,937 | 5,937 | ok |
| 06_nested_hierarchy_boq.pdf | 1 | 16 | 443 | 1,989 | 5,989 | ok |
| 02_multipage_boq.pdf | 2 | 16 | 558 | 2,104 | 6,104 | ok |
| 10_large_multipage_boq.pdf | 3 | 36 | 1,171 | 2,717 | 6,717 | ok |
| **Contract_BOQ_179.pdf** | **12** | **325** | **10,505** | **12,051** | **16,051** | **201%** |

Marginal cost across the corpus: **~32.3 tokens per table row**. Within a single
large document the average is lower (~25.6), because the fixed per-table header
line is amortised over more rows.

## The real document

`Contract_BOQ_179.pdf` — 12 pages, 11 tables, 325 rows, 32 text blocks.

```
system prompt                       907    5.7%   fixed
response schema                     498    3.1%   fixed
prompt header + footer               62    0.4%   fixed
chat envelope                       141    0.9%   fixed
reserved completion budget        4,000   24.9%   fixed
page markers                         55    0.3%   scales
text blocks                       2,082   13.0%   scales
table rows                        8,306   51.8%   scales
```

Two things stand out.

**Text blocks cost 2,082 tokens — 20% of the prompt — and contain no BOQ
items.** The single heaviest component in the whole prompt is not a table at
all; it is one 1,378-token prose block ("Supply and installation of complete
waste water…"), 13.1% of the prompt on its own.

**Table rows are 52% of the prompt** and are the only part the extracted items
actually come from.

## Funnel

The prompt renderer already does real work; this is the size of the same
document in each representation:

| Form | Tokens |
|---|---:|
| Docling document JSON | 298,889 |
| Evidence JSON (indent=2) | 21,912 |
| Evidence JSON (compact) | 14,245 |
| Rendered prompt | 10,505 |

Rendering saves **26.2%** against sending the evidence as JSON, and **96.5%**
against Docling's own output. The remaining 10,505 tokens are close to the floor
for "every row of this document, as text".

## Confidence in these numbers

The tokenizer is exact, not an estimate. On the live run, counted input was
1,712 tokens against 1,853 charged by Groq — a gap of 141 tokens that is
entirely the chat envelope, since applying the measured envelope reconciles the
two to the token. No character heuristic is used anywhere; `tiktoken` is a hard
requirement and the benchmark errors rather than approximating.

Static-mode figures for documents that were never sent live are marked
`input_is_lower_bound` unless `--envelope-tokens` supplies a measured envelope,
as it does throughout this document.

## Reproducing

```bash
python -m app.benchmark --corpus data/input --envelope-tokens 141
python -m app.benchmark --evidence data/benchmark/evidence/sample_boq.evidence.json --live
```

Records are written to `data/benchmark/*.benchmark.json`.

## Not in scope here

This phase measures only. No prompt, schema, evidence-building or chunking
behaviour was changed, and none is proposed in this document.
