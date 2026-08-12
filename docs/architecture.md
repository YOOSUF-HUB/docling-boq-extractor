# Architecture

## Purpose

A **standalone** service that turns an uploaded PDF Bill of Quantities into a
validated, canonical BOQ JSON document.

It is deliberately *not* an LLM wrapper. Document understanding is done by
Docling, structural rules are enforced by Python, and the LLM is used only for
the one thing it is actually good at: semantic restructuring of already-extracted
evidence.

This project has **no runtime dependency on the PMS application**. The PMS BOQ
contract is used only as a *target data format*.

## Pipeline

```text
PDF upload
   ↓
PDF validation            deterministic  (Phase 1)
   ↓
Docling document parse    document AI    (Phase 1)
   ↓
Evidence package          deterministic  (Phase 2)
   ↓
BOQ candidate detection   deterministic  (Phase 2/4)
   ↓
GPT-OSS 120B structuring  LLM            (Phase 4)
   ↓
Pydantic validation       deterministic  (Phase 3)
   ↓
BOQ rule validation       deterministic  (Phase 3)
   ↓
Canonical BOQ JSON + extraction report
```

## Responsibility split

| Stage | Owner | Responsibility |
|---|---|---|
| Validation | Python | Is this a readable PDF within limits? |
| Docling | Docling | Text, tables, layout, pages, OCR |
| Evidence | Python | Compact, ordered, provenance-carrying view of the document |
| Structuring | LLM | Which rows are items, which are sections, how they nest |
| Schema | Pydantic | Types, enums, required fields |
| BOQ rules | Python | Duplicate codes, hierarchy sanity, numeric sanity |
| Report | Python | Status, warnings, errors, statistics, provenance |

The LLM never generates IDs, timestamps, totals, or any PMS-internal field, and
never invents a value that is not present in the evidence.

## Layout

```text
app/
  config.py              settings from .env (no hardcoded secrets)
  logging_config.py      standard logging setup
  errors.py              document / extraction / AI / validation error split
  main.py                CLI entry point
  document/
    pdf_validator.py     deterministic input checks
    docling_processor.py Docling conversion + summary
    evidence_builder.py  DoclingDocument -> EvidencePackage
  schemas/
    evidence.py          agent-facing evidence schema
    boq.py               canonical BOQ contract (document-derived subset)
    report.py            extraction report (status, issues, statistics)
  validation/
    boq_validator.py     deterministic BOQ rules
  agent/
    prompts.py           system prompt, strict response schema, evidence rendering
    groq_client.py       Groq API wrapper + error mapping
    boq_agent.py         prompt -> validate -> controlled retry
    test_agent.py        manual live check (not part of the pytest suite)
  services/              end-to-end orchestration (Phase 5)
  api/                   FastAPI layer          (Phase 6)
scripts/                 fixture generation
tests/                   unit / integration / (opt-in) live tests
data/                    input, output, debug artifacts
```

## Target contract (summary)

The canonical output is the **document-derived subset** of the PMS BOQ contract:

`level_path`, `boq_item_code`, `boq_item_name`, `boq_description`, `quantity`,
`unit`, `cost_type`, the cost breakdown fields, `profit`, `discount`,
`fixed_rate`.

Five fields must never be guessed — `level_path`, `boq_item_code`,
`boq_item_name`, `quantity`, `unit`. If they cannot be extracted, the run
reports a failure instead of hallucinating.

Derived values (totals, tree depth, parents) and PMS/backend values (database
IDs, temp IDs, project-prefixed codes, persistence metadata) are explicitly out
of scope. `extra="forbid"` on every BOQ model enforces this: a payload
containing `id` or `temp_id` fails to parse rather than passing the field
through.

## Two validation layers

| Layer | Enforces | On failure |
|---|---|---|
| Pydantic (`app/schemas/boq.py`) | types, enums, required fields, non-negative finite numbers, no unknown fields | hard parse failure |
| Rules (`app/validation/boq_validator.py`) | duplicate codes, summary rows as items, numeric units, numbering used as section names, rate/breakdown conflicts | issues in the report |

Rule failures are split into **errors** (the result cannot be trusted as a BOQ)
and **warnings** (a human should look), which decide the report status:
`failed`, `partial` or `success`.

The report carries **no numeric confidence score** — there is no calibrated
basis for one, so an explicit status is used instead.

## How the LLM is constrained

The model is given evidence and asked for items. Everything it could get wrong
that Python can get right is kept away from it:

| Constraint | Mechanism |
|---|---|
| Response shape | Groq `response_format: json_schema` with `strict: true` |
| No invented fields | `extra="forbid"`, and the schema omits amount/total/id entirely |
| No timestamps or filenames | the model returns `items` only; `BOQDocument.assemble()` adds the rest |
| No guessed required fields | it reports rows it cannot extract in `unresolved` instead |
| Malformed output | one repair round quoting the validation errors, then a hard failure |

Nothing is ever repaired locally: a bad response is re-requested or reported,
because patching JSON into shape is indistinguishable from inventing data.

Self-reported `unresolved` rows become report *warnings* (`MISSING_UNIT` and
friends) — the row is missing from the output, which a human needs to know,
but the agent behaved correctly by refusing to guess.

## Phase boundaries

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Project foundation + Docling processing | done |
| 2 | Evidence package | done |
| 3 | Canonical BOQ schema + deterministic validation | done |
| 4 | Groq GPT-OSS 120B integration | done |
| 5 | End-to-end PDF → BOQ JSON | pending |
| 6 | FastAPI upload API | pending |
| 7 | Robustness + real BOQ test suite | pending |
| 8 | Observability + debug artifacts | pending |
| 9 | Docker | pending |
| 10 | Documentation | pending |
