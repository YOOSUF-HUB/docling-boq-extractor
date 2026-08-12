# Super Prompt — Build a Standalone PDF BOQ Structure Extraction Agent

## Role

You are Claude Code working inside a **new, standalone repository** for a personal AI engineering project.

I want to build an AI Agent that extracts a Bill of Quantities (BOQ) from **user-uploaded PDF documents**.

This project is **completely separate from the existing PMS application**.

The PMS repository must NOT be imported, copied, modified, or used as a runtime dependency.

The only reason the existing PMS BOQ contract is relevant is that the final extracted JSON should be structurally compatible with the previously reverse-engineered document-derived BOQ contract described below.

The goal is to learn and demonstrate a complete production-style document AI pipeline:

```text
User uploads PDF
       ↓
PDF validation
       ↓
Docling document processing
       ↓
Structured extraction / evidence
       ↓
BOQ candidate detection
       ↓
AI BOQ extraction / restructuring
       ↓
Deterministic validation
       ↓
Canonical BOQ JSON
       ↓
Extraction report
```

---

# IMPORTANT: BUILD THIS IN PHASES

Do NOT implement the entire system in one step.

Build it in clearly separated phases so I can independently inspect, test, commit, and push each phase to Git.

After completing each phase:

1. Run the relevant tests.
2. Run the application or CLI if applicable.
3. Show the resulting output.
4. Explain what was implemented.
5. List files created/modified.
6. Give me the exact Git commit suggestion.
7. STOP.

Do NOT automatically continue to the next phase.

I will explicitly tell you to continue.

Use small, logical commits.

---

# Technology Stack

Use:

- Python 3.13
- Docling for PDF/document extraction
- Groq API
- `openai/gpt-oss-120b` as the LLM
- Pydantic v2 for schemas
- FastAPI for the API layer
- Uvicorn
- pytest
- python-dotenv
- standard Python logging

Use LangChain/LangGraph only if there is a real architectural reason.

Do NOT add unnecessary frameworks.

The core extraction pipeline should remain understandable without a framework abstraction.

---

# Primary Objective

Build a standalone service capable of processing a BOQ PDF such as:

```text
PROJECT: Sample Commercial Building

BILL OF QUANTITIES

1 EARTHWORKS

1.1 Excavation for foundations       m3       125.50
1.2 Filling with selected material   m3        80.00

2 CONCRETE WORKS

2.1 Blinding concrete                m3        15.00
2.2 Reinforced concrete foundations  m3        50.00
2.3 Reinforced concrete columns      m3        35.00

3 MASONRY

3.1 Brickwork in cement mortar       m2       250.00
```

and produce canonical JSON such as:

```json
{
  "boq_source_version": "1.0",
  "document": {
    "filename": "sample_boq.pdf",
    "extracted_at": "2026-08-12T09:14:00Z"
  },
  "items": [
    {
      "level_path": ["EARTHWORKS"],
      "boq_item_code": "1.1",
      "boq_item_name": "Excavation for foundations",
      "boq_description": "",
      "quantity": 125.5,
      "unit": "m3",
      "cost_type": "per_unit",
      "labour": 0.0,
      "machine": 0.0,
      "material": 0.0,
      "fuel": 0.0,
      "miscellaneous": 0.0,
      "subcontract": 0.0,
      "site_overhead": 0.0,
      "head_office_overhead": 0.0,
      "profit": {
        "mode": "amount",
        "value": 0.0
      },
      "discount": {
        "mode": "amount",
        "value": 0.0
      },
      "fixed_rate": null
    }
  ]
}
```

The exact target contract is described later in this prompt.

---

# CORE ARCHITECTURE

The system should have a clear separation of responsibilities:

```text
                    ┌──────────────────┐
                    │   PDF Upload     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ PDF Validation   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │     Docling      │
                    │  Document Parse  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Evidence Package │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ BOQ Candidate    │
                    │ Detection        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ GPT-OSS 120B     │
                    │ BOQ Structuring  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Pydantic         │
                    │ Validation       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Deterministic    │
                    │ BOQ Validation   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Canonical JSON   │
                    └──────────────────┘
```

Do not allow the LLM to be responsible for everything.

Use deterministic Python wherever possible.

The LLM should be responsible for semantic restructuring and interpretation of the extracted evidence.

---

# TARGET JSON CONTRACT

The following contract comes from the existing PMS repository's reverse-engineered BOQ contract.

The new project must remain standalone.

This contract is a **target data format**, not a runtime dependency.

The PDF agent should produce the document-derived subset only.

## Canonical JSON

```json
{
  "boq_source_version": "1.0",
  "document": {
    "filename": "string",
    "extracted_at": "ISO-8601 UTC"
  },
  "items": [
    {
      "level_path": ["Section", "Subsection"],
      "boq_item_code": "string",
      "boq_item_name": "string",
      "boq_description": "string",
      "quantity": 0.0,
      "unit": "string",
      "cost_type": "per_unit",
      "labour": 0.0,
      "machine": 0.0,
      "material": 0.0,
      "fuel": 0.0,
      "miscellaneous": 0.0,
      "subcontract": 0.0,
      "site_overhead": 0.0,
      "head_office_overhead": 0.0,
      "profit": {
        "mode": "amount",
        "value": 0.0
      },
      "discount": {
        "mode": "amount",
        "value": 0.0
      },
      "fixed_rate": null
    }
  ]
}
```

The contract requires:

- `level_path`
- `boq_item_code`
- `boq_item_name`
- `quantity`
- `unit`

These fields must never be guessed.

If the agent cannot reliably extract one of them, the system should report an extraction/validation failure.

The contract explicitly distinguishes:

```text
DOCUMENT-DERIVED
    level_path
    boq_item_code
    boq_item_name
    boq_description
    quantity
    unit
    cost_type
    labour
    machine
    material
    fuel
    miscellaneous
    subcontract
    site_overhead
    head_office_overhead
    profit
    discount
    fixed_rate

DERIVED
    hierarchy tree
    depths
    parent relationships
    computed values
    summaries

PMS/BACKEND-GENERATED
    database IDs
    temp IDs
    project-prefixed item codes
    database timestamps
    persistence metadata
```

The standalone PDF agent must NOT generate PMS-specific IDs or database fields.

Source contract: `BOQ_JSON_CONTRACT.md`. The contract states that the standalone PDF agent should output only the document-derived subset and must not invent PMS-internal fields. fileciteturn6file0L76-L109

---

# HIERARCHY MODEL

The BOQ hierarchy is represented through:

```json
"level_path": [
  "EARTHWORKS",
  "BULK EXCAVATION"
]
```

This means:

```text
EARTHWORKS
└── BULK EXCAVATION
    └── item
```

A top-level section:

```text
1 EARTHWORKS
```

becomes:

```json
"level_path": ["EARTHWORKS"]
```

A subsection:

```text
1 EARTHWORKS
1.1 BULK EXCAVATION
1.1.1 Excavation in Rock
```

becomes:

```json
"level_path": [
  "EARTHWORKS",
  "BULK EXCAVATION"
]
```

The item code remains:

```json
"boq_item_code": "1.1.1"
```

The section numbering itself should not automatically be treated as the item code.

The contract defines hierarchy through item paths. fileciteturn6file0L151-L163

---

# IMPORTANT BOQ RULES

Implement these rules explicitly.

## 1. One row = one BOQ item

Do not treat:

```text
TOTAL
SUBTOTAL
CARRIED FORWARD
BROUGHT FORWARD
GRAND TOTAL
```

as BOQ items.

---

## 2. Section headings are not items

For example:

```text
2 CONCRETE WORKS
```

is a section.

It becomes part of:

```json
"level_path": ["CONCRETE WORKS"]
```

It does not become an item unless the document clearly represents it as a priced BOQ line.

---

## 3. Never guess required fields

Required:

```text
level_path
boq_item_code
boq_item_name
quantity
unit
```

If one is missing or ambiguous:

```text
DO NOT hallucinate it.
```

Return a validation/extraction issue.

The source contract explicitly says these five fields must fail loudly rather than be guessed. fileciteturn6file0L31-L37

---

## 4. Amount is NOT a target output field

If the PDF contains:

```text
Quantity | Rate | Amount
```

extract:

```text
quantity
fixed_rate
```

Do NOT put the amount into the canonical JSON.

The amount should be treated as verification evidence.

The existing contract explicitly says amount/total fields are computed outputs, not document inputs. fileciteturn6file0L167-L184

---

## 5. Rate-only BOQ

If the PDF contains:

```text
Item | Description | Unit | Qty | Rate
```

and there is no cost breakdown:

```json
"fixed_rate": 250000.0
```

and:

```json
"labour": 0.0,
"machine": 0.0,
"material": 0.0,
"fuel": 0.0,
"miscellaneous": 0.0,
"subcontract": 0.0,
"site_overhead": 0.0,
"head_office_overhead": 0.0
```

The contract identifies `fixed_rate` as the correct field for a tender BOQ that provides a direct rate instead of a cost breakdown. fileciteturn6file0L143-L149

---

## 6. Cost breakdown

If the PDF actually contains:

```text
Labour
Machine
Material
Fuel
Miscellaneous
Subcontract
Site Overhead
Head Office Overhead
```

extract them into their corresponding fields.

Do not invent zero-value costs when a real value exists.

---

## 7. Profit and discount

If the document contains:

```text
10%
```

represent:

```json
{
  "mode": "percent",
  "value": 10.0
}
```

If the document contains an amount:

```text
5000
```

represent:

```json
{
  "mode": "amount",
  "value": 5000.0
}
```

Do not collapse percentage and amount semantics.

---

# DOCLING RESPONSIBILITY

Docling is responsible for document understanding/extraction.

It should be used to extract:

- text
- tables
- document structure
- page information
- layout information where available
- OCR content when required
- document JSON/Markdown representation

The system should preserve enough evidence for the LLM to reason about the BOQ.

Do not immediately flatten everything into plain text if structured Docling information is available.

The evidence package should preferably preserve:

```text
document metadata
pages
tables
text blocks
headings
table rows
table columns
document order
```

---

# EVIDENCE-FIRST DESIGN

Do NOT send an uncontrolled full PDF dump directly to the LLM.

Create an intermediate evidence representation.

For example:

```json
{
  "document": {
    "filename": "sample_boq.pdf",
    "page_count": 3
  },
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "tables": [
        {
          "columns": ["Item", "Description", "Unit", "Quantity", "Rate"],
          "rows": [
            ["1.1", "Excavation", "m3", "125.5", "450.00"]
          ]
        }
      ]
    }
  ]
}
```

This is an architectural example.

Design the actual evidence schema based on what Docling provides.

Do not fabricate Docling fields that do not exist.

---

# LLM RESPONSIBILITY

Use:

```text
Groq
    ↓
openai/gpt-oss-120b
```

The model should receive:

1. System instructions
2. BOQ extraction rules
3. Target schema
4. Docling evidence
5. Explicit instruction to preserve document values
6. Explicit instruction not to hallucinate missing values

The LLM should return structured JSON.

Use structured output / JSON schema where supported.

Validate the response with Pydantic.

If the LLM returns invalid JSON/schema:

```text
LLM output
    ↓
Validation failure
    ↓
Controlled retry / failure
```

Do not silently repair arbitrary malformed output.

---

# DO NOT MAKE THE LLM DO DETERMINISTIC WORK

The LLM should not:

- calculate totals unnecessarily
- generate database IDs
- generate PMS IDs
- invent item codes
- invent quantities
- invent units
- invent rates
- generate timestamps
- create database metadata

Use Python for deterministic validation and normalization.

---

# AGENT BEHAVIOUR

The system should behave like an extraction agent rather than a generic chatbot.

The agent should be able to:

1. Inspect the extracted document evidence.
2. Determine whether the document appears to contain a BOQ.
3. Identify BOQ sections.
4. Identify BOQ item rows.
5. Identify item codes.
6. Identify descriptions/names.
7. Identify quantities.
8. Identify units.
9. Identify rates.
10. Identify cost breakdowns when available.
11. Identify profit/discount.
12. Build hierarchy using `level_path`.
13. Ignore totals/subtotals.
14. Detect ambiguous/missing information.
15. Produce canonical JSON.
16. Produce an extraction report.

---

# EXTRACTION REPORT

In addition to canonical JSON, create a separate report structure for diagnostics.

Example:

```json
{
  "status": "success",
  "confidence": 0.94,
  "warnings": [
    {
      "type": "MISSING_UNIT",
      "message": "Unit could not be confidently identified for item 2.2"
    }
  ],
  "errors": [],
  "statistics": {
    "pages_processed": 3,
    "tables_detected": 4,
    "items_extracted": 17,
    "sections_detected": 5
  }
}
```

This report is for the standalone project only.

Do not mix it into the canonical PMS-compatible BOQ JSON.

Do not claim confidence scores are scientifically calibrated.

If confidence cannot be reliably calculated, do not invent numeric confidence.

Prefer explicit statuses such as:

```text
success
partial
failed
```

---

# SUPPORT DIFFERENT PDF TYPES

The system should be designed for PDFs that may be:

### Type A — Structured text PDF

Text and tables are selectable.

```text
Best case.
```

### Type B — Scanned PDF

Pages contain images and OCR is required.

### Type C — Mixed PDF

Some pages contain text and others scanned images.

### Type D — Complex BOQ

Tables may span multiple pages.

### Type E — Poorly structured PDF

The document may contain text that visually looks like a BOQ but has weak machine-readable structure.

The system should not assume every PDF will be perfectly structured.

---

# IMPORTANT: DOCLING FIRST, LLM SECOND

The intended architecture is:

```text
PDF
 ↓
Docling
 ↓
structured evidence
 ↓
LLM
 ↓
canonical BOQ
```

NOT:

```text
PDF
 ↓
LLM directly
```

The purpose of this project is specifically to understand how Docling can provide the document processing layer and how an AI agent can perform semantic restructuring afterwards.

---

# PROJECT STRUCTURE

Create a clean structure similar to:

```text
docling-boq-extractor/
│
├── app/
│   ├── __init__.py
│   │
│   ├── main.py
│   │
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── document/
│   │   ├── __init__.py
│   │   ├── docling_processor.py
│   │   └── evidence_builder.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── boq_agent.py
│   │   ├── prompts.py
│   │   └── groq_client.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── evidence.py
│   │   ├── boq.py
│   │   └── report.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── boq_validator.py
│   │
│   └── services/
│       ├── __init__.py
│       └── extraction_service.py
│
├── tests/
│   ├── fixtures/
│   │   └── ...
│   ├── test_docling.py
│   ├── test_evidence.py
│   ├── test_boq_schema.py
│   ├── test_validation.py
│   └── test_agent.py
│
├── data/
│   ├── input/
│   ├── output/
│   └── debug/
│
├── docs/
│   ├── architecture.md
│   ├── extraction-pipeline.md
│   └── testing.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

Adjust the structure if a simpler structure is more appropriate.

Do not over-engineer the first version.

---

# PHASE PLAN

## PHASE 1 — Project Foundation + Docling

### Goal

Create the project and prove that Docling can process a BOQ PDF.

Implement:

- Python environment configuration
- dependency management
- `.env.example`
- configuration
- logging
- Docling processor
- PDF input handling
- Markdown extraction
- Docling JSON export
- basic CLI entry point

Input:

```text
data/input/sample_boq.pdf
```

Output:

```text
data/output/docling_output.md
data/output/docling_document.json
```

Also print a useful summary:

```text
Pages:
Tables:
Text blocks:
Pictures:
```

### Acceptance criteria

Running:

```bash
python -m app.main
```

must successfully process the sample PDF.

The implementation must clearly demonstrate:

```text
PDF → Docling → extracted representation
```

### Commit

Suggested:

```bash
git add .
git commit -m "feat: add Docling PDF processing foundation"
```

STOP after Phase 1.

---

# PHASE 2 — Evidence Package

### Goal

Convert Docling output into a clean, explicit evidence package for the AI agent.

Implement:

```text
DoclingDocument
       ↓
EvidenceBuilder
       ↓
EvidencePackage
```

Preserve useful information such as:

- page number
- text
- tables
- table rows
- headings where available
- document order
- source references where available

Create Pydantic models.

Example conceptual schema:

```python
EvidencePackage
    document
    pages[]
    tables[]
    text_blocks[]
```

Do not blindly copy the entire Docling JSON.

The evidence package should be optimized for downstream BOQ reasoning.

### Tests

Test:

- text extraction
- table extraction
- multiple pages
- empty tables
- documents with no tables

### Output

```text
data/output/evidence.json
```

### Commit

```bash
git add .
git commit -m "feat: build Docling evidence package"
```

STOP.

---

# PHASE 3 — BOQ Schema + Deterministic Validation

### Goal

Implement the canonical BOQ JSON schema before introducing the LLM.

Create Pydantic models for:

```text
BOQDocument
BOQItem
ValueSpec
ExtractionReport
```

Implement validation for:

- required fields
- quantity
- unit
- item code
- item name
- level path
- duplicate codes
- level path validity
- numeric fields
- enum values

Do NOT call Groq yet.

Create a manually authored JSON fixture.

Example:

```text
tests/fixtures/valid_boq.json
```

Test that it passes.

Create invalid fixtures and verify they fail.

### Important

Do not implement PMS-specific fields.

The schema should follow the standalone PDF target contract.

### Commit

```bash
git add .
git commit -m "feat: add canonical BOQ schema and validation"
```

STOP.

---

# PHASE 4 — Groq GPT-OSS 120B Integration

### Goal

Integrate:

```text
Groq API
openai/gpt-oss-120b
```

Implement:

```text
EvidencePackage
       ↓
Prompt
       ↓
Groq
       ↓
Structured JSON
       ↓
Pydantic validation
```

Use:

```env
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

Do not hardcode secrets.

Create:

```text
app/agent/groq_client.py
app/agent/prompts.py
app/agent/boq_agent.py
```

The system prompt must explicitly state:

- extract only from evidence
- do not hallucinate
- preserve exact numbers
- preserve item codes
- distinguish section headings from items
- ignore subtotal/total rows
- construct `level_path`
- do not generate PMS internal fields
- do not calculate fields that are not present
- return the required JSON schema

### Test

Run:

```bash
python -m app.agent.test_agent
```

against a known evidence fixture.

### Commit

```bash
git add .
git commit -m "feat: integrate Groq GPT-OSS BOQ extraction agent"
```

STOP.

---

# PHASE 5 — End-to-End PDF → BOQ JSON

### Goal

Connect everything:

```text
PDF
 ↓
Docling
 ↓
Evidence
 ↓
GPT-OSS 120B
 ↓
Pydantic
 ↓
Validation
 ↓
Canonical BOQ JSON
```

Create:

```text
app/services/extraction_service.py
```

Expose a single high-level function:

```python
extract_boq_from_pdf(path)
```

It should return:

```text
BOQResult
    ├── boq
    ├── report
    └── evidence_metadata
```

Create CLI:

```bash
python -m app.main --input data/input/sample_boq.pdf
```

Output:

```text
========== DOCLING ==========
...

========== EVIDENCE ==========
...

========== AI BOQ ==========
...

========== VALIDATION ==========
...

========== FINAL BOQ ==========
...
```

### Commit

```bash
git add .
git commit -m "feat: add end-to-end PDF BOQ extraction pipeline"
```

STOP.

---

# PHASE 6 — FastAPI Upload API

### Goal

Add a clean API without changing the extraction core.

Endpoint:

```http
POST /api/v1/boq/extract
```

Accept:

```text
multipart/form-data
file=<PDF>
```

Return:

```json
{
  "status": "success",
  "boq": {},
  "report": {}
}
```

Handle:

- invalid file
- unsupported extension
- empty file
- Docling failure
- LLM failure
- schema validation failure
- extraction failure

Do not expose internal stack traces to the client.

### Add

```text
GET /health
```

### Commit

```bash
git add .
git commit -m "feat: add BOQ extraction FastAPI endpoint"
```

STOP.

---

# PHASE 7 — Robustness + Real BOQ Test Suite

### Goal

Test multiple document types.

Create fixtures for:

```text
01_structured_boq.pdf
02_multi_page_boq.pdf
03_scanned_boq.pdf
04_mixed_boq.pdf
05_nested_boq.pdf
06_missing_unit.pdf
07_missing_quantity.pdf
08_totals_and_subtotals.pdf
09_rate_only_boq.pdf
10_cost_breakdown_boq.pdf
```

Test:

- section detection
- item extraction
- hierarchy
- missing values
- totals
- multi-page tables
- OCR
- rate extraction
- cost breakdown
- duplicate codes
- malformed output
- LLM failure

Where possible, separate:

```text
Docling tests
```

from:

```text
LLM tests
```

Do not make the entire test suite depend on Groq API calls.

Use mocked LLM responses for deterministic tests.

### Commit

```bash
git add .
git commit -m "test: add comprehensive BOQ extraction fixtures"
```

STOP.

---

# PHASE 8 — Extraction Quality + Observability

### Goal

Make the agent useful for debugging.

Store optional debug artifacts:

```text
data/debug/
    docling.json
    markdown.md
    evidence.json
    llm_request.json
    llm_response.json
    final_boq.json
    extraction_report.json
```

Do not log API keys.

Add structured logging for:

```text
PDF received
Docling started
Docling completed
pages processed
tables detected
evidence generated
LLM request started
LLM response received
validation started
validation completed
extraction completed
```

Add timing information.

### Commit

```bash
git add .
git commit -m "feat: add extraction observability and debug artifacts"
```

STOP.

---

# PHASE 9 — Docker + Production Readiness

### Goal

Containerize the standalone agent.

Add:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

The application should run as:

```bash
docker compose up --build
```

Expose:

```text
8000
```

Add health check.

Do not introduce PMS dependencies.

### Commit

```bash
git add .
git commit -m "chore: containerize BOQ extraction agent"
```

STOP.

---

# PHASE 10 — Documentation

Create a professional README containing:

## 1. Project Overview

## 2. Architecture

```text
PDF
 ↓
Docling
 ↓
Evidence Package
 ↓
GPT-OSS 120B
 ↓
Pydantic
 ↓
Validation
 ↓
BOQ JSON
```

## 3. Installation

## 4. Environment Variables

## 5. CLI Usage

## 6. API Usage

## 7. Example Input

## 8. Example Output

## 9. Supported PDF Types

## 10. Limitations

## 11. Error Handling

## 12. Project Structure

## 13. Development Phases

## 14. Future Improvements

### Commit

```bash
git add .
git commit -m "docs: document BOQ extraction agent"
```

STOP.

---

# AGENT PROMPT DESIGN

Create a strong system prompt for the BOQ extraction model.

The model should be instructed approximately as follows:

```text
You are a BOQ document extraction agent.

Your task is to convert document evidence extracted from a PDF into
a canonical structured BOQ.

The evidence comes from Docling.

You must extract only information supported by the evidence.

Do not invent values.

Do not infer missing quantities.

Do not invent units.

Do not invent item codes.

Do not invent item names.

Do not convert section headings into items.

Do not extract subtotal, total, grand total, carried-forward,
or summary rows as BOQ items.

Preserve the document's item code.

Build hierarchy using level_path.

A section heading such as:

2 EARTHWORKS

is a hierarchy node, not an item.

An item such as:

2.1 Excavation  m3  125.5

belongs to:

level_path = ["EARTHWORKS"]

If nested sections exist:

EARTHWORKS
    BULK EXCAVATION
        2.1 Excavation

then:

level_path = ["EARTHWORKS", "BULK EXCAVATION"]

Use fixed_rate when the document provides a direct rate
without a cost breakdown.

Do not emit amount or total fields.

Do not emit PMS database fields.

If a required field cannot be confidently extracted,
report the problem instead of guessing.

Return only the requested structured output.
```

Do not blindly use this text.

Adapt the final prompt to the actual implementation and schema.

---

# IMPORTANT: EVIDENCE PROVENANCE

Whenever possible, preserve where extracted information came from.

For internal debugging, an item may have evidence such as:

```text
page = 3
table = 1
row = 8
```

However, do NOT add these fields to the canonical BOQ JSON unless explicitly defined by the schema.

Keep provenance in the separate extraction/debug report.

---

# ERROR HANDLING

The system should distinguish:

## Document errors

```text
PDF unreadable
PDF corrupted
unsupported document
empty document
```

## Extraction errors

```text
no BOQ detected
table extraction failed
OCR failure
ambiguous table
```

## AI errors

```text
API failure
timeout
rate limit
invalid structured output
```

## Validation errors

```text
missing item code
missing quantity
missing unit
invalid hierarchy
duplicate code
invalid number
```

Do not hide these errors.

---

# COST AND PERFORMANCE

Do not send unnecessary content to GPT-OSS 120B.

Before the LLM call:

1. Process PDF with Docling.
2. Extract relevant evidence.
3. Remove clearly irrelevant content where safe.
4. Preserve document ordering.
5. Avoid sending duplicate representations unnecessarily.

Do not prematurely optimize at the expense of extraction correctness.

First establish correctness.

---

# SECURITY

Never hardcode:

```text
GROQ_API_KEY
```

Use `.env`.

Never log:

```text
GROQ_API_KEY
Authorization headers
secrets
```

Validate uploaded files.

Do not execute anything from the PDF.

Treat document contents as untrusted input.

---

# TESTING PHILOSOPHY

Use three testing layers.

## Unit Tests

Test:

```text
schemas
validators
evidence builder
normalizers
```

without external services.

## Integration Tests

Test:

```text
PDF → Docling → Evidence
```

using local fixtures.

## LLM Tests

Mock Groq responses for deterministic CI.

Keep a small optional live test for manual development:

```text
tests/live/
```

This test should only run when explicitly enabled.

---

# DEFINITION OF DONE

The project is considered complete when:

```text
[✓] PDF upload works
[✓] Docling processes the PDF
[✓] Evidence package is generated
[✓] BOQ sections are identified
[✓] BOQ items are identified
[✓] Hierarchy is reconstructed
[✓] GPT-OSS 120B restructures the evidence
[✓] Canonical JSON is validated
[✓] Missing required fields are reported
[✓] Totals/subtotals are ignored
[✓] Direct rates map to fixed_rate
[✓] Cost breakdowns are preserved
[✓] No PMS-internal fields are generated
[✓] CLI works
[✓] FastAPI works
[✓] Tests exist
[✓] Debug artifacts are available
[✓] Docker works
[✓] Documentation exists
```

---

# VERY IMPORTANT DEVELOPMENT RULE

Do not attempt to solve every PDF type immediately.

Start with a controlled structured BOQ PDF.

Then progressively introduce:

```text
structured PDF
      ↓
multi-page PDF
      ↓
complex tables
      ↓
scanned PDF
      ↓
OCR
      ↓
mixed PDF
      ↓
poorly structured PDF
```

This allows extraction failures to be isolated properly.

---

# FIRST TASK — START HERE

Before writing implementation code:

1. Inspect the current repository.
2. Confirm whether it is empty/new or already contains files.
3. Create a short architecture plan.
4. Confirm the phase boundaries.
5. Start ONLY with **PHASE 1**.
6. Implement the Docling foundation.
7. Test it with `data/input/sample_boq.pdf`.
8. Show the extracted Markdown/Docling output.
9. Show the project tree.
10. Explain the Phase 1 implementation.
11. Give the Git commit command.
12. STOP.

Do not implement Phase 2 or anything beyond Phase 1 until I explicitly say:

```text
Continue to Phase 2
```

The same rule applies to every subsequent phase.

---

# FINAL PRINCIPLE

This is an AI document-processing project, not simply an LLM wrapper.

The architecture should demonstrate:

```text
Document Intelligence
        +
Deterministic Processing
        +
LLM Semantic Structuring
        +
Schema Validation
        +
Error Handling
        +
Observability
```

The most important requirement is:

> **Docling extracts the document evidence; the AI Agent interprets and restructures that evidence into a validated canonical BOQ structure.**

Keep the project standalone, modular, testable, understandable, and commit-friendly.
