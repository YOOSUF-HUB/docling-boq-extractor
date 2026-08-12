# Docling BOQ Extraction Agent

A standalone document-AI service that turns a PDF **Bill of Quantities** into a
validated, canonical BOQ JSON structure.

```text
PDF → Docling → Evidence Package → GPT-OSS 120B → Pydantic → Validation → BOQ JSON
```

Docling does the document understanding. Python does the deterministic work.
The LLM only performs semantic restructuring of extracted evidence.

This project is **standalone** — it has no runtime dependency on the PMS
application. The PMS BOQ contract is used purely as a target output format.

> **Status: Phase 2 of 10 complete** — Docling processing and the evidence
> package. See [docs/architecture.md](docs/architecture.md) for the phase plan.

---

## Requirements

- Python 3.13
- ~2 GB disk for Docling's models (downloaded on first run)

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # GROQ_API_KEY only needed from Phase 4
```

## Usage

Generate the sample fixture (already committed, regenerate if needed):

```bash
python -m scripts.make_sample_pdf
```

Run the pipeline:

```bash
python -m app.main                                   # uses data/input/sample_boq.pdf
python -m app.main --input path/to/your_boq.pdf
python -m app.main --no-ocr                          # faster for text-only PDFs
python -m app.main --preview-lines 0                 # skip the markdown preview
```

Outputs:

```text
data/output/docling_output.md        Markdown rendering of the document
data/output/docling_document.json    full Docling document export
data/output/evidence.json            the agent-facing evidence package
```

plus a printed `DOCLING` and `EVIDENCE` summary.

Docling loads its layout/OCR models in a worker process that does its own
logging, so a few model-loading lines appear on stderr during the first
conversion. `--no-ocr` removes the OCR ones.

## The evidence package

`evidence.json` is what the LLM will be given in Phase 4 — not the raw Docling
JSON, which is ~15x larger and mostly geometry the model cannot use.

```text
EvidencePackage
├── document      filename, page count, extraction time, Docling status/version
├── pages[]       page-level metadata and per-page element counts
├── text_blocks[] non-table text, with label (section_header / text / …) and layer
├── tables[]      rows of cell text, header rows flagged, original row indices
└── statistics    counts used by the CLI and, later, the extraction report
```

Design decisions:

- **Content is never duplicated.** Text lives in `text_blocks`, table content in
  `tables`; `pages` carries metadata only.
- **`sequence` preserves reading order** across both lists. A heading that
  precedes a table is what tells the agent which section its rows belong to.
- **Provenance is preserved** — Docling `ref`s (`#/tables/0`), page numbers, and
  original `row_index` values, so an extracted BOQ item can be traced back to
  page/table/row in the extraction report.
- **Nothing is interpreted.** Whitespace is collapsed, empty text blocks and
  all-empty rows are dropped, and that is the extent of it. No number parsing,
  no item detection, no BOQ semantics — those belong to later phases.

Exit codes: `0` success, `1` document error (missing/empty/not a PDF),
`2` extraction error (Docling could not process the document).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq credentials (Phase 4+). Never commit it. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM used for restructuring |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DOCLING_DO_OCR` | `true` | OCR pages without a text layer |
| `DOCLING_DO_TABLE_STRUCTURE` | `true` | Reconstruct table rows/columns |
| `DOCLING_TABLE_CELL_MATCHING` | `true` | Match table cells to PDF text cells |
| `MAX_PDF_SIZE_MB` | `50` | Upload/size limit |

## Tests

```bash
pytest -m "not docling"    # fast unit tests (~4s, no models)
pytest -m docling          # real Docling run against the sample PDF (slow)
pytest                     # everything
```

Evidence tests run against captured Docling output
(`tests/fixtures/docling_sample_boq.json`) plus documents assembled in
`tests/factories.py`, so they need no PDF, no models and no network.

## Project structure

```text
app/
  config.py  logging_config.py  errors.py  main.py
  document/  pdf_validator.py  docling_processor.py  evidence_builder.py
  schemas/   evidence.py
  agent/  validation/  services/  api/                (later phases)
scripts/make_sample_pdf.py
tests/  data/  docs/
```
