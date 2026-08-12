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

> **Status: Phase 1 of 10 complete** — project foundation and Docling
> processing. See [docs/architecture.md](docs/architecture.md) for the phase plan.

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

## Usage (Phase 1)

Generate the sample fixture (already committed, regenerate if needed):

```bash
python -m scripts.make_sample_pdf
```

Run the Docling stage:

```bash
python -m app.main                                   # uses data/input/sample_boq.pdf
python -m app.main --input path/to/your_boq.pdf
python -m app.main --no-ocr                          # faster for text-only PDFs
```

Outputs:

```text
data/output/docling_output.md        Markdown rendering of the document
data/output/docling_document.json    full Docling document export
```

plus a printed summary of pages, tables, text blocks and pictures.

Docling loads its layout/OCR models in a worker process that does its own
logging, so a few model-loading lines appear on stderr during the first
conversion. `--no-ocr` removes the OCR ones.

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
pytest -m "not docling"    # fast unit tests
pytest -m docling          # real Docling run against the sample PDF (slow)
pytest                     # everything
```

## Project structure

```text
app/
  config.py  logging_config.py  errors.py  main.py
  document/  pdf_validator.py  docling_processor.py
  schemas/  agent/  validation/  services/  api/     (later phases)
scripts/make_sample_pdf.py
tests/  data/  docs/
```
