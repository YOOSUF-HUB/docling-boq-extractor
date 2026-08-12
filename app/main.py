"""CLI entry point.

Current scope (Phases 1-3):

    python -m app.main [--input data/input/sample_boq.pdf]

validates a PDF, runs it through Docling, builds the agent-facing evidence
package, writes all three artifacts to `data/output/`, and prints a summary.

    python -m app.main --validate-boq tests/fixtures/valid_boq.json

parses a canonical BOQ JSON document and applies the deterministic BOQ rules,
without touching Docling or the LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import get_settings
from app.document.docling_processor import DoclingProcessor, DoclingResult
from app.document.evidence_builder import build_evidence_from_result
from app.errors import DocumentError, ExtractionError
from app.logging_config import configure_logging
from app.schemas.boq import BOQDocument
from app.schemas.evidence import EvidencePackage, TableEvidence
from app.schemas.report import ExtractionReport
from app.validation.boq_validator import (
    failed_report,
    issues_from_validation_error,
    validate_boq,
)

logger = logging.getLogger(__name__)

MARKDOWN_FILENAME = "docling_output.md"
DOCUMENT_JSON_FILENAME = "docling_document.json"
EVIDENCE_FILENAME = "evidence.json"

EXIT_OK = 0
EXIT_DOCUMENT_ERROR = 1
EXIT_EXTRACTION_ERROR = 2
EXIT_VALIDATION_ERROR = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Process a BOQ PDF with Docling and export its structure.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.default_input_pdf,
        help="Path to the PDF to process (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_dir,
        help="Directory for the Docling exports (default: %(default)s)",
    )
    parser.add_argument(
        "--validate-boq",
        type=Path,
        metavar="PATH",
        help="Validate a canonical BOQ JSON file and exit (no Docling, no LLM).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (faster for text PDFs; scanned pages will be empty).",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=25,
        help="Lines of extracted Markdown to print (0 disables, default: %(default)s)",
    )
    parser.add_argument(
        "--preview-elements",
        type=int,
        default=12,
        help="Evidence elements to print in document order (default: %(default)s)",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Rows to print per previewed table (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args(argv)


def write_outputs(
    result: DoclingResult,
    evidence: EvidencePackage,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / MARKDOWN_FILENAME
    markdown_path.write_text(result.markdown, encoding="utf-8")

    document_path = output_dir / DOCUMENT_JSON_FILENAME
    document_path.write_text(
        json.dumps(result.document_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    evidence_path = output_dir / EVIDENCE_FILENAME
    evidence_path.write_text(
        evidence.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return {
        "markdown": markdown_path,
        "document_json": document_path,
        "evidence": evidence_path,
    }


def print_docling_summary(result: DoclingResult, markdown_path: Path, preview_lines: int) -> None:
    summary = result.summary
    print()
    print("========== DOCLING ==========")
    print(f"File          : {result.filename}")
    print(f"Status        : {result.status}")
    print(f"Duration      : {result.duration_seconds:.2f}s")
    print(f"Pages         : {summary.page_count}")
    print(f"Tables        : {summary.table_count}")
    print(f"Text blocks   : {summary.text_block_count}")
    print(f"Pictures      : {summary.picture_count}")

    if summary.tables:
        print()
        print("Detected tables:")
        for table in summary.tables:
            page = table.page_number if table.page_number is not None else "?"
            print(f"  [{table.index}] page {page} — {table.num_rows} rows x {table.num_cols} cols")

    if preview_lines > 0 and result.markdown.strip():
        lines = result.markdown.splitlines()
        print()
        print(f"---------- MARKDOWN PREVIEW (first {min(preview_lines, len(lines))} lines) ----------")
        for line in lines[:preview_lines]:
            print(line)
        if len(lines) > preview_lines:
            print(f"... ({len(lines) - preview_lines} more lines in {markdown_path})")


def print_evidence_summary(evidence: EvidencePackage, max_elements: int, max_rows: int) -> None:
    stats = evidence.statistics
    print()
    print("========== EVIDENCE ==========")
    print(f"Version       : {evidence.evidence_version}")
    print(f"Pages         : {stats.pages}")
    print(f"Text blocks   : {stats.text_blocks}")
    print(f"Tables        : {stats.tables}")
    print(f"Table rows    : {stats.table_rows}")
    print(f"Pictures      : {stats.pictures}")

    elements = evidence.elements_in_document_order()
    if not elements or max_elements <= 0:
        return

    print()
    print("Document order:")
    for element in elements[:max_elements]:
        page = element.page_number if element.page_number is not None else "?"
        if isinstance(element, TableEvidence):
            print(
                f"  [{element.sequence}] p{page} table#{element.table_index} "
                f"({element.num_rows}x{element.num_cols}, {len(element.rows)} non-empty rows)"
            )
            for row in element.rows[:max_rows]:
                marker = "H" if row.is_header else " "
                print(f"        {marker} r{row.row_index}: {' | '.join(row.cells)}")
            if len(element.rows) > max_rows:
                print(f"        … {len(element.rows) - max_rows} more rows")
        else:
            print(f"  [{element.sequence}] p{page} {element.label}: {element.text}")

    if len(elements) > max_elements:
        print(f"  … {len(elements) - max_elements} more elements")


def print_validation_report(report: ExtractionReport) -> None:
    print()
    print("========== VALIDATION ==========")
    print(f"Status        : {report.status.value}")
    stats = report.statistics
    print(f"Items         : {stats.items_extracted}")
    print(f"Sections      : {stats.sections_detected}")
    print(f"Errors        : {len(report.errors)}")
    print(f"Warnings      : {len(report.warnings)}")

    for label, issues in (("Errors", report.errors), ("Warnings", report.warnings)):
        if not issues:
            continue
        print()
        print(f"{label}:")
        for issue in issues:
            location = f" [item {issue.item_index}]" if issue.item_index is not None else ""
            print(f"  {issue.type.value}{location}: {issue.message}")
    print()


def run_boq_validation(path: Path) -> int:
    """`--validate-boq`: parse a canonical BOQ JSON file and apply the rules."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read BOQ JSON: %s", exc)
        return EXIT_DOCUMENT_ERROR

    try:
        boq = BOQDocument.model_validate(payload)
    except ValidationError as exc:
        report = failed_report(issues_from_validation_error(exc))
        print_validation_report(report)
        return EXIT_VALIDATION_ERROR

    report = validate_boq(boq)
    print_validation_report(report)
    return EXIT_OK if report.is_valid else EXIT_VALIDATION_ERROR


def print_outputs(outputs: dict[str, Path]) -> None:
    print()
    print("Outputs:")
    for path in outputs.values():
        print(f"  {path}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.validate_boq is not None:
        return run_boq_validation(args.validate_boq)

    settings = get_settings()
    if args.no_ocr:
        settings = settings.model_copy(update={"docling_do_ocr": False})

    logger.info("PDF received: %s", args.input)
    processor = DoclingProcessor(settings=settings)

    try:
        result = processor.process(args.input)
    except DocumentError as exc:
        logger.error("Document error: %s", exc)
        return EXIT_DOCUMENT_ERROR
    except ExtractionError as exc:
        logger.error("Extraction error: %s", exc)
        return EXIT_EXTRACTION_ERROR

    evidence = build_evidence_from_result(result)

    outputs = write_outputs(result, evidence, args.output_dir)
    print_docling_summary(result, outputs["markdown"], args.preview_lines)
    print_evidence_summary(evidence, args.preview_elements, args.preview_rows)
    print_outputs(outputs)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
