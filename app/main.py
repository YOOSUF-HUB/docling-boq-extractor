"""CLI entry point.

    python -m app.main [--input data/input/sample_boq.pdf]

runs the full pipeline — PDF -> Docling -> evidence -> GPT-OSS 120B -> canonical
BOQ -> validation — printing each stage and writing the artifacts to
`data/output/`.

    python -m app.main --evidence-only

stops after the evidence package, which needs no API key.

    python -m app.main --validate-boq tests/fixtures/valid_boq.json

applies the deterministic BOQ rules to an existing canonical BOQ JSON file,
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
from app.document.docling_processor import DoclingResult
from app.errors import AIRequestError, ConfigurationError, DocumentError, ExtractionError
from app.logging_config import configure_logging
from app.schemas.boq import BOQDocument
from app.schemas.evidence import EvidencePackage, TableEvidence
from app.schemas.report import ExtractionReport, IssueType
from app.services.extraction_service import (
    BOQResult,
    extract_boq_from_evidence,
    run_document_stage,
)
from app.validation.boq_validator import (
    failed_report,
    issues_from_validation_error,
    validate_boq,
)

logger = logging.getLogger(__name__)

MARKDOWN_FILENAME = "docling_output.md"
DOCUMENT_JSON_FILENAME = "docling_document.json"
EVIDENCE_FILENAME = "evidence.json"
BOQ_FILENAME = "boq.json"
REPORT_FILENAME = "extraction_report.json"

EXIT_OK = 0
EXIT_DOCUMENT_ERROR = 1
EXIT_EXTRACTION_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_CONFIG_ERROR = 4
EXIT_AI_ERROR = 5

#: Which exit code an error in the report maps to. Anything unlisted is a
#: failure of the extracted BOQ itself, i.e. EXIT_VALIDATION_ERROR.
_DOCUMENT_ISSUES = frozenset({IssueType.DOCUMENT_UNREADABLE, IssueType.EMPTY_DOCUMENT})
_EXTRACTION_ISSUES = frozenset(
    {
        IssueType.NO_BOQ_DETECTED,
        IssueType.NO_ITEMS_EXTRACTED,
        IssueType.TABLE_EXTRACTION_FAILED,
        IssueType.LLM_REQUEST_FAILED,
        IssueType.LLM_INVALID_OUTPUT,
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Extract a canonical Bill of Quantities from a PDF.",
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
        help="Directory for the extraction artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Stop after the evidence package (no LLM call, no API key needed).",
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
        "--preview-items",
        type=int,
        default=20,
        help="BOQ items to print in the final summary (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args(argv)


def write_outputs(
    docling: DoclingResult,
    evidence: EvidencePackage,
    output_dir: Path,
    boq_result: BOQResult | None = None,
) -> dict[str, Path]:
    """Write every artifact the run produced, in pipeline order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    markdown_path = output_dir / MARKDOWN_FILENAME
    markdown_path.write_text(docling.markdown, encoding="utf-8")
    outputs["markdown"] = markdown_path

    document_path = output_dir / DOCUMENT_JSON_FILENAME
    document_path.write_text(
        json.dumps(docling.document_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    outputs["document_json"] = document_path

    evidence_path = output_dir / EVIDENCE_FILENAME
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    outputs["evidence"] = evidence_path

    if boq_result is None:
        return outputs

    if boq_result.boq is not None:
        boq_path = output_dir / BOQ_FILENAME
        boq_path.write_text(boq_result.boq.model_dump_json(indent=2), encoding="utf-8")
        outputs["boq"] = boq_path

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(boq_result.report.model_dump_json(indent=2), encoding="utf-8")
    outputs["report"] = report_path

    return outputs


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


def print_agent_summary(result: BOQResult) -> None:
    print()
    print("========== AI BOQ ==========")
    llm = result.llm
    if llm is None:
        print("Model         : (not called — nothing in the document to interpret)")
        return

    print(f"Model         : {llm.model}")
    print(f"Attempts      : {llm.attempts}")
    print(f"Schema mode   : {'json_schema' if llm.used_json_schema else 'json_object'}")
    print(f"Duration      : {llm.duration_seconds:.2f}s")
    print(f"Tokens        : {llm.prompt_tokens} in / {llm.completion_tokens} out")
    print(f"Items         : {len(result.items)}")
    print(f"Unresolved    : {llm.unresolved_rows}")


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


def print_final_boq(result: BOQResult, max_items: int) -> None:
    print("========== FINAL BOQ ==========")
    boq = result.boq
    if boq is None:
        print("No canonical BOQ was produced — see the errors above.")
        return

    info = boq.document
    print(f"Version       : {boq.boq_source_version}")
    print(f"Document      : {info.filename}")
    print(f"Extracted at  : {info.extracted_at.isoformat()}")
    print(f"Items         : {len(boq.items)}")
    print(f"Sections      : {', '.join(boq.sections) or '—'}")

    if max_items <= 0 or not boq.items:
        return

    print()
    for item in boq.items[:max_items]:
        rate = "" if item.fixed_rate is None else f"  rate={item.fixed_rate}"
        print(
            f"  {' > '.join(item.level_path):<30} {item.boq_item_code:<8} "
            f"{item.boq_item_name:<38} {item.quantity:>10} {item.unit}{rate}"
        )
    if len(boq.items) > max_items:
        print(f"  … {len(boq.items) - max_items} more items")


def print_outputs(outputs: dict[str, Path]) -> None:
    print()
    print("Outputs:")
    for path in outputs.values():
        print(f"  {path}")
    print()


def exit_code_for(report: ExtractionReport) -> int:
    """Map a report onto a shell exit code, by the stage that failed."""
    if report.is_valid:
        return EXIT_OK

    types = {issue.type for issue in report.errors}
    if types & _DOCUMENT_ISSUES:
        return EXIT_DOCUMENT_ERROR
    if types & _EXTRACTION_ISSUES:
        return EXIT_EXTRACTION_ERROR
    return EXIT_VALIDATION_ERROR


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.validate_boq is not None:
        return run_boq_validation(args.validate_boq)

    settings = get_settings()
    if args.no_ocr:
        settings = settings.model_copy(update={"docling_do_ocr": False})

    logger.info("PDF received: %s", args.input)

    try:
        docling_result, evidence = run_document_stage(args.input, settings=settings)
    except DocumentError as exc:
        logger.error("Document error: %s", exc)
        return EXIT_DOCUMENT_ERROR
    except ExtractionError as exc:
        logger.error("Extraction error: %s", exc)
        return EXIT_EXTRACTION_ERROR

    print_docling_summary(docling_result, args.output_dir / MARKDOWN_FILENAME, args.preview_lines)
    print_evidence_summary(evidence, args.preview_elements, args.preview_rows)

    if args.evidence_only:
        print_outputs(write_outputs(docling_result, evidence, args.output_dir))
        return EXIT_OK

    try:
        result = extract_boq_from_evidence(evidence, settings=settings)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIG_ERROR
    except AIRequestError as exc:
        logger.error("The model could not be reached: %s", exc)
        return EXIT_AI_ERROR

    print_agent_summary(result)
    print_validation_report(result.report)
    print_final_boq(result, args.preview_items)
    print_outputs(write_outputs(docling_result, evidence, args.output_dir, result))
    return exit_code_for(result.report)


if __name__ == "__main__":
    sys.exit(main())
