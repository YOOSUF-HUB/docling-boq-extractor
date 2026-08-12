"""CLI entry point.

Phase 1 scope:

    python -m app.main [--input data/input/sample_boq.pdf]

validates a PDF, runs it through Docling, writes the Markdown and Docling JSON
exports to `data/output/`, and prints a summary of what was found.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.document.docling_processor import DoclingProcessor, DoclingResult
from app.errors import DocumentError, ExtractionError
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)

MARKDOWN_FILENAME = "docling_output.md"
DOCUMENT_JSON_FILENAME = "docling_document.json"

EXIT_OK = 0
EXIT_DOCUMENT_ERROR = 1
EXIT_EXTRACTION_ERROR = 2


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
        "--log-level",
        default=settings.log_level,
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args(argv)


def write_outputs(result: DoclingResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / MARKDOWN_FILENAME
    markdown_path.write_text(result.markdown, encoding="utf-8")

    document_path = output_dir / DOCUMENT_JSON_FILENAME
    document_path.write_text(
        json.dumps(result.document_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {"markdown": markdown_path, "document_json": document_path}


def print_summary(result: DoclingResult, outputs: dict[str, Path], preview_lines: int) -> None:
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

    print()
    print("Outputs:")
    for path in outputs.values():
        print(f"  {path}")

    if preview_lines > 0 and result.markdown.strip():
        lines = result.markdown.splitlines()
        print()
        print(f"---------- MARKDOWN PREVIEW (first {min(preview_lines, len(lines))} lines) ----------")
        for line in lines[:preview_lines]:
            print(line)
        if len(lines) > preview_lines:
            print(f"... ({len(lines) - preview_lines} more lines in {outputs['markdown']})")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

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

    outputs = write_outputs(result, args.output_dir)
    print_summary(result, outputs, args.preview_lines)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
