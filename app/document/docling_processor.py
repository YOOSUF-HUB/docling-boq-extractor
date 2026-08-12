"""Docling document processing layer.

This is the *document intelligence* stage of the pipeline:

    PDF -> validation -> Docling -> DoclingDocument (+ markdown / JSON exports)

Nothing here interprets BOQ semantics. It only produces a faithful, structured
representation of the document that later stages (evidence package, LLM
restructuring, validation) can reason about.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.config import Settings, get_settings
from app.errors import ExtractionError
from app.document.pdf_validator import validate_pdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSummary:
    """Shape of one detected table (no cell content — that is Phase 2's job)."""

    index: int
    page_number: int | None
    num_rows: int
    num_cols: int


@dataclass(frozen=True)
class DoclingSummary:
    """Counts describing what Docling found in the document."""

    page_count: int
    table_count: int
    text_block_count: int
    picture_count: int
    tables: list[TableSummary] = field(default_factory=list)


@dataclass
class DoclingResult:
    """Everything Phase 1 produces for a single PDF."""

    source_path: Path
    filename: str
    status: str
    markdown: str
    document_dict: dict[str, Any]
    summary: DoclingSummary
    duration_seconds: float
    document: Any = None  # the DoclingDocument itself, consumed in Phase 2


class DoclingProcessor:
    """Thin, explicit wrapper around Docling's `DocumentConverter`.

    The converter is built lazily and reused: constructing it loads the layout
    and table-structure models, which is expensive and should happen once per
    process rather than once per document.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._converter: DocumentConverter | None = None

    # -- converter -----------------------------------------------------

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.settings.docling_do_ocr
        pipeline_options.do_table_structure = self.settings.docling_do_table_structure
        pipeline_options.table_structure_options.do_cell_matching = (
            self.settings.docling_table_cell_matching
        )

        logger.info(
            "Building Docling converter (ocr=%s, table_structure=%s, cell_matching=%s)",
            pipeline_options.do_ocr,
            pipeline_options.do_table_structure,
            pipeline_options.table_structure_options.do_cell_matching,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    # -- processing ----------------------------------------------------

    def process(self, path: str | Path) -> DoclingResult:
        """Validate and convert a PDF into a `DoclingResult`.

        Raises `DocumentError` for bad input and `ExtractionError` when Docling
        cannot produce a usable document.
        """
        pdf_path = validate_pdf(path, self.settings.max_pdf_size_bytes)
        logger.info("Docling started: %s", pdf_path.name)

        started = time.perf_counter()
        try:
            conversion = self.converter.convert(pdf_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a pipeline error
            raise ExtractionError(f"Docling failed to convert {pdf_path.name}: {exc}") from exc
        duration = time.perf_counter() - started

        status = getattr(conversion.status, "name", str(conversion.status))
        if conversion.status == ConversionStatus.FAILURE:
            raise ExtractionError(
                f"Docling could not process {pdf_path.name} (status={status})"
            )
        if conversion.status != ConversionStatus.SUCCESS:
            logger.warning("Docling returned a non-success status: %s", status)

        document = conversion.document
        if document is None:
            raise ExtractionError(f"Docling returned no document for {pdf_path.name}")

        summary = self._summarize(document)
        logger.info(
            "Docling completed: %s in %.2fs (pages=%d, tables=%d, text_blocks=%d, pictures=%d)",
            pdf_path.name,
            duration,
            summary.page_count,
            summary.table_count,
            summary.text_block_count,
            summary.picture_count,
        )

        return DoclingResult(
            source_path=pdf_path,
            filename=pdf_path.name,
            status=status,
            markdown=document.export_to_markdown(),
            document_dict=document.export_to_dict(),
            summary=summary,
            duration_seconds=duration,
            document=document,
        )

    # -- summary -------------------------------------------------------

    @staticmethod
    def _summarize(document: Any) -> DoclingSummary:
        tables = list(getattr(document, "tables", []) or [])
        table_summaries: list[TableSummary] = []
        for index, table in enumerate(tables):
            data = getattr(table, "data", None)
            table_summaries.append(
                TableSummary(
                    index=index,
                    page_number=_first_page_number(table),
                    num_rows=getattr(data, "num_rows", 0) or 0,
                    num_cols=getattr(data, "num_cols", 0) or 0,
                )
            )

        return DoclingSummary(
            page_count=len(getattr(document, "pages", {}) or {}),
            table_count=len(tables),
            text_block_count=len(getattr(document, "texts", []) or []),
            picture_count=len(getattr(document, "pictures", []) or []),
            tables=table_summaries,
        )


def _first_page_number(item: Any) -> int | None:
    """Page number of a Docling item, via its first provenance entry."""
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)
