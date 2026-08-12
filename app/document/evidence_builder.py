"""Build an `EvidencePackage` from a `DoclingDocument`.

    DoclingDocument -> EvidenceBuilder -> EvidencePackage

This is deterministic Python. It selects and normalizes; it never interprets.
The only content-affecting decisions it makes are:

* whitespace inside text and cells is collapsed,
* text blocks that are empty after normalization are dropped,
* table rows whose cells are all empty are dropped (their `row_index` gaps
  remain visible, so nothing is silently renumbered).

Both the body and furniture content layers are kept: BOQ page headers/footers
often carry the bill number or a "continued" marker, which is real evidence.
Each block records which layer it came from so downstream stages can weigh it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    PictureItem,
    TableItem,
    TextItem,
)

from app.document.docling_processor import DoclingResult
from app.schemas.evidence import (
    DocumentEvidence,
    EvidencePackage,
    EvidenceStatistics,
    PageEvidence,
    TableEvidence,
    TableRowEvidence,
    TextBlockEvidence,
)

logger = logging.getLogger(__name__)

INCLUDED_CONTENT_LAYERS = {ContentLayer.BODY, ContentLayer.FURNITURE}


def normalize_text(value: str | None) -> str:
    """Collapse whitespace and strip. Cell/​text values are compared literally
    downstream, so a stable representation matters more than the original
    line breaks."""
    if not value:
        return ""
    return " ".join(value.split())


def _page_number(item: DocItem) -> int | None:
    """Page of an item, taken from its first provenance entry."""
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


class EvidenceBuilder:
    """Turns a Docling document into the agent-facing evidence package."""

    def build(
        self,
        document: DoclingDocument,
        *,
        filename: str,
        docling_status: str | None = None,
        extracted_at: datetime | None = None,
    ) -> EvidencePackage:
        text_blocks: list[TextBlockEvidence] = []
        tables: list[TableEvidence] = []
        picture_pages: list[int | None] = []

        for sequence, (item, _level) in enumerate(
            document.iterate_items(included_content_layers=INCLUDED_CONTENT_LAYERS)
        ):
            if isinstance(item, TableItem):
                tables.append(
                    self._build_table(
                        item,
                        document=document,
                        sequence=sequence,
                        table_index=len(tables),
                    )
                )
            elif isinstance(item, PictureItem):
                picture_pages.append(_page_number(item))
            elif isinstance(item, TextItem):
                block = self._build_text_block(item, sequence=sequence)
                if block is not None:
                    text_blocks.append(block)

        pages = self._build_pages(document, text_blocks, tables, picture_pages)

        package = EvidencePackage(
            document=DocumentEvidence(
                filename=filename,
                page_count=len(document.pages or {}),
                extracted_at=extracted_at or datetime.now(UTC),
                docling_status=docling_status,
                docling_schema_version=str(getattr(document, "version", "")) or None,
            ),
            pages=pages,
            text_blocks=text_blocks,
            tables=tables,
            statistics=EvidenceStatistics(
                pages=len(pages),
                text_blocks=len(text_blocks),
                tables=len(tables),
                table_rows=sum(len(table.rows) for table in tables),
                pictures=len(picture_pages),
            ),
        )

        logger.info(
            "Evidence generated: %d pages, %d text blocks, %d tables, %d rows",
            package.statistics.pages,
            package.statistics.text_blocks,
            package.statistics.tables,
            package.statistics.table_rows,
        )
        return package

    # -- elements ------------------------------------------------------

    def _build_text_block(self, item: TextItem, *, sequence: int) -> TextBlockEvidence | None:
        text = normalize_text(item.text)
        if not text:
            return None

        return TextBlockEvidence(
            sequence=sequence,
            ref=item.self_ref,
            page_number=_page_number(item),
            label=str(getattr(item.label, "value", item.label)),
            content_layer=str(getattr(item.content_layer, "value", item.content_layer)),
            text=text,
            heading_level=getattr(item, "level", None),
        )

    def _build_table(
        self,
        item: TableItem,
        *,
        document: DoclingDocument,
        sequence: int,
        table_index: int,
    ) -> TableEvidence:
        data = item.data
        rows: list[TableRowEvidence] = []

        for row_index, grid_row in enumerate(getattr(data, "grid", []) or []):
            cells = [normalize_text(cell.text) for cell in grid_row]
            if not any(cells):
                continue  # separator / spacer rows carry no evidence
            rows.append(
                TableRowEvidence(
                    row_index=row_index,
                    is_header=any(cell.column_header for cell in grid_row),
                    cells=cells,
                )
            )

        caption = normalize_text(item.caption_text(document))

        return TableEvidence(
            sequence=sequence,
            ref=item.self_ref,
            table_index=table_index,
            page_number=_page_number(item),
            num_rows=getattr(data, "num_rows", 0) or 0,
            num_cols=getattr(data, "num_cols", 0) or 0,
            caption=caption or None,
            rows=rows,
        )

    # -- pages ---------------------------------------------------------

    @staticmethod
    def _build_pages(
        document: DoclingDocument,
        text_blocks: list[TextBlockEvidence],
        tables: list[TableEvidence],
        picture_pages: list[int | None],
    ) -> list[PageEvidence]:
        pages: dict[int, PageEvidence] = {}
        for page_no, page in sorted((document.pages or {}).items()):
            size = getattr(page, "size", None)
            pages[page_no] = PageEvidence(
                page_number=page_no,
                width=getattr(size, "width", None),
                height=getattr(size, "height", None),
            )

        def bump(page_number: int | None, field: str) -> None:
            if page_number is None:
                return
            page = pages.get(page_number)
            if page is None:  # element references a page Docling did not report
                page = PageEvidence(page_number=page_number)
                pages[page_number] = page
            setattr(page, field, getattr(page, field) + 1)

        for block in text_blocks:
            bump(block.page_number, "text_block_count")
        for table in tables:
            bump(table.page_number, "table_count")
        for page_number in picture_pages:
            bump(page_number, "picture_count")

        return [pages[key] for key in sorted(pages)]


def build_evidence_from_result(
    result: DoclingResult,
    *,
    extracted_at: datetime | None = None,
) -> EvidencePackage:
    """Convenience wrapper for the Phase 1 `DoclingResult`."""
    return EvidenceBuilder().build(
        result.document,
        filename=result.filename,
        docling_status=result.status,
        extracted_at=extracted_at,
    )
