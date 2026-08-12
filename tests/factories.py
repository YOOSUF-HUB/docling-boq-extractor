"""Helpers for building `DoclingDocument` instances in tests.

Docling's own output is captured once in `tests/fixtures/docling_sample_boq.json`;
everything else (multi-page documents, empty tables, documents without tables)
is assembled here so the evidence tests stay fast and deterministic — no model
loading, no PDFs, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_BOQ_DOCLING_JSON = FIXTURES_DIR / "docling_sample_boq.json"

A4 = Size(width=595.0, height=842.0)


def load_sample_boq_document() -> DoclingDocument:
    """The real Docling output for `data/input/sample_boq.pdf`."""
    return DoclingDocument.model_validate(json.loads(SAMPLE_BOQ_DOCLING_JSON.read_text()))


def provenance(page_no: int) -> ProvenanceItem:
    """Minimal provenance; only `page_no` is read by the evidence builder."""
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=0, t=10, r=100, b=0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 0),
    )


def make_document(name: str = "test_doc", page_count: int = 1) -> DoclingDocument:
    document = DoclingDocument(name=name)
    for page_no in range(1, page_count + 1):
        document.add_page(page_no=page_no, size=A4)
    return document


def add_heading(document: DoclingDocument, text: str, page_no: int = 1, level: int = 1) -> None:
    document.add_heading(text=text, level=level, prov=provenance(page_no))


def add_paragraph(document: DoclingDocument, text: str, page_no: int = 1) -> None:
    document.add_text(label=DocItemLabel.TEXT, text=text, prov=provenance(page_no))


def make_table_data(rows: list[list[str]], header_row: bool = True) -> TableData:
    """Build `TableData` from a rectangular list of strings."""
    num_rows = len(rows)
    num_cols = max((len(row) for row in rows), default=0)

    cells: list[TableCell] = []
    for row_index, row in enumerate(rows):
        for col_index, text in enumerate(row):
            cells.append(
                TableCell(
                    text=text,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + 1,
                    column_header=header_row and row_index == 0,
                )
            )

    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def add_table(
    document: DoclingDocument,
    rows: list[list[str]],
    page_no: int = 1,
    header_row: bool = True,
) -> None:
    document.add_table(
        data=make_table_data(rows, header_row=header_row),
        prov=provenance(page_no),
    )
