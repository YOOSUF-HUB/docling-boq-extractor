"""Unit tests for the evidence package (no Docling models, no network).

Real-document tests run against `tests/fixtures/docling_sample_boq.json`, which
is genuine Docling output for `data/input/sample_boq.pdf`. Edge cases are built
with `tests/factories.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel

from app.document.evidence_builder import EvidenceBuilder, normalize_text
from app.schemas.evidence import EvidencePackage, TableEvidence
from tests.factories import (
    add_heading,
    add_paragraph,
    add_table,
    load_sample_boq_document,
    make_document,
    provenance,
)

EXTRACTED_AT = datetime(2026, 8, 12, 9, 14, tzinfo=UTC)


def build(document, filename="test.pdf", status="SUCCESS") -> EvidencePackage:
    return EvidenceBuilder().build(
        document,
        filename=filename,
        docling_status=status,
        extracted_at=EXTRACTED_AT,
    )


@pytest.fixture(scope="module")
def sample_evidence() -> EvidencePackage:
    return build(load_sample_boq_document(), filename="sample_boq.pdf")


# --- real document ----------------------------------------------------


def test_document_metadata(sample_evidence):
    document = sample_evidence.document
    assert document.filename == "sample_boq.pdf"
    assert document.page_count == 1
    assert document.docling_status == "SUCCESS"
    assert document.docling_schema_version
    assert document.extracted_at == EXTRACTED_AT


def test_text_blocks_are_extracted(sample_evidence):
    texts = [block.text for block in sample_evidence.text_blocks]
    assert "PROJECT: Sample Commercial Building" in texts
    assert "BILL OF QUANTITIES" in texts
    assert all(block.label == "section_header" for block in sample_evidence.text_blocks)
    assert all(block.page_number == 1 for block in sample_evidence.text_blocks)


def test_table_is_extracted_with_its_header(sample_evidence):
    assert len(sample_evidence.tables) == 1
    table = sample_evidence.tables[0]

    assert (table.num_rows, table.num_cols) == (10, 4)
    assert table.page_number == 1
    assert table.column_headers == ["Item", "Description", "Unit", "Quantity"]
    assert len(table.data_rows) == 9


def test_table_rows_preserve_document_values(sample_evidence):
    rows = {row.cells[0]: row.cells for row in sample_evidence.tables[0].data_rows}

    assert rows["1.1"] == ["1.1", "Excavation for foundations", "m3", "125.50"]
    assert rows["3.1"] == ["3.1", "Brickwork in cement mortar", "m2", "250.00"]
    # A section row keeps its empty unit/quantity cells rather than losing columns.
    assert rows["1"] == ["1", "EARTHWORKS", "", ""]


def test_document_order_is_preserved(sample_evidence):
    elements = sample_evidence.elements_in_document_order()
    sequences = [element.sequence for element in elements]

    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    # Both headings are read before the table they introduce.
    assert isinstance(elements[-1], TableEvidence)


def test_statistics_match_contents(sample_evidence):
    stats = sample_evidence.statistics
    assert stats.pages == len(sample_evidence.pages) == 1
    assert stats.text_blocks == len(sample_evidence.text_blocks) == 2
    assert stats.tables == 1
    assert stats.table_rows == 10
    assert stats.pictures == 0


def test_evidence_is_json_round_trippable(sample_evidence):
    restored = EvidencePackage.model_validate_json(sample_evidence.model_dump_json())
    assert restored == sample_evidence


# --- edge cases -------------------------------------------------------


def test_multiple_pages_are_counted_separately():
    document = make_document(page_count=3)
    add_heading(document, "EARTHWORKS", page_no=1)
    add_table(document, [["Item", "Qty"], ["1.1", "5"]], page_no=1)
    add_heading(document, "CONCRETE WORKS", page_no=2)
    add_paragraph(document, "Continued overleaf", page_no=2)

    evidence = build(document)

    assert [page.page_number for page in evidence.pages] == [1, 2, 3]
    page_1, page_2, page_3 = evidence.pages
    assert (page_1.text_block_count, page_1.table_count) == (1, 1)
    assert (page_2.text_block_count, page_2.table_count) == (2, 0)
    assert (page_3.text_block_count, page_3.table_count) == (0, 0)
    assert evidence.document.page_count == 3


def test_empty_table_is_kept_but_has_no_rows():
    document = make_document()
    add_table(document, [["", ""], ["", ""]], header_row=False)

    evidence = build(document)

    assert len(evidence.tables) == 1
    table = evidence.tables[0]
    assert table.num_rows == 2  # what Docling reported
    assert table.rows == []  # nothing worth sending to the agent
    assert table.column_headers == []
    assert evidence.statistics.table_rows == 0


def test_document_without_tables():
    document = make_document()
    add_heading(document, "PRELIMINARIES")
    add_paragraph(document, "This document contains no priced items.")

    evidence = build(document)

    assert evidence.tables == []
    assert evidence.has_tables is False
    assert evidence.statistics.tables == 0
    assert len(evidence.text_blocks) == 2


def test_blank_rows_are_dropped_but_row_index_is_not_renumbered():
    document = make_document()
    add_table(document, [["Item", "Qty"], ["", ""], ["1.1", "5"]])

    table = build(document).tables[0]

    assert [row.row_index for row in table.rows] == [0, 2]
    assert table.data_rows[0].cells == ["1.1", "5"]


def test_blank_text_blocks_are_dropped():
    document = make_document()
    add_paragraph(document, "   ")
    add_paragraph(document, "Real content")

    evidence = build(document)

    assert [block.text for block in evidence.text_blocks] == ["Real content"]


def test_furniture_layer_is_kept_and_labelled():
    document = make_document()
    document.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="Bill No. 2 — continued",
        prov=provenance(1),
        content_layer=ContentLayer.FURNITURE,
    )
    add_heading(document, "MASONRY")

    layers = {block.text: block.content_layer for block in build(document).text_blocks}

    assert layers["Bill No. 2 — continued"] == "furniture"
    assert layers["MASONRY"] == "body"


def test_text_is_normalized():
    document = make_document()
    add_paragraph(document, "Excavation\nfor   foundations\t(deep) ")

    assert build(document).text_blocks[0].text == "Excavation for foundations (deep)"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("m3", "m3"),
        (" 125.50 ", "125.50"),
        ("Reinforced\nconcrete", "Reinforced concrete"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected
