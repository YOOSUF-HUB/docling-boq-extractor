"""Integration test: sample BOQ PDF -> Docling -> structured representation.

Marked `docling` because it loads the real models and is slow:

    pytest -m docling
    pytest -m "not docling"   # unit tests only
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.document.docling_processor import DoclingProcessor

pytestmark = pytest.mark.docling


@pytest.fixture(scope="module")
def result():
    settings = get_settings()
    pdf_path = settings.default_input_pdf
    if not pdf_path.exists():
        pytest.skip(f"{pdf_path} missing — run `python -m scripts.make_sample_pdf` first")
    return DoclingProcessor(settings=settings).process(pdf_path)


def test_conversion_succeeds(result):
    assert result.status == "SUCCESS"
    assert result.summary.page_count >= 1


def test_markdown_contains_document_text(result):
    markdown = result.markdown
    assert "BILL OF QUANTITIES" in markdown
    assert "Excavation for foundations" in markdown


def test_table_structure_is_detected(result):
    assert result.summary.table_count >= 1
    table = result.summary.tables[0]
    # Header + 3 sections + 6 items, across Item/Description/Unit/Quantity.
    assert table.num_rows >= 10
    assert table.num_cols == 4


def test_document_json_export_is_serializable(result):
    document = result.document_dict
    assert isinstance(document, dict)
    assert "texts" in document
    assert "tables" in document
