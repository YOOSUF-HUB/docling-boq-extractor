"""Unit tests for deterministic PDF input validation (no Docling, no network)."""

from __future__ import annotations

import pytest

from app.document.pdf_validator import validate_pdf
from app.errors import DocumentError

MAX_SIZE = 10 * 1024 * 1024
MINIMAL_PDF = b"%PDF-1.7\n% minimal stub for validation tests\n"


def test_accepts_a_plausible_pdf(tmp_path):
    path = tmp_path / "ok.pdf"
    path.write_bytes(MINIMAL_PDF)

    assert validate_pdf(path, MAX_SIZE) == path.resolve()


def test_rejects_missing_file(tmp_path):
    with pytest.raises(DocumentError, match="File not found"):
        validate_pdf(tmp_path / "nope.pdf", MAX_SIZE)


def test_rejects_directory(tmp_path):
    with pytest.raises(DocumentError, match="Not a file"):
        validate_pdf(tmp_path, MAX_SIZE)


def test_rejects_wrong_extension(tmp_path):
    path = tmp_path / "boq.txt"
    path.write_bytes(MINIMAL_PDF)

    with pytest.raises(DocumentError, match="Unsupported file extension"):
        validate_pdf(path, MAX_SIZE)


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(DocumentError, match="empty"):
        validate_pdf(path, MAX_SIZE)


def test_rejects_oversized_file(tmp_path):
    path = tmp_path / "big.pdf"
    path.write_bytes(MINIMAL_PDF + b"0" * 2048)

    with pytest.raises(DocumentError, match="exceeds"):
        validate_pdf(path, 1024)


def test_rejects_file_without_pdf_header(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"this is not a pdf at all")

    with pytest.raises(DocumentError, match="PDF header"):
        validate_pdf(path, MAX_SIZE)
