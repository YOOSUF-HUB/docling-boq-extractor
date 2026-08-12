"""Deterministic PDF input validation.

This runs *before* Docling. It is cheap, has no dependencies, and keeps
unreadable input from reaching the (expensive) document conversion stage.
Document content is always treated as untrusted input: nothing here parses or
executes anything from the file, it only inspects the container.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.errors import DocumentError

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
ALLOWED_SUFFIXES = {".pdf"}


def validate_pdf(path: str | Path, max_size_bytes: int) -> Path:
    """Validate that `path` is a readable, non-empty, plausible PDF.

    Returns the resolved path, or raises `DocumentError` describing the problem.
    """
    pdf_path = Path(path).expanduser().resolve()

    if not pdf_path.exists():
        raise DocumentError(f"File not found: {pdf_path}")
    if not pdf_path.is_file():
        raise DocumentError(f"Not a file: {pdf_path}")
    if pdf_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise DocumentError(
            f"Unsupported file extension '{pdf_path.suffix}'. Expected one of: "
            f"{', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    size = pdf_path.stat().st_size
    if size == 0:
        raise DocumentError(f"File is empty: {pdf_path}")
    if size > max_size_bytes:
        raise DocumentError(
            f"File is {size / 1024 / 1024:.1f} MB, which exceeds the "
            f"{max_size_bytes / 1024 / 1024:.0f} MB limit"
        )

    try:
        with pdf_path.open("rb") as handle:
            header = handle.read(len(PDF_MAGIC))
    except OSError as exc:
        raise DocumentError(f"File could not be read: {pdf_path} ({exc})") from exc

    if header != PDF_MAGIC:
        raise DocumentError(
            f"File does not start with a PDF header ({PDF_MAGIC.decode()}): {pdf_path}"
        )

    logger.debug("PDF validated: %s (%d bytes)", pdf_path.name, size)
    return pdf_path
