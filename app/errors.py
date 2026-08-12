"""Error hierarchy for the extraction pipeline.

The pipeline deliberately distinguishes *where* a failure came from, so the
extraction report (and later the API layer) can tell a user whether the
document, the extraction, the LLM, or validation was at fault.
"""


class BOQExtractionError(Exception):
    """Base class for every error raised by this application."""


class DocumentError(BOQExtractionError):
    """The input document is missing, unreadable, or not a usable PDF."""


class ExtractionError(BOQExtractionError):
    """Docling ran but could not produce a usable document representation."""
