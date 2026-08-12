"""Standard-library logging setup.

Kept intentionally small: one configuration entry point, called once from the
CLI (and later from the API startup). Modules just use `logging.getLogger(__name__)`.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Docling's model stack is chatty at INFO; keep our own logs readable.
_NOISY_LOGGERS = (
    "docling",
    "docling_core",
    "docling_ibm_models",
    "RapidOCR",
    "transformers",
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
)

# torch.compile emits multi-screen graph-break warnings while the layout model
# runs. They are informational and drown out the pipeline log.
_SILENCED_LOGGERS = ("torch._dynamo", "torch._inductor", "torch._logging")


def quiet_third_party_loggers() -> None:
    """Turn down third-party logging in this process.

    Note: Docling initialises its OCR/layout models in a worker that configures
    its own logging, so a few model-loading lines still reach stderr on the
    first conversion. `--no-ocr` avoids the OCR ones for text PDFs.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call more than once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
    quiet_third_party_loggers()
