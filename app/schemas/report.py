"""Extraction report — diagnostics for this project only.

The report is deliberately kept *outside* the canonical BOQ JSON: it carries
statuses, issues, statistics and evidence provenance, none of which belong in a
PMS-compatible BOQ document.

**No confidence score.** A number like `0.94` would have to be invented — there
is no calibrated basis for it — so the report states an explicit status instead:

    success   the BOQ validated cleanly
    partial   items were extracted, but something needs a human look
    failed    the result cannot be trusted as a BOQ
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExtractionStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class IssueType(StrEnum):
    """Machine-readable issue codes, grouped by the stage that raises them."""

    # --- document ---
    DOCUMENT_UNREADABLE = "DOCUMENT_UNREADABLE"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"

    # --- extraction ---
    NO_BOQ_DETECTED = "NO_BOQ_DETECTED"
    NO_ITEMS_EXTRACTED = "NO_ITEMS_EXTRACTED"
    TABLE_EXTRACTION_FAILED = "TABLE_EXTRACTION_FAILED"

    # --- AI ---
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"

    # --- schema / validation ---
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    MISSING_LEVEL_PATH = "MISSING_LEVEL_PATH"
    MISSING_ITEM_CODE = "MISSING_ITEM_CODE"
    MISSING_ITEM_NAME = "MISSING_ITEM_NAME"
    MISSING_QUANTITY = "MISSING_QUANTITY"
    MISSING_UNIT = "MISSING_UNIT"
    DUPLICATE_ITEM_CODE = "DUPLICATE_ITEM_CODE"
    INVALID_HIERARCHY = "INVALID_HIERARCHY"
    INVALID_UNIT = "INVALID_UNIT"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    TOTAL_ROW_AS_ITEM = "TOTAL_ROW_AS_ITEM"
    AMBIGUOUS_RATE = "AMBIGUOUS_RATE"
    SUSPICIOUS_PERCENTAGE = "SUSPICIOUS_PERCENTAGE"


class EvidenceReference(BaseModel):
    """Where an issue came from. Debug-only — never part of the BOQ JSON."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    ref: str | None = None


class ExtractionIssue(BaseModel):
    """One warning or error."""

    model_config = ConfigDict(extra="forbid")

    type: IssueType
    message: str
    item_index: int | None = None
    item_code: str | None = None
    field: str | None = None
    evidence: EvidenceReference | None = None


class ExtractionStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages_processed: int = Field(default=0, ge=0)
    tables_detected: int = Field(default=0, ge=0)
    items_extracted: int = Field(default=0, ge=0)
    sections_detected: int = Field(default=0, ge=0)


class ExtractionReport(BaseModel):
    """Diagnostics accompanying (never embedded in) the canonical BOQ."""

    model_config = ConfigDict(extra="forbid")

    status: ExtractionStatus
    warnings: list[ExtractionIssue] = Field(default_factory=list)
    errors: list[ExtractionIssue] = Field(default_factory=list)
    statistics: ExtractionStatistics = Field(default_factory=ExtractionStatistics)

    @property
    def is_valid(self) -> bool:
        """True when nothing blocking was found (warnings are allowed)."""
        return not self.errors

    def summary_line(self) -> str:
        return (
            f"{self.status.value}: {self.statistics.items_extracted} items, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )
