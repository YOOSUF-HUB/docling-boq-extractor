"""Deterministic BOQ validation.

Validation happens in two layers, and it matters which does what:

1. **Pydantic** (`app/schemas/boq.py`) — structure. Types, enums, required
   fields, non-negative numbers, no PMS-internal fields. A failure here is a
   hard parse failure.
2. **This module** — everything structure cannot express: duplicate codes,
   totals masquerading as items, columns that look shifted, hierarchy smells.

Nothing here calls an LLM, and nothing here repairs data. It reports.

`validate_boq` returns an `ExtractionReport`; the caller decides what to do
with it. Errors mean the result must not be trusted as a BOQ; warnings mean a
human should look.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from pydantic import ValidationError

from app.schemas.boq import BOQDocument, BOQItem, ValueMode
from app.schemas.report import (
    ExtractionIssue,
    ExtractionReport,
    ExtractionStatistics,
    ExtractionStatus,
    IssueType,
)

logger = logging.getLogger(__name__)

#: Rows that summarise other rows are not BOQ items. Anchored at both ends so
#: that a genuine item such as "Total station survey" is not caught.
SUMMARY_ROW_PATTERN = re.compile(
    r"""^(
        (page|section|bill|sub|item)?[\s\-]*total(s)?(\s+(for|of)\b.*)?
        | grand\s+total(\s+(for|of)\b.*)?
        | (total\s+)?(carried|brought)\s+(forward|down)
        | c\s*/\s*f | b\s*/\s*f
        | collection | summary | subtotal
    )[\s:.\-]*$""",
    re.IGNORECASE | re.VERBOSE,
)

#: A unit cell that is really a number almost always means shifted columns.
NUMERIC_UNIT_PATTERN = re.compile(r"^[\d\s.,]+$")

#: Percentages above this are legal but worth a second look.
PERCENT_WARNING_THRESHOLD = 100.0


def is_summary_row_name(name: str) -> bool:
    """True if `name` reads as a total / subtotal / carried-forward row."""
    return bool(SUMMARY_ROW_PATTERN.match(name.strip()))


def _issue(
    issue_type: IssueType,
    message: str,
    *,
    item: BOQItem | None = None,
    index: int | None = None,
    field: str | None = None,
) -> ExtractionIssue:
    return ExtractionIssue(
        type=issue_type,
        message=message,
        item_index=index,
        item_code=item.boq_item_code if item else None,
        field=field,
    )


def validate_boq(
    boq: BOQDocument,
    statistics: ExtractionStatistics | None = None,
) -> ExtractionReport:
    """Apply the deterministic BOQ rules to an already schema-valid document."""
    errors: list[ExtractionIssue] = []
    warnings: list[ExtractionIssue] = []

    if not boq.items:
        errors.append(
            _issue(
                IssueType.NO_ITEMS_EXTRACTED,
                "No BOQ items were extracted from the document.",
            )
        )

    errors.extend(_check_duplicate_codes(boq))

    for index, item in enumerate(boq.items):
        item_errors, item_warnings = _check_item(item, index)
        errors.extend(item_errors)
        warnings.extend(item_warnings)

    stats = statistics.model_copy() if statistics else ExtractionStatistics()
    stats.items_extracted = len(boq.items)
    stats.sections_detected = len(boq.sections)

    report = ExtractionReport(
        status=derive_status(errors, warnings),
        errors=errors,
        warnings=warnings,
        statistics=stats,
    )
    logger.info("Validation completed — %s", report.summary_line())
    return report


def derive_status(
    errors: list[ExtractionIssue],
    warnings: list[ExtractionIssue],
) -> ExtractionStatus:
    if errors:
        return ExtractionStatus.FAILED
    if warnings:
        return ExtractionStatus.PARTIAL
    return ExtractionStatus.SUCCESS


# --- rules ------------------------------------------------------------


def _check_duplicate_codes(boq: BOQDocument) -> list[ExtractionIssue]:
    """Item codes identify rows; repeats mean rows were merged or invented."""
    counts = Counter(boq.item_codes)
    issues: list[ExtractionIssue] = []
    for code, count in counts.items():
        if count > 1:
            issues.append(
                ExtractionIssue(
                    type=IssueType.DUPLICATE_ITEM_CODE,
                    message=f"Item code '{code}' appears {count} times; codes must be unique.",
                    item_code=code,
                    field="boq_item_code",
                )
            )
    return issues


def _check_item(item: BOQItem, index: int) -> tuple[list[ExtractionIssue], list[ExtractionIssue]]:
    errors: list[ExtractionIssue] = []
    warnings: list[ExtractionIssue] = []

    # Totals / subtotals / carried-forward rows are not BOQ items.
    if is_summary_row_name(item.boq_item_name):
        errors.append(
            _issue(
                IssueType.TOTAL_ROW_AS_ITEM,
                f"Item '{item.boq_item_code}' is named '{item.boq_item_name}', "
                "which is a summary row, not a BOQ item.",
                item=item,
                index=index,
                field="boq_item_name",
            )
        )

    # A section heading was turned into a hierarchy node *and* an item name.
    if any(is_summary_row_name(segment) for segment in item.level_path):
        errors.append(
            _issue(
                IssueType.INVALID_HIERARCHY,
                f"level_path {item.level_path} contains a summary row heading.",
                item=item,
                index=index,
                field="level_path",
            )
        )

    # Numbering used as a section name means the heading text was lost.
    numeric_segments = [segment for segment in item.level_path if _looks_numeric(segment)]
    if numeric_segments:
        warnings.append(
            _issue(
                IssueType.INVALID_HIERARCHY,
                f"level_path {item.level_path} contains numbering "
                f"({', '.join(numeric_segments)}) instead of a section name.",
                item=item,
                index=index,
                field="level_path",
            )
        )

    if _looks_numeric(item.unit):
        warnings.append(
            _issue(
                IssueType.INVALID_UNIT,
                f"Unit '{item.unit}' for item '{item.boq_item_code}' is numeric, "
                "which usually means the columns are misaligned.",
                item=item,
                index=index,
                field="unit",
            )
        )

    if item.quantity == 0:
        warnings.append(
            _issue(
                IssueType.INVALID_QUANTITY,
                f"Item '{item.boq_item_code}' has a quantity of 0.",
                item=item,
                index=index,
                field="quantity",
            )
        )

    # A direct rate and a cost breakdown are alternatives, not a pair.
    if item.fixed_rate is not None and item.has_cost_breakdown:
        warnings.append(
            _issue(
                IssueType.AMBIGUOUS_RATE,
                f"Item '{item.boq_item_code}' has both a fixed_rate and a cost "
                "breakdown; the document should supply one or the other.",
                item=item,
                index=index,
                field="fixed_rate",
            )
        )

    for field_name in ("profit", "discount"):
        spec = getattr(item, field_name)
        if spec.mode is ValueMode.PERCENT and spec.value > PERCENT_WARNING_THRESHOLD:
            warnings.append(
                _issue(
                    IssueType.SUSPICIOUS_PERCENTAGE,
                    f"Item '{item.boq_item_code}' has {field_name} of {spec.value}%.",
                    item=item,
                    index=index,
                    field=field_name,
                )
            )

    return errors, warnings


def _looks_numeric(value: str) -> bool:
    return bool(NUMERIC_UNIT_PATTERN.match(value.strip()))


# --- schema failures --------------------------------------------------

#: Maps a required field to the issue code reported when it is missing/invalid.
_FIELD_ISSUE_TYPES = {
    "level_path": IssueType.MISSING_LEVEL_PATH,
    "boq_item_code": IssueType.MISSING_ITEM_CODE,
    "boq_item_name": IssueType.MISSING_ITEM_NAME,
    "quantity": IssueType.MISSING_QUANTITY,
    "unit": IssueType.MISSING_UNIT,
}


def issues_from_validation_error(error: ValidationError) -> list[ExtractionIssue]:
    """Turn a Pydantic `ValidationError` into reportable issues.

    Used when a payload (in Phase 4, the LLM response) does not satisfy the
    canonical schema. Required-field failures keep their specific issue code so
    "unit could not be extracted" stays distinguishable from "the JSON was
    malformed".
    """
    issues: list[ExtractionIssue] = []
    for detail in error.errors():
        location = [str(part) for part in detail["loc"]]
        field_name = next(
            (part for part in reversed(location) if part in _FIELD_ISSUE_TYPES),
            None,
        )
        item_index = next(
            (int(part) for part in location if part.isdigit()),
            None,
        )
        issues.append(
            ExtractionIssue(
                type=_FIELD_ISSUE_TYPES.get(field_name, IssueType.SCHEMA_VALIDATION_FAILED),
                message=f"{'.'.join(location) or 'payload'}: {detail['msg']}",
                item_index=item_index,
                field=field_name,
            )
        )
    return issues


def failed_report(
    issues: list[ExtractionIssue],
    statistics: ExtractionStatistics | None = None,
) -> ExtractionReport:
    """A `failed` report for a document that never reached BOQ validation."""
    return ExtractionReport(
        status=ExtractionStatus.FAILED,
        errors=issues,
        statistics=statistics or ExtractionStatistics(),
    )
