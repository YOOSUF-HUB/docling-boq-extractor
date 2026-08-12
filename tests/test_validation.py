"""Tests for the deterministic BOQ validator (no LLM, no Docling)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.boq import BOQDocument
from app.schemas.report import ExtractionStatistics, ExtractionStatus, IssueType
from app.validation.boq_validator import (
    failed_report,
    is_summary_row_name,
    issues_from_validation_error,
    validate_boq,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_boq(name: str) -> BOQDocument:
    return BOQDocument.model_validate(json.loads((FIXTURES / name).read_text()))


def boq_with(**item_overrides) -> BOQDocument:
    item = {
        "level_path": ["EARTHWORKS"],
        "boq_item_code": "1.1",
        "boq_item_name": "Excavation for foundations",
        "quantity": 125.5,
        "unit": "m3",
    } | item_overrides
    return BOQDocument.model_validate(
        {
            "document": {"filename": "x.pdf", "extracted_at": "2026-08-12T09:14:00Z"},
            "items": [item],
        }
    )


def issue_types(issues) -> set[IssueType]:
    return {issue.type for issue in issues}


# --- clean document ---------------------------------------------------


def test_valid_boq_validates_cleanly():
    report = validate_boq(load_boq("valid_boq.json"))

    assert report.status is ExtractionStatus.SUCCESS
    assert report.errors == []
    assert report.warnings == []
    assert report.is_valid is True


def test_statistics_are_filled_in():
    report = validate_boq(
        load_boq("valid_boq.json"),
        statistics=ExtractionStatistics(pages_processed=3, tables_detected=4),
    )

    assert report.statistics.pages_processed == 3
    assert report.statistics.tables_detected == 4
    assert report.statistics.items_extracted == 6
    assert report.statistics.sections_detected == 3


def test_passed_in_statistics_are_not_mutated():
    stats = ExtractionStatistics(pages_processed=3)
    validate_boq(load_boq("valid_boq.json"), statistics=stats)

    assert stats.items_extracted == 0


# --- errors -----------------------------------------------------------


def test_empty_boq_is_an_error():
    boq = BOQDocument.model_validate(
        {"document": {"filename": "x.pdf", "extracted_at": "2026-08-12T09:14:00Z"}, "items": []}
    )
    report = validate_boq(boq)

    assert report.status is ExtractionStatus.FAILED
    assert issue_types(report.errors) == {IssueType.NO_ITEMS_EXTRACTED}


def test_duplicate_item_codes_are_an_error():
    report = validate_boq(load_boq("invalid_boq_duplicate_codes.json"))

    assert report.status is ExtractionStatus.FAILED
    assert issue_types(report.errors) == {IssueType.DUPLICATE_ITEM_CODE}
    assert report.errors[0].item_code == "1.1"
    assert "appears 2 times" in report.errors[0].message


def test_summary_rows_are_rejected_as_items():
    report = validate_boq(load_boq("invalid_boq_total_row.json"))

    assert report.status is ExtractionStatus.FAILED
    assert issue_types(report.errors) == {IssueType.TOTAL_ROW_AS_ITEM}
    assert report.errors[0].item_code == "1.3"
    assert report.errors[0].item_index == 1


def test_summary_heading_in_level_path_is_an_error():
    report = validate_boq(boq_with(level_path=["EARTHWORKS", "SUBTOTAL"]))

    assert IssueType.INVALID_HIERARCHY in issue_types(report.errors)


@pytest.mark.parametrize(
    "name",
    [
        "Total",
        "TOTAL",
        "Sub-total",
        "Subtotal",
        "Grand Total",
        "Total carried forward",
        "Brought forward",
        "Page total",
        "Total for section 2",
        "Collection",
        "c/f",
        "TOTAL:",
    ],
)
def test_summary_row_names_are_detected(name):
    assert is_summary_row_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Total station survey",
        "Totalising flow meter",
        "Excavation for foundations",
        "Brickwork in cement mortar",
        "Summary sheet metal flashing",
        "Collection chamber, 900mm dia",
    ],
)
def test_genuine_items_are_not_mistaken_for_summary_rows(name):
    assert is_summary_row_name(name) is False


# --- warnings ---------------------------------------------------------


def test_shifted_columns_produce_warnings_not_errors():
    report = validate_boq(load_boq("invalid_boq_shifted_columns.json"))

    assert report.status is ExtractionStatus.PARTIAL
    assert report.errors == []
    assert issue_types(report.warnings) == {
        IssueType.INVALID_UNIT,
        IssueType.INVALID_QUANTITY,
        IssueType.INVALID_HIERARCHY,
    }
    assert report.is_valid is True


def test_zero_quantity_warns():
    report = validate_boq(boq_with(quantity=0.0))

    assert issue_types(report.warnings) == {IssueType.INVALID_QUANTITY}
    assert report.warnings[0].field == "quantity"


def test_numeric_unit_warns():
    report = validate_boq(boq_with(unit="125.50"))

    assert IssueType.INVALID_UNIT in issue_types(report.warnings)


def test_numeric_level_path_segment_warns():
    report = validate_boq(boq_with(level_path=["2"]))

    assert IssueType.INVALID_HIERARCHY in issue_types(report.warnings)


def test_fixed_rate_together_with_cost_breakdown_warns():
    report = validate_boq(boq_with(fixed_rate=250.0, labour=100.0))

    assert IssueType.AMBIGUOUS_RATE in issue_types(report.warnings)


def test_rate_only_item_does_not_warn():
    report = validate_boq(boq_with(fixed_rate=250000.0))

    assert report.status is ExtractionStatus.SUCCESS


def test_cost_breakdown_only_item_does_not_warn():
    report = validate_boq(boq_with(labour=5000.0, material=12000.0))

    assert report.status is ExtractionStatus.SUCCESS


def test_percentage_above_100_warns():
    report = validate_boq(boq_with(profit={"mode": "percent", "value": 150.0}))

    assert IssueType.SUSPICIOUS_PERCENTAGE in issue_types(report.warnings)
    assert report.warnings[0].field == "profit"


def test_amount_above_100_does_not_warn():
    report = validate_boq(boq_with(profit={"mode": "amount", "value": 5000.0}))

    assert report.status is ExtractionStatus.SUCCESS


# --- schema failures mapped into the report ---------------------------


def test_missing_unit_maps_to_its_own_issue_type():
    payload = json.loads((FIXTURES / "invalid_boq_missing_unit.json").read_text())
    with pytest.raises(ValidationError) as exc_info:
        BOQDocument.model_validate(payload)

    issues = issues_from_validation_error(exc_info.value)

    assert issue_types(issues) == {IssueType.MISSING_UNIT}
    assert issues[0].item_index == 0
    assert issues[0].field == "unit"


def test_pms_fields_map_to_a_schema_failure():
    payload = json.loads((FIXTURES / "invalid_boq_pms_fields.json").read_text())
    with pytest.raises(ValidationError) as exc_info:
        BOQDocument.model_validate(payload)

    issues = issues_from_validation_error(exc_info.value)

    assert issue_types(issues) == {IssueType.SCHEMA_VALIDATION_FAILED}
    assert any("temp_id" in issue.message for issue in issues)


def test_failed_report_carries_the_issues():
    payload = json.loads((FIXTURES / "invalid_boq_missing_quantity.json").read_text())
    with pytest.raises(ValidationError) as exc_info:
        BOQDocument.model_validate(payload)

    report = failed_report(issues_from_validation_error(exc_info.value))

    assert report.status is ExtractionStatus.FAILED
    assert report.is_valid is False
    assert issue_types(report.errors) == {IssueType.MISSING_QUANTITY}


def test_report_has_no_confidence_field():
    # Deliberate: an invented confidence number would be misleading.
    report = validate_boq(load_boq("valid_boq.json"))
    assert "confidence" not in report.model_dump()
