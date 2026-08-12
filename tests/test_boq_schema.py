"""Schema-level tests for the canonical BOQ contract.

These cover what Pydantic must enforce on its own: required fields, types,
enums, numeric ranges, and the refusal of PMS-internal fields.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.boq import (
    BOQ_SOURCE_VERSION,
    BOQDocument,
    BOQItem,
    CostType,
    ValueMode,
    ValueSpec,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def valid_boq() -> BOQDocument:
    return BOQDocument.model_validate(load_fixture("valid_boq.json"))


# --- the authored valid fixture ---------------------------------------


def test_valid_fixture_parses(valid_boq):
    assert valid_boq.boq_source_version == BOQ_SOURCE_VERSION
    assert valid_boq.document.filename == "sample_boq.pdf"
    assert valid_boq.document.extracted_at == datetime(2026, 8, 12, 9, 14, tzinfo=UTC)
    assert len(valid_boq.items) == 6


def test_valid_fixture_hierarchy(valid_boq):
    assert valid_boq.sections == ["EARTHWORKS", "CONCRETE WORKS", "MASONRY"]
    assert ("CONCRETE WORKS", "SUPERSTRUCTURE") in valid_boq.level_paths
    assert valid_boq.item_codes == ["1.1", "1.2", "2.1", "2.2", "2.3.1", "3.1"]


def test_valid_fixture_rate_and_cost_breakdown(valid_boq):
    rate_only = next(item for item in valid_boq.items if item.boq_item_code == "2.2")
    assert rate_only.fixed_rate == 42500.0
    assert rate_only.has_cost_breakdown is False

    broken_down = next(item for item in valid_boq.items if item.boq_item_code == "2.3.1")
    assert broken_down.fixed_rate is None
    assert broken_down.has_cost_breakdown is True
    assert broken_down.cost_breakdown["material"] == 12000.0
    assert broken_down.profit == ValueSpec(mode=ValueMode.PERCENT, value=10.0)


def test_valid_fixture_round_trips(valid_boq):
    restored = BOQDocument.model_validate_json(valid_boq.model_dump_json())
    assert restored == valid_boq


def test_extracted_at_serializes_as_utc(valid_boq):
    payload = json.loads(valid_boq.model_dump_json())
    assert payload["document"]["extracted_at"] == "2026-08-12T09:14:00Z"


# --- required fields must fail loudly ---------------------------------


@pytest.mark.parametrize(
    ("fixture_name", "expected_field"),
    [
        ("invalid_boq_missing_unit.json", "unit"),
        ("invalid_boq_missing_quantity.json", "quantity"),
        ("invalid_boq_empty_item_name.json", "boq_item_name"),
        ("invalid_boq_empty_level_path.json", "level_path"),
        ("invalid_boq_negative_quantity.json", "quantity"),
        ("invalid_boq_bad_enum.json", "cost_type"),
    ],
)
def test_invalid_fixtures_are_rejected(fixture_name, expected_field):
    with pytest.raises(ValidationError) as exc_info:
        BOQDocument.model_validate(load_fixture(fixture_name))

    failed_fields = {str(detail["loc"][-1]) for detail in exc_info.value.errors()}
    assert expected_field in failed_fields


def test_pms_internal_fields_are_rejected():
    with pytest.raises(ValidationError) as exc_info:
        BOQDocument.model_validate(load_fixture("invalid_boq_pms_fields.json"))

    rejected = {str(detail["loc"][-1]) for detail in exc_info.value.errors()}
    assert {"id", "temp_id", "project_boq_item_code", "created_at"} <= rejected
    assert all(detail["type"] == "extra_forbidden" for detail in exc_info.value.errors())


def test_unknown_top_level_fields_are_rejected():
    payload = load_fixture("valid_boq.json") | {"project_id": 17}
    with pytest.raises(ValidationError):
        BOQDocument.model_validate(payload)


# --- field-level rules ------------------------------------------------


def minimal_item(**overrides) -> dict:
    return {
        "level_path": ["EARTHWORKS"],
        "boq_item_code": "1.1",
        "boq_item_name": "Excavation for foundations",
        "quantity": 125.5,
        "unit": "m3",
    } | overrides


def test_defaults_match_the_contract():
    item = BOQItem.model_validate(minimal_item())

    assert item.boq_description == ""
    assert item.cost_type is CostType.PER_UNIT
    assert all(value == 0.0 for value in item.cost_breakdown.values())
    assert item.profit == ValueSpec(mode=ValueMode.AMOUNT, value=0.0)
    assert item.discount == ValueSpec(mode=ValueMode.AMOUNT, value=0.0)
    assert item.fixed_rate is None


def test_text_fields_are_stripped():
    item = BOQItem.model_validate(
        minimal_item(
            boq_item_code="  1.1  ",
            boq_item_name="  Excavation  ",
            unit=" m3 ",
            boq_description="  fill  ",
            level_path=["  EARTHWORKS  "],
        )
    )

    assert item.boq_item_code == "1.1"
    assert item.boq_item_name == "Excavation"
    assert item.unit == "m3"
    assert item.boq_description == "fill"
    assert item.level_path == ["EARTHWORKS"]


@pytest.mark.parametrize("bad_code", ["", "   "])
def test_blank_item_code_is_rejected(bad_code):
    with pytest.raises(ValidationError, match="boq_item_code"):
        BOQItem.model_validate(minimal_item(boq_item_code=bad_code))


def test_blank_level_path_segment_is_rejected():
    with pytest.raises(ValidationError, match="empty segments"):
        BOQItem.model_validate(minimal_item(level_path=["EARTHWORKS", "  "]))


@pytest.mark.parametrize("field_name", BOQItem.COST_FIELDS)
def test_cost_fields_reject_negatives(field_name):
    with pytest.raises(ValidationError, match=field_name):
        BOQItem.model_validate(minimal_item(**{field_name: -1.0}))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_numbers_are_rejected(bad_value):
    with pytest.raises(ValidationError):
        BOQItem.model_validate(minimal_item(quantity=bad_value))


def test_negative_fixed_rate_is_rejected():
    with pytest.raises(ValidationError, match="fixed_rate"):
        BOQItem.model_validate(minimal_item(fixed_rate=-1.0))


def test_zero_quantity_is_allowed_by_the_schema():
    # Whether a zero quantity is acceptable is a judgement call, so it belongs
    # to the validator (as a warning), not to the schema.
    assert BOQItem.model_validate(minimal_item(quantity=0.0)).quantity == 0.0


def test_value_spec_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        ValueSpec.model_validate({"mode": "ratio", "value": 1.0})


def test_value_spec_rejects_negative_value():
    with pytest.raises(ValidationError):
        ValueSpec.model_validate({"mode": "percent", "value": -10.0})


def test_naive_timestamp_is_read_as_utc():
    document = BOQDocument.model_validate(
        {
            "document": {"filename": "x.pdf", "extracted_at": "2026-08-12T09:14:00"},
            "items": [],
        }
    )
    assert document.document.extracted_at == datetime(2026, 8, 12, 9, 14, tzinfo=UTC)


def test_offset_timestamp_is_normalized_to_utc():
    document = BOQDocument.model_validate(
        {
            "document": {"filename": "x.pdf", "extracted_at": "2026-08-12T14:44:00+05:30"},
            "items": [],
        }
    )
    assert document.document.extracted_at == datetime(2026, 8, 12, 9, 14, tzinfo=UTC)
