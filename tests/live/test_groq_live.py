"""Opt-in live test against the real Groq API.

Skipped unless both are true:

    RUN_LIVE_TESTS=1        set explicitly
    GROQ_API_KEY            configured in the environment or .env

Run it with:

    RUN_LIVE_TESTS=1 pytest -m live tests/live

It costs tokens and is not deterministic, so it never runs in the default suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agent.boq_agent import BOQAgent
from app.config import get_settings
from app.schemas.boq import BOQDocument
from app.schemas.evidence import EvidencePackage
from app.schemas.report import ExtractionStatus
from app.validation.boq_validator import validate_boq

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("RUN_LIVE_TESTS"),
        reason="live tests are opt-in; set RUN_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        not get_settings().groq_api_key,
        reason="GROQ_API_KEY is not configured",
    ),
]

EVIDENCE = Path(__file__).parent.parent / "fixtures" / "evidence_sample_boq.json"


@pytest.fixture(scope="module")
def result():
    evidence = EvidencePackage.model_validate_json(EVIDENCE.read_text())
    return BOQAgent().extract(evidence), evidence


def test_agent_extracts_the_sample_boq(result):
    agent_result, _ = result

    assert agent_result.attempts >= 1
    assert len(agent_result.items) == 6
    assert agent_result.items[0].boq_item_code == "1.1"


def test_section_rows_are_not_extracted_as_items(result):
    agent_result, _ = result
    codes = [item.boq_item_code for item in agent_result.items]

    # "1", "2" and "3" are section headings, not items.
    assert codes == ["1.1", "1.2", "2.1", "2.2", "2.3", "3.1"]


def test_hierarchy_uses_section_names(result):
    agent_result, _ = result
    paths = {item.boq_item_code: item.level_path for item in agent_result.items}

    assert paths["1.1"] == ["EARTHWORKS"]
    assert paths["2.1"] == ["CONCRETE WORKS"]
    assert paths["3.1"] == ["MASONRY"]


def test_document_values_are_preserved(result):
    agent_result, _ = result
    item = next(item for item in agent_result.items if item.boq_item_code == "1.1")

    assert item.quantity == 125.5
    assert item.unit == "m3"
    assert item.boq_item_name == "Excavation for foundations"


def test_no_rates_are_invented_when_the_document_has_none(result):
    agent_result, _ = result

    assert all(item.fixed_rate is None for item in agent_result.items)
    assert all(not item.has_cost_breakdown for item in agent_result.items)


def test_extraction_validates_cleanly(result):
    agent_result, evidence = result
    boq = BOQDocument.assemble(agent_result.items, filename=evidence.document.filename)

    report = validate_boq(boq)

    assert report.status is ExtractionStatus.SUCCESS
    assert report.errors == []
