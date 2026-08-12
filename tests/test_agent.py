"""Agent tests with mocked LLM responses — no network, no API key needed.

The live counterpart is `tests/live/test_groq_live.py` (opt-in) and
`python -m app.agent.test_agent` (manual).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import groq
import httpx
import pytest

from app.agent.boq_agent import BOQAgent, issues_from_unresolved
from app.agent.groq_client import GroqClient, LLMResponse
from app.agent.prompts import (
    BOQ_RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_user_prompt,
)
from app.config import Settings
from app.errors import AIRequestError, AIResponseError, ConfigurationError
from app.schemas.boq import BOQItem, UnresolvedItem
from app.schemas.evidence import (
    DocumentEvidence,
    EvidencePackage,
    TableEvidence,
    TableRowEvidence,
)
from app.schemas.report import IssueType

FIXTURES = Path(__file__).parent / "fixtures"

SETTINGS = Settings(groq_api_key="test-key", llm_max_attempts=2)


@pytest.fixture(scope="module")
def evidence() -> EvidencePackage:
    return EvidencePackage.model_validate_json(
        (FIXTURES / "evidence_sample_boq.json").read_text()
    )


def item_payload(**overrides) -> dict:
    payload = {
        "level_path": ["EARTHWORKS"],
        "boq_item_code": "1.1",
        "boq_item_name": "Excavation for foundations",
        "boq_description": "",
        "quantity": 125.5,
        "unit": "m3",
        "cost_type": "per_unit",
        "labour": 0.0,
        "machine": 0.0,
        "material": 0.0,
        "fuel": 0.0,
        "miscellaneous": 0.0,
        "subcontract": 0.0,
        "site_overhead": 0.0,
        "head_office_overhead": 0.0,
        "profit": {"mode": "amount", "value": 0.0},
        "discount": {"mode": "amount", "value": 0.0},
        "fixed_rate": None,
    }
    return payload | overrides


def response_payload(items=None, unresolved=None) -> str:
    return json.dumps(
        {
            "items": items if items is not None else [item_payload()],
            "unresolved": unresolved or [],
        }
    )


class FakeClient:
    """Stands in for `GroqClient`, returning canned completions in order."""

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict[str, str]]] = []

    def complete_json(self, messages, *, json_schema=None, schema_name="response"):
        self.calls.append(messages)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return LLMResponse(
            content=content,
            model="openai/gpt-oss-120b",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=200,
            duration_seconds=0.5,
            used_json_schema=json_schema is not None,
        )


def make_agent(*contents: str) -> tuple[BOQAgent, FakeClient]:
    client = FakeClient(*contents)
    return BOQAgent(client=client, settings=SETTINGS), client


# --- happy path -------------------------------------------------------


def test_valid_response_is_accepted(evidence):
    agent, client = make_agent(response_payload())

    result = agent.extract(evidence)

    assert len(client.calls) == 1
    assert result.attempts == 1
    assert result.used_json_schema is True
    assert result.warnings == []
    assert result.errors == []
    assert result.items[0].boq_item_code == "1.1"
    assert result.items[0].quantity == 125.5


def test_prompt_sent_contains_system_rules_and_evidence(evidence):
    agent, client = make_agent(response_payload())
    agent.extract(evidence)

    system, user = client.calls[0]
    assert system["role"] == "system"
    assert system["content"] == SYSTEM_PROMPT
    assert "Excavation for foundations" in user["content"]


def test_items_are_returned_as_validated_models(evidence):
    agent, _ = make_agent(response_payload(items=[item_payload(fixed_rate=250000.0)]))

    item = agent.extract(evidence).items[0]

    assert isinstance(item, BOQItem)
    assert item.fixed_rate == 250000.0
    assert item.has_cost_breakdown is False


# --- the model reports what it could not extract -----------------------


def test_unresolved_rows_become_warnings(evidence):
    agent, _ = make_agent(
        response_payload(
            unresolved=[
                {
                    "boq_item_code": "2.2",
                    "description": "Reinforced concrete foundations",
                    "missing_fields": ["unit"],
                    "reason": "The unit column was empty for this row.",
                    "evidence_ref": "table#0 row 6",
                }
            ]
        )
    )

    result = agent.extract(evidence)

    assert len(result.unresolved) == 1
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.type is IssueType.MISSING_UNIT
    assert warning.item_code == "2.2"
    assert "unit could not be confidently identified" in warning.message.lower()


def test_one_warning_per_missing_field():
    warnings = issues_from_unresolved(
        [
            UnresolvedItem(
                boq_item_code="4.1",
                missing_fields=["quantity", "unit"],
                reason="Row spans a page break.",
            )
        ]
    )

    assert [warning.type for warning in warnings] == [
        IssueType.MISSING_QUANTITY,
        IssueType.MISSING_UNIT,
    ]


def test_unresolved_without_named_fields_still_reports():
    warnings = issues_from_unresolved(
        [UnresolvedItem(description="illegible row", reason="OCR produced no text.")]
    )

    assert warnings[0].type is IssueType.UNRESOLVED_ROW
    assert "illegible row" in warnings[0].message


# --- invalid output: controlled retry, never repair --------------------


def test_malformed_json_triggers_one_repair_attempt(evidence):
    agent, client = make_agent("this is not JSON at all", response_payload())

    result = agent.extract(evidence)

    assert result.attempts == 2
    assert len(client.calls) == 2
    # The repair round shows the model its own output and the errors.
    repair_messages = client.calls[1]
    assert repair_messages[2]["role"] == "assistant"
    assert repair_messages[2]["content"] == "this is not JSON at all"
    assert "did not satisfy the required schema" in repair_messages[3]["content"]


def test_schema_violation_triggers_a_repair_attempt(evidence):
    invalid = json.dumps({"items": [{"boq_item_code": "1.1"}], "unresolved": []})
    agent, client = make_agent(invalid, response_payload())

    result = agent.extract(evidence)

    assert result.attempts == 2
    assert len(result.items) == 1


def test_persistently_invalid_output_fails_loudly(evidence):
    agent, client = make_agent("{not json")

    with pytest.raises(AIResponseError) as exc_info:
        agent.extract(evidence)

    assert len(client.calls) == SETTINGS.llm_max_attempts
    assert exc_info.value.issues
    assert exc_info.value.issues[0].type is IssueType.LLM_INVALID_OUTPUT


def test_missing_required_field_is_reported_specifically(evidence):
    payload = json.dumps(
        {"items": [item_payload(unit=None)], "unresolved": []},
    )
    agent, _ = make_agent(payload)

    with pytest.raises(AIResponseError) as exc_info:
        agent.extract(evidence)

    assert {issue.type for issue in exc_info.value.issues} == {IssueType.MISSING_UNIT}


def test_pms_fields_in_the_response_are_rejected(evidence):
    payload = json.dumps(
        {"items": [item_payload(id=42, temp_id="tmp_1")], "unresolved": []},
    )
    agent, _ = make_agent(payload)

    with pytest.raises(AIResponseError) as exc_info:
        agent.extract(evidence)

    assert {issue.type for issue in exc_info.value.issues} == {
        IssueType.SCHEMA_VALIDATION_FAILED
    }


def test_hallucinated_extra_field_is_rejected(evidence):
    payload = json.dumps(
        {"items": [item_payload(amount=56475.0)], "unresolved": []},
    )
    agent, _ = make_agent(payload)

    with pytest.raises(AIResponseError):
        agent.extract(evidence)


# --- the LLM is not called when there is nothing to extract ------------


def test_empty_evidence_skips_the_llm():
    empty = EvidencePackage(
        document=DocumentEvidence(
            filename="blank.pdf",
            page_count=1,
            extracted_at=datetime(2026, 8, 12, tzinfo=UTC),
        )
    )
    agent, client = make_agent(response_payload())

    result = agent.extract(empty)

    assert client.calls == []
    assert result.called_llm is False
    assert result.items == []
    assert result.errors[0].type is IssueType.NO_BOQ_DETECTED


def test_evidence_with_only_a_table_still_calls_the_llm():
    package = EvidencePackage(
        document=DocumentEvidence(
            filename="x.pdf",
            page_count=1,
            extracted_at=datetime(2026, 8, 12, tzinfo=UTC),
        ),
        tables=[
            TableEvidence(
                sequence=0,
                ref="#/tables/0",
                table_index=0,
                page_number=1,
                num_rows=1,
                num_cols=4,
                rows=[TableRowEvidence(row_index=0, cells=["1.1", "Excavation", "m3", "5"])],
            )
        ],
    )
    agent, client = make_agent(response_payload())

    agent.extract(package)

    assert len(client.calls) == 1


# --- prompts ----------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Never invent a value",
        "unresolved",
        "level_path",
        "CARRIED FORWARD",
        "fixed_rate",
        "per_unit",
        "percent",
    ],
)
def test_system_prompt_states_the_rules(phrase):
    assert phrase in SYSTEM_PROMPT


def test_user_prompt_renders_evidence_in_reading_order(evidence):
    prompt = build_user_prompt(evidence)

    assert "--- PAGE 1 ---" in prompt
    assert "HEADER | Item | Description | Unit | Quantity" in prompt
    assert "row 2 | 1.1 | Excavation for foundations | m3 | 125.50" in prompt
    assert prompt.index("BILL OF QUANTITIES") < prompt.index("row 2 |")


def test_user_prompt_is_not_a_json_dump(evidence):
    prompt = build_user_prompt(evidence)

    assert '"self_ref"' not in prompt
    assert "bbox" not in prompt
    assert len(prompt) < len(evidence.model_dump_json())


def test_repair_prompt_forbids_inventing_values():
    prompt = build_repair_prompt("{bad", "items: invalid")

    assert "Do not invent values to satisfy the schema" in prompt
    assert "unresolved" in prompt


# --- response schema --------------------------------------------------


def test_response_schema_matches_the_boq_item_model():
    schema_fields = set(BOQ_RESPONSE_SCHEMA["properties"]["items"]["items"]["properties"])

    assert schema_fields == set(BOQItem.model_fields)


def test_response_schema_is_strict_mode_compatible():
    def check(node: dict) -> None:
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        elif node.get("type") == "array":
            check(node["items"])

    check(BOQ_RESPONSE_SCHEMA)


def test_response_schema_has_no_document_or_total_fields():
    item_fields = BOQ_RESPONSE_SCHEMA["properties"]["items"]["items"]["properties"]

    for forbidden in ("amount", "total", "id", "temp_id", "extracted_at", "filename"):
        assert forbidden not in item_fields
    assert set(BOQ_RESPONSE_SCHEMA["properties"]) == {"items", "unresolved"}


# --- Groq client ------------------------------------------------------


class FakeCompletions:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if isinstance(self.outcome, list):
            outcome = self.outcome[min(len(self.kwargs) - 1, len(self.outcome) - 1)]
        else:
            outcome = self.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeGroq:
    def __init__(self, outcome) -> None:
        self.completions = FakeCompletions(outcome)
        self.chat = type("Chat", (), {"completions": self.completions})()


def completion(content: str = '{"items": [], "unresolved": []}', finish_reason: str = "stop"):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 20})()
    return type(
        "Completion",
        (),
        {"choices": [choice], "model": "openai/gpt-oss-120b", "usage": usage},
    )()


def api_error(cls, status_code: int, message: str):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls(message, response=response, body=None)


def test_missing_api_key_is_a_configuration_error():
    client = GroqClient(settings=Settings(groq_api_key=None))

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        client.complete_json([{"role": "user", "content": "hi"}])


def test_client_requests_strict_json_schema():
    fake = FakeGroq(completion())
    client = GroqClient(settings=SETTINGS, client=fake)

    response = client.complete_json(
        [{"role": "user", "content": "hi"}],
        json_schema=BOQ_RESPONSE_SCHEMA,
        schema_name="boq_extraction",
    )

    sent = fake.completions.kwargs[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["name"] == "boq_extraction"
    assert response.used_json_schema is True
    assert response.prompt_tokens == 10


def test_client_falls_back_to_json_mode_when_schema_is_rejected():
    rejection = api_error(
        groq.BadRequestError, 400, "response_format json_schema is not supported"
    )
    fake = FakeGroq([rejection, completion()])
    client = GroqClient(settings=SETTINGS, client=fake)

    response = client.complete_json(
        [{"role": "user", "content": "hi"}], json_schema=BOQ_RESPONSE_SCHEMA
    )

    assert response.used_json_schema is False
    assert fake.completions.kwargs[1]["response_format"] == {"type": "json_object"}


def test_unrelated_bad_request_is_not_retried():
    rejection = api_error(groq.BadRequestError, 400, "model does not exist")
    fake = FakeGroq(rejection)
    client = GroqClient(settings=SETTINGS, client=fake)

    with pytest.raises(AIRequestError, match="rejected the request"):
        client.complete_json([{"role": "user", "content": "hi"}], json_schema=BOQ_RESPONSE_SCHEMA)

    assert len(fake.completions.kwargs) == 1


def test_rate_limit_becomes_an_ai_request_error():
    fake = FakeGroq(api_error(groq.RateLimitError, 429, "rate limit exceeded"))
    client = GroqClient(settings=SETTINGS, client=fake)

    with pytest.raises(AIRequestError, match="429"):
        client.complete_json([{"role": "user", "content": "hi"}])


def test_timeout_becomes_an_ai_request_error():
    request = httpx.Request("POST", "https://api.groq.com/")
    fake = FakeGroq(groq.APITimeoutError(request=request))
    client = GroqClient(settings=SETTINGS, client=fake)

    with pytest.raises(AIRequestError, match="Could not reach"):
        client.complete_json([{"role": "user", "content": "hi"}])


def test_truncated_response_is_an_error():
    fake = FakeGroq(completion(finish_reason="length"))
    client = GroqClient(settings=SETTINGS, client=fake)

    with pytest.raises(AIRequestError, match="cut off"):
        client.complete_json([{"role": "user", "content": "hi"}])


def test_empty_response_is_an_error():
    fake = FakeGroq(completion(content="   "))
    client = GroqClient(settings=SETTINGS, client=fake)

    with pytest.raises(AIRequestError, match="empty response"):
        client.complete_json([{"role": "user", "content": "hi"}])


def test_api_key_is_never_logged(caplog):
    fake = FakeGroq(completion())
    client = GroqClient(settings=Settings(groq_api_key="gsk_supersecret"), client=fake)

    with caplog.at_level("DEBUG"):
        client.complete_json([{"role": "user", "content": "hi"}])

    assert "gsk_supersecret" not in caplog.text
