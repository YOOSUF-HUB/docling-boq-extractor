"""The BOQ extraction agent.

    EvidencePackage -> prompt -> Groq (gpt-oss-120b) -> JSON -> Pydantic

The agent owns exactly one decision the LLM cannot be trusted with: whether the
answer is acceptable. Output is parsed and validated against the canonical
schema; if it fails, the agent asks the model to try again with the validation
errors quoted, and gives up after `llm_max_attempts`.

Nothing is ever repaired locally. A malformed response is either re-requested
or reported — never patched into shape, because patching is indistinguishable
from inventing data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.agent.groq_client import GroqClient, LLMResponse
from app.agent.prompts import (
    BOQ_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_NAME,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_user_prompt,
)
from app.config import Settings, get_settings
from app.errors import AIResponseError
from app.schemas.boq import BOQExtractionResponse, BOQItem, UnresolvedItem
from app.schemas.evidence import EvidencePackage
from app.schemas.report import ExtractionIssue, IssueType
from app.validation.boq_validator import issues_from_validation_error

logger = logging.getLogger(__name__)

#: Which issue code a self-reported missing field maps to.
_MISSING_FIELD_ISSUES = {
    "level_path": IssueType.MISSING_LEVEL_PATH,
    "boq_item_code": IssueType.MISSING_ITEM_CODE,
    "boq_item_name": IssueType.MISSING_ITEM_NAME,
    "quantity": IssueType.MISSING_QUANTITY,
    "unit": IssueType.MISSING_UNIT,
}


@dataclass
class AgentResult:
    """What the agent produced, plus how it got there."""

    items: list[BOQItem] = field(default_factory=list)
    unresolved: list[UnresolvedItem] = field(default_factory=list)
    warnings: list[ExtractionIssue] = field(default_factory=list)
    errors: list[ExtractionIssue] = field(default_factory=list)
    model: str = ""
    attempts: int = 0
    duration_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    used_json_schema: bool = False
    raw_response: str = ""

    @property
    def called_llm(self) -> bool:
        return self.attempts > 0


class BOQAgent:
    """Turns document evidence into validated canonical BOQ items."""

    def __init__(
        self,
        client: GroqClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or GroqClient(settings=self.settings)

    def extract(self, evidence: EvidencePackage) -> AgentResult:
        """Extract BOQ items from an evidence package.

        Raises `AIResponseError` if the model never returns schema-valid output,
        and `AIRequestError` if the API cannot be reached. Returns a result
        carrying a `NO_BOQ_DETECTED` error — without calling the LLM — when the
        evidence contains nothing to extract from.
        """
        empty_result = self._reject_empty_evidence(evidence)
        if empty_result is not None:
            return empty_result

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(evidence)},
        ]

        started = time.perf_counter()
        response: LLMResponse | None = None
        failures: list[ExtractionIssue] = []

        for attempt in range(1, self.settings.llm_max_attempts + 1):
            response = self.client.complete_json(
                messages,
                json_schema=BOQ_RESPONSE_SCHEMA,
                schema_name=RESPONSE_SCHEMA_NAME,
            )

            parsed, issues = _parse_response(response.content)
            if parsed is not None:
                logger.info(
                    "LLM output validated on attempt %d: %d items, %d unresolved",
                    attempt,
                    len(parsed.items),
                    len(parsed.unresolved),
                )
                return self._build_result(
                    parsed,
                    response=response,
                    attempts=attempt,
                    duration=time.perf_counter() - started,
                )

            failures = issues
            logger.warning(
                "LLM output rejected on attempt %d/%d: %s",
                attempt,
                self.settings.llm_max_attempts,
                "; ".join(issue.message for issue in issues[:3]),
            )
            if attempt < self.settings.llm_max_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": build_repair_prompt(
                            response.content,
                            "\n".join(issue.message for issue in issues),
                        ),
                    },
                ]

        raise AIResponseError(
            f"The model did not return schema-valid BOQ output after "
            f"{self.settings.llm_max_attempts} attempts.",
            issues=failures,
        )

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _reject_empty_evidence(evidence: EvidencePackage) -> AgentResult | None:
        """Skip the LLM when there is demonstrably nothing to extract."""
        if evidence.tables or evidence.text_blocks:
            return None

        logger.warning("Evidence contains no tables and no text; skipping the LLM call.")
        return AgentResult(
            errors=[
                ExtractionIssue(
                    type=IssueType.NO_BOQ_DETECTED,
                    message=(
                        f"No text or tables were extracted from "
                        f"{evidence.document.filename}; there is no BOQ to interpret."
                    ),
                )
            ],
        )

    def _build_result(
        self,
        parsed: BOQExtractionResponse,
        *,
        response: LLMResponse,
        attempts: int,
        duration: float,
    ) -> AgentResult:
        return AgentResult(
            items=parsed.items,
            unresolved=parsed.unresolved,
            warnings=issues_from_unresolved(parsed.unresolved),
            model=response.model,
            attempts=attempts,
            duration_seconds=duration,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            used_json_schema=response.used_json_schema,
            raw_response=response.content,
        )


def _parse_response(content: str) -> tuple[BOQExtractionResponse | None, list[ExtractionIssue]]:
    """Parse and validate one response. Never repairs."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [
            ExtractionIssue(
                type=IssueType.LLM_INVALID_OUTPUT,
                message=f"Response was not valid JSON: {exc}",
            )
        ]

    try:
        return BOQExtractionResponse.model_validate(payload), []
    except ValidationError as exc:
        # Reuse the canonical mapping so a missing unit reports MISSING_UNIT
        # rather than a generic schema failure.
        return None, issues_from_validation_error(exc)


def issues_from_unresolved(unresolved: list[UnresolvedItem]) -> list[ExtractionIssue]:
    """Turn self-reported unresolved rows into report warnings.

    One issue per missing required field, so "unit could not be identified for
    item 2.2" stays distinguishable from "the whole row was unreadable".
    """
    issues: list[ExtractionIssue] = []
    for entry in unresolved:
        label = entry.boq_item_code or entry.description or "an unidentified row"
        recognised = [name for name in entry.missing_fields if name in _MISSING_FIELD_ISSUES]

        if not recognised:
            issues.append(
                ExtractionIssue(
                    type=IssueType.UNRESOLVED_ROW,
                    message=f"Row '{label}' could not be extracted: {entry.reason}",
                    item_code=entry.boq_item_code,
                )
            )
            continue

        for name in recognised:
            issues.append(
                ExtractionIssue(
                    type=_MISSING_FIELD_ISSUES[name],
                    message=(
                        f"{name} could not be confidently identified for "
                        f"'{label}': {entry.reason}"
                    ),
                    item_code=entry.boq_item_code,
                    field=name,
                )
            )
    return issues
