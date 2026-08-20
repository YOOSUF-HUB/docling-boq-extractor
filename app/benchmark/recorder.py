"""A Groq client that measures the request instead of altering it.

`BOQAgent(client=...)` already accepts an injected client, so a live benchmark
needs no change to the agent: this wrapper delegates every call to a real
`GroqClient` and keeps a copy of what went over the wire.

Recording the messages is what makes a live run *verifiable*. The benchmark
builds its own copy of the prompt from `app.agent.prompts`; the recorder shows
what the agent actually sent, and `token_benchmark` compares the two. If they
ever diverge, the benchmark says so instead of reporting numbers for a request
that was never made.

Repair rounds are recorded as separate attempts, because the second call resends
the whole conversation plus the rejected output — the most expensive request the
pipeline can make, and invisible in a single prompt-token figure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.agent.groq_client import GroqClient, LLMResponse


@dataclass(frozen=True)
class RecordedCall:
    """One `complete_json` call: what was sent, and what came back."""

    messages: list[dict[str, str]]
    used_json_schema_request: bool
    response: LLMResponse | None = None
    error: str | None = None

    @property
    def sent_characters(self) -> int:
        return sum(len(message.get("content", "")) for message in self.messages)

    @property
    def sent_text(self) -> str:
        """The message contents alone, with no chat-format markup around them."""
        return "".join(message.get("content", "") for message in self.messages)


@dataclass
class RecordingGroqClient:
    """Wraps a `GroqClient`, recording each request and response."""

    inner: GroqClient
    calls: list[RecordedCall] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def settings(self) -> Any:
        return self.inner.settings

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMResponse:
        sent = copy.deepcopy(messages)
        try:
            response = self.inner.complete_json(
                messages, json_schema=json_schema, schema_name=schema_name
            )
        except Exception as exc:
            # Recorded, then re-raised untouched: a 413 for an oversized request
            # is a benchmark result, not something to swallow.
            self.calls.append(
                RecordedCall(
                    messages=sent,
                    used_json_schema_request=json_schema is not None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

        self.calls.append(
            RecordedCall(
                messages=sent,
                used_json_schema_request=json_schema is not None,
                response=response,
            )
        )
        return response

    @property
    def first_call(self) -> RecordedCall | None:
        return self.calls[0] if self.calls else None
