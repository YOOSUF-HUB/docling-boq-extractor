"""Groq API client.

A thin, explicit wrapper. Its whole job is: send two messages, ask for
structured JSON, hand back the text and token usage — and turn Groq's exception
hierarchy into this application's error taxonomy so callers can distinguish an
AI failure from a document or validation failure.

The API key is read from configuration and never logged. Requests are logged by
model name and prompt size only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import groq

from app.config import Settings, get_settings
from app.errors import AIRequestError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """One completion."""

    content: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_seconds: float
    used_json_schema: bool


class GroqClient:
    """Structured-output client for Groq chat completions."""

    def __init__(self, settings: Settings | None = None, client: groq.Groq | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def model(self) -> str:
        return self.settings.groq_model

    @property
    def client(self) -> groq.Groq:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise ConfigurationError(
                    "GROQ_API_KEY is not set. Add it to .env (see .env.example)."
                )
            self._client = groq.Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.groq_timeout_seconds,
                max_retries=self.settings.groq_max_retries,
            )
        return self._client

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMResponse:
        """Request a JSON response, preferring strict schema enforcement.

        If the model rejects `json_schema` response format, this falls back once
        to plain JSON mode. The fallback is logged: it is a real reduction in
        guarantees, not a silent equivalent.
        """
        prompt_chars = sum(len(message["content"]) for message in messages)
        logger.info(
            "LLM request started (model=%s, messages=%d, prompt_chars=%d)",
            self.model,
            len(messages),
            prompt_chars,
        )

        use_schema = json_schema is not None
        started = time.perf_counter()
        try:
            completion = self._create(messages, json_schema if use_schema else None, schema_name)
        except groq.BadRequestError as exc:
            if not use_schema or not _is_response_format_rejection(exc):
                raise AIRequestError(f"Groq rejected the request: {exc}") from exc
            logger.warning(
                "Model %s rejected json_schema response format; "
                "falling back to JSON mode without schema enforcement.",
                self.model,
            )
            use_schema = False
            completion = self._create(messages, None, schema_name)
        except groq.APIStatusError as exc:
            raise AIRequestError(f"Groq API error ({exc.status_code}): {exc}") from exc
        except (groq.APITimeoutError, groq.APIConnectionError) as exc:
            raise AIRequestError(f"Could not reach the Groq API: {exc}") from exc

        duration = time.perf_counter() - started
        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = getattr(completion, "usage", None)

        logger.info(
            "LLM response received in %.2fs (finish_reason=%s, completion_tokens=%s)",
            duration,
            choice.finish_reason,
            getattr(usage, "completion_tokens", None),
        )

        if choice.finish_reason == "length":
            raise AIRequestError(
                "The model's response was cut off by the token limit. "
                "Increase GROQ_MAX_COMPLETION_TOKENS or split the document."
            )
        if not content.strip():
            raise AIRequestError("The model returned an empty response.")

        return LLMResponse(
            content=content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            duration_seconds=duration,
            used_json_schema=use_schema,
        )

    def _create(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> Any:
        if json_schema is not None:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            }
        else:
            response_format = {"type": "json_object"}

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            temperature=self.settings.groq_temperature,
            max_completion_tokens=self.settings.groq_max_completion_tokens,
        )


def _is_response_format_rejection(error: groq.BadRequestError) -> bool:
    """True when a 400 is about the structured-output format specifically."""
    text = str(error).lower()
    return "response_format" in text or "json_schema" in text or "json schema" in text
