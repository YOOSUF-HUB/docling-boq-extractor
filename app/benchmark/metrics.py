"""The shape of one token-usage measurement.

Every model here is `extra="forbid"` for the same reason the BOQ schema is: a
benchmark record that quietly grows fields is a benchmark you cannot compare
across runs.

Two conventions carry through the whole record:

* **Counted vs. reported.** A `*_tokens` value produced by our tokenizer is a
  count of a string we hold. A value from Groq's `usage` is what the API
  charged. They are kept in separate fields and reconciled, never merged.
* **Lower bounds are labelled.** Without a live call the chat envelope Groq
  wraps around the messages is unknown, so `input_tokens` is marked
  `input_is_lower_bound=True` rather than being presented as the real figure.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.benchmark.tokenizer import TextMeasurement, TokenizerInfo

#: Bump when the record's meaning changes, so old results stay comparable.
BENCHMARK_VERSION = "1.0"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizerReport(_Record):
    """Which tokenizer produced the counts in this record."""

    encoding: str
    library: str
    library_version: str
    target_model: str
    is_native: bool
    note: str

    @classmethod
    def of(cls, info: TokenizerInfo) -> TokenizerReport:
        return cls(
            encoding=info.encoding,
            library=info.library,
            library_version=info.library_version,
            target_model=info.target_model,
            is_native=info.is_native,
            note=info.note,
        )


class Measurement(_Record):
    """One measured string: its size in characters and in tokens."""

    characters: int = Field(ge=0)
    tokens: int = Field(ge=0)
    chars_per_token: float = Field(ge=0)

    @classmethod
    def of(cls, measurement: TextMeasurement) -> Measurement:
        return cls(
            characters=measurement.characters,
            tokens=measurement.tokens,
            chars_per_token=round(measurement.chars_per_token, 3),
        )


class DocumentMetrics(_Record):
    """What Docling found in the source document."""

    filename: str
    file_size_bytes: int | None = None
    page_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    text_block_count: int = Field(ge=0)
    table_row_count: int = Field(ge=0)
    picture_count: int = Field(ge=0)
    docling_duration_seconds: float | None = None


class EvidenceMetrics(_Record):
    """How large the evidence is, in each form it exists in.

    The three forms are the funnel the pipeline already implements: Docling's
    own JSON, the evidence package derived from it, and the text actually sent
    to the model.
    """

    docling_document_json: Measurement | None = None
    serialized_compact: Measurement
    serialized_indented: Measurement
    rendered_prompt: Measurement
    #: How much the prompt renderer saves against sending the evidence as JSON.
    rendering_saving_percent: float


class PromptMetrics(_Record):
    """The request's input side, component by component."""

    system_prompt: Measurement
    user_prompt: Measurement
    #: The strict response schema, which Groq serializes into the request.
    response_schema: Measurement

    #: system + user + schema, all counted from strings we hold.
    counted_input_tokens: int = Field(ge=0)
    #: Chat-format overhead. Known only from a live call, or supplied from one.
    envelope_tokens: int | None = None
    envelope_source: str = "not measured"
    #: counted + envelope. A lower bound while the envelope is unknown.
    input_tokens: int = Field(ge=0)
    input_is_lower_bound: bool = True


class PromptComponent(_Record):
    """One labelled piece of the user prompt and what it costs."""

    kind: str
    label: str
    sequence: int | None = None
    page_number: int | None = None
    row_count: int | None = None
    characters: int = Field(ge=0)
    tokens: int = Field(ge=0)
    share_of_user_prompt_percent: float
    tokens_per_row: float | None = None
    scales_with_document: bool


class CostCentre(_Record):
    """Tokens grouped by what would have to change to reduce them."""

    name: str
    tokens: int = Field(ge=0)
    share_percent: float
    scales_with_document: bool
    note: str = ""


class BudgetMetrics(_Record):
    """What the request reserves against the rate limit.

    Groq counts `prompt_tokens + max_completion_tokens` towards tokens-per-
    minute *before* the call runs, so the completion budget is spent whether or
    not the model uses it. That is why `requested_budget_tokens`, not
    `input_tokens`, is what a TPM limit rejects.
    """

    max_completion_tokens: int = Field(gt=0)
    tpm_limit: int = Field(gt=0)
    requested_budget_tokens: int = Field(ge=0)
    headroom_tokens: int
    utilisation_percent: float
    exceeds_limit: bool
    budget_is_lower_bound: bool


class AttemptMetrics(_Record):
    """One HTTP call to the model. A repair round is a second attempt."""

    index: int = Field(ge=1)
    message_count: int = Field(ge=0)
    sent_characters: int = Field(ge=0)
    sent_tokens: int = Field(ge=0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    duration_seconds: float = 0.0
    used_json_schema: bool = False


class CompletionMetrics(_Record):
    """What a live run actually cost and produced."""

    model: str
    attempts: int = Field(ge=0)
    finish_reason: str | None = None
    #: The first attempt — the baseline request this benchmark describes.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Every attempt summed, including repair rounds.
    total_tokens_all_attempts: int | None = None
    llm_duration_seconds: float = 0.0
    items_extracted: int = Field(default=0, ge=0)
    unresolved_items: int = Field(default=0, ge=0)
    validation_status: str | None = None
    per_attempt: list[AttemptMetrics] = Field(default_factory=list)
    #: Set when the call failed outright — a 413 for an oversized request is
    #: itself a measurement, so it is recorded rather than raised away.
    error: str | None = None
    #: False when the recorded request differed from the prompt we measured.
    request_matches_measurement: bool = True

    #: Counted input vs. what Groq charged, for the first attempt.
    estimate_error_tokens: int | None = None
    estimate_error_percent: float | None = None


class TokenBenchmark(_Record):
    """One document measured end to end."""

    benchmark_version: str = BENCHMARK_VERSION
    measured_at: datetime
    source: str
    mode: str
    model: str
    tokenizer: TokenizerReport
    document: DocumentMetrics
    evidence: EvidenceMetrics
    prompt: PromptMetrics
    components: list[PromptComponent] = Field(default_factory=list)
    cost_centres: list[CostCentre] = Field(default_factory=list)
    budget: BudgetMetrics
    completion: CompletionMetrics | None = None

    @property
    def is_live(self) -> bool:
        return self.completion is not None

    def heaviest_components(self, limit: int = 10) -> list[PromptComponent]:
        return sorted(self.components, key=lambda part: part.tokens, reverse=True)[:limit]
