"""Canonical BOQ schema — the document-derived subset of the PMS BOQ contract.

This is a *target data format*, not a runtime dependency on PMS. The standalone
agent emits only what a document can actually support:

    DOCUMENT-DERIVED   level_path, boq_item_code, boq_item_name, boq_description,
                       quantity, unit, cost_type, the eight cost fields,
                       profit, discount, fixed_rate

    NOT EMITTED        derived values (totals, tree depth, parents) and anything
                       PMS/backend-generated (database ids, temp ids,
                       project-prefixed codes, persistence metadata)

`extra="forbid"` is what enforces that second line: if an LLM response contains
`id`, `temp_id`, `boq_id` or any other invented field, parsing fails loudly
instead of quietly carrying a PMS-internal field into the output.

Five fields must never be guessed — `level_path`, `boq_item_code`,
`boq_item_name`, `quantity`, `unit`. They have no defaults here, so an omitted
or empty value is a hard schema failure rather than a silent zero.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

BOQ_SOURCE_VERSION = "1.0"

#: Money-ish and quantity fields: finite, non-negative.
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class CostType(StrEnum):
    """How an item's cost is expressed.

    Only `per_unit` exists in the document-derived contract; other PMS cost
    types are not inferable from a PDF and are deliberately not invented here.
    """

    PER_UNIT = "per_unit"


class ValueMode(StrEnum):
    """Whether a profit/discount value is an absolute amount or a percentage.

    The two must never be collapsed: `10` as an amount and `10%` mean very
    different things, and only the document can say which one it is.
    """

    AMOUNT = "amount"
    PERCENT = "percent"


class ValueSpec(BaseModel):
    """A profit or discount entry: `{"mode": "percent", "value": 10.0}`."""

    model_config = ConfigDict(extra="forbid")

    mode: ValueMode = ValueMode.AMOUNT
    value: Amount = 0.0


def _require_text(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


class BOQItem(BaseModel):
    """One priced BOQ line.

    Section headings are *not* items — they appear in `level_path`. Totals,
    subtotals and carried/brought-forward rows are not items either; the
    deterministic validator rejects them if they slip through.
    """

    model_config = ConfigDict(extra="forbid")

    # --- required: never guessed -------------------------------------
    level_path: list[str] = Field(min_length=1)
    boq_item_code: str
    boq_item_name: str
    quantity: Amount
    unit: str

    # --- optional document-derived -----------------------------------
    boq_description: str = ""
    cost_type: CostType = CostType.PER_UNIT

    labour: Amount = 0.0
    machine: Amount = 0.0
    material: Amount = 0.0
    fuel: Amount = 0.0
    miscellaneous: Amount = 0.0
    subcontract: Amount = 0.0
    site_overhead: Amount = 0.0
    head_office_overhead: Amount = 0.0

    profit: ValueSpec = Field(default_factory=ValueSpec)
    discount: ValueSpec = Field(default_factory=ValueSpec)

    #: Used when the document gives a direct rate instead of a cost breakdown.
    fixed_rate: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    COST_FIELDS: ClassVar[tuple[str, ...]] = (
        "labour",
        "machine",
        "material",
        "fuel",
        "miscellaneous",
        "subcontract",
        "site_overhead",
        "head_office_overhead",
    )

    @field_validator("level_path", mode="after")
    @classmethod
    def _validate_level_path(cls, value: list[str]) -> list[str]:
        cleaned = [segment.strip() for segment in value]
        if any(not segment for segment in cleaned):
            raise ValueError("level_path must not contain empty segments")
        return cleaned

    @field_validator("boq_item_code", "boq_item_name", "unit", mode="after")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _require_text(value, info.field_name)

    @field_validator("boq_description", mode="after")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        return value.strip()

    @property
    def cost_breakdown(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.COST_FIELDS}

    @property
    def has_cost_breakdown(self) -> bool:
        """True when the document supplied at least one real cost component."""
        return any(value > 0 for value in self.cost_breakdown.values())

    @property
    def section(self) -> str:
        """Top-level section this item belongs to."""
        return self.level_path[0]


class BOQDocumentInfo(BaseModel):
    """Source document metadata. No database ids, no persistence metadata."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    extracted_at: datetime

    @field_validator("filename", mode="after")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        return _require_text(value, "filename")

    @field_validator("extracted_at", mode="after")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """The contract says ISO-8601 UTC, so normalize to it.

        A naive timestamp is ambiguous rather than wrong — it is read as UTC
        rather than rejected, since it is generated by this application.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class BOQDocument(BaseModel):
    """The canonical output document."""

    model_config = ConfigDict(extra="forbid")

    boq_source_version: str = BOQ_SOURCE_VERSION
    document: BOQDocumentInfo
    items: list[BOQItem] = Field(default_factory=list)

    @classmethod
    def assemble(
        cls,
        items: list[BOQItem],
        *,
        filename: str,
        extracted_at: datetime | None = None,
    ) -> BOQDocument:
        """Build the canonical document around LLM-extracted items.

        The filename and timestamp are deterministic Python values — the model
        is never asked to produce them.
        """
        return cls(
            document=BOQDocumentInfo(
                filename=filename,
                extracted_at=extracted_at or datetime.now(UTC),
            ),
            items=items,
        )

    @property
    def item_codes(self) -> list[str]:
        return [item.boq_item_code for item in self.items]

    @property
    def sections(self) -> list[str]:
        """Distinct top-level sections, in first-seen order."""
        return list(dict.fromkeys(item.section for item in self.items))

    @property
    def level_paths(self) -> list[tuple[str, ...]]:
        """Distinct hierarchy paths, in first-seen order."""
        return list(dict.fromkeys(tuple(item.level_path) for item in self.items))


class UnresolvedItem(BaseModel):
    """A row the model believes is a BOQ item but could not fully extract.

    This is the alternative to guessing. Rather than inventing a unit or a
    quantity, the model reports what it saw and which required fields it could
    not establish; the extraction report surfaces it as a warning.
    """

    model_config = ConfigDict(extra="forbid")

    boq_item_code: str | None = None
    description: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    reason: str
    evidence_ref: str | None = None


class BOQExtractionResponse(BaseModel):
    """Exactly what the LLM is allowed to return.

    Note what is *absent*: no filename, no timestamp, no totals, no ids. Those
    are either deterministic Python values or computed downstream, so the model
    is never given the opportunity to invent them.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[BOQItem] = Field(default_factory=list)
    unresolved: list[UnresolvedItem] = Field(default_factory=list)
