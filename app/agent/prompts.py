"""What the model is told, and what it is allowed to say back.

Three things live here:

* `SYSTEM_PROMPT` — the extraction rules.
* `BOQ_RESPONSE_SCHEMA` — the strict JSON schema sent as `response_format`, so
  the shape of the answer is enforced by the API rather than by pleading.
* `build_user_prompt` — renders the evidence package as compact, ordered text.

The evidence is *not* dumped as raw JSON. It is rendered as a line per element
in reading order, with page markers, table indices and row indices preserved,
so the model can cite what it used and we stay within a sane token budget.
"""

from __future__ import annotations

from typing import Any

from app.schemas.evidence import EvidencePackage, TableEvidence, TextBlockEvidence

# --- system prompt ----------------------------------------------------

SYSTEM_PROMPT = """\
You are a Bill of Quantities (BOQ) extraction agent.

You convert document evidence extracted from a PDF by Docling into a canonical, \
structured BOQ. You are the semantic step only: the document has already been \
parsed for you, and everything you output must be traceable to the evidence you \
are given.

# Absolute rules

1. Extract ONLY what the evidence supports. Never invent a value.
2. Never infer a missing quantity, unit, item code or item name. If you cannot \
establish one of the required fields for a row, do NOT emit an item for it — \
report it in `unresolved` instead.
3. Preserve numbers exactly as the document states them. Do not round, convert \
units, recalculate, or normalise 125.50 into anything other than 125.5.
4. Preserve the document's own item code. Never renumber, never add a project \
prefix, never generate a code for a row that has none.
5. Never emit database ids, temporary ids, project-prefixed codes, timestamps, \
filenames, totals, or any field not defined in the response schema.
6. Never compute amounts. Quantity x rate is a downstream calculation, not an \
extraction. There is no amount or total field in the output.

# Sections versus items

A section heading is a hierarchy node, NOT an item.

    2 CONCRETE WORKS                        -> a section
    2.1 Blinding concrete   m3   15.00      -> an item

The item belongs to that section:

    "level_path": ["CONCRETE WORKS"]
    "boq_item_code": "2.1"

Nested sections nest the path, deepest last:

    1   EARTHWORKS
    1.1 BULK EXCAVATION
    1.1.1 Excavation in rock   m3   40.00

    "level_path": ["EARTHWORKS", "BULK EXCAVATION"]
    "boq_item_code": "1.1.1"

Use the section's NAME in `level_path`, never its numbering. `["EARTHWORKS"]`, \
not `["1"]`. A row with an item code but no quantity and no unit is almost \
always a section heading, not an item.

# Rows that are never items

Ignore rows that summarise other rows:

    TOTAL, SUB-TOTAL, GRAND TOTAL, PAGE TOTAL, COLLECTION, SUMMARY,
    CARRIED FORWARD, BROUGHT FORWARD, C/F, B/F

They are not items and must not appear in `items` or in `level_path`.

# Rates and costs

* If the document gives a direct rate per unit and no cost breakdown, put it in \
`fixed_rate` and leave all eight cost fields at 0.
* If the document gives a cost breakdown (labour, machine, material, fuel, \
miscellaneous, subcontract, site overhead, head office overhead), put each real \
value in its own field and leave `fixed_rate` null.
* Never populate both from the same row.
* If the document shows neither, leave `fixed_rate` null and all cost fields 0. \
That is a normal quantity-only BOQ, not an error.
* An `Amount` or `Total` column is verification evidence only. Never output it.

# Profit and discount

Percentages and amounts are different things and must not be collapsed:

    10%   -> {"mode": "percent", "value": 10.0}
    5000  -> {"mode": "amount",  "value": 5000.0}

If the document says nothing, use {"mode": "amount", "value": 0.0}.

# Descriptions

`boq_item_name` is the row's own description. Put supplementary text (specs, \
notes attached to the row) in `boq_description`, or leave it as "".

# When you cannot extract a row

Add an entry to `unresolved` with the row's code or description, which required \
fields you could not establish, why, and the evidence reference (for example \
"table#0 row 7"). Reporting a problem is always correct; guessing never is.

# Output

Return only JSON matching the provided schema. `cost_type` is always \
"per_unit". Emit every field for every item, using 0 or null where the document \
provides nothing.\
"""


# --- response schema (strict structured output) -----------------------

_VALUE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["amount", "percent"]},
        "value": {"type": "number"},
    },
    "required": ["mode", "value"],
    "additionalProperties": False,
}

_COST_FIELDS = (
    "labour",
    "machine",
    "material",
    "fuel",
    "miscellaneous",
    "subcontract",
    "site_overhead",
    "head_office_overhead",
)

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "level_path": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Section names from outermost to innermost. Never numbering.",
        },
        "boq_item_code": {"type": "string", "description": "The document's own item code."},
        "boq_item_name": {"type": "string"},
        "boq_description": {"type": "string"},
        "quantity": {"type": "number"},
        "unit": {"type": "string"},
        "cost_type": {"type": "string", "enum": ["per_unit"]},
        **{name: {"type": "number"} for name in _COST_FIELDS},
        "profit": _VALUE_SPEC_SCHEMA,
        "discount": _VALUE_SPEC_SCHEMA,
        "fixed_rate": {
            "type": ["number", "null"],
            "description": "Direct rate when the document gives no cost breakdown.",
        },
    },
    "additionalProperties": False,
}
# Strict structured output requires every property to be listed as required.
_ITEM_SCHEMA["required"] = list(_ITEM_SCHEMA["properties"])

_UNRESOLVED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "boq_item_code": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "missing_fields": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "level_path",
                    "boq_item_code",
                    "boq_item_name",
                    "quantity",
                    "unit",
                ],
            },
        },
        "reason": {"type": "string"},
        "evidence_ref": {
            "type": ["string", "null"],
            "description": "Where the row was seen, e.g. 'table#0 row 7'.",
        },
    },
    "required": ["boq_item_code", "description", "missing_fields", "reason", "evidence_ref"],
    "additionalProperties": False,
}

BOQ_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": _ITEM_SCHEMA},
        "unresolved": {"type": "array", "items": _UNRESOLVED_SCHEMA},
    },
    "required": ["items", "unresolved"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA_NAME = "boq_extraction"


# --- user prompt ------------------------------------------------------

#: Sentinel so the first element always emits a page marker, even for page None.
_NO_PAGE_YET = object()


def build_user_prompt(evidence: EvidencePackage) -> str:
    """Render the evidence package as ordered, compact text."""
    document = evidence.document
    lines = [
        "DOCUMENT EVIDENCE",
        f"filename: {document.filename}",
        f"pages: {document.page_count}",
        f"tables: {evidence.statistics.tables}",
        "",
        "Elements are listed in document reading order. `seq` is the reading",
        "position; use `table#N row M` when referring to evidence.",
        "",
    ]

    current_page: Any = _NO_PAGE_YET
    for element in evidence.elements_in_document_order():
        if element.page_number != current_page:
            current_page = element.page_number
            page_label = current_page if current_page is not None else "unknown"
            lines.append(f"--- PAGE {page_label} ---")

        if isinstance(element, TableEvidence):
            lines.extend(_render_table(element))
        else:
            lines.append(_render_text_block(element))

    lines.extend(["", "Extract the BOQ from this evidence."])
    return "\n".join(lines)


def _render_text_block(block: TextBlockEvidence) -> str:
    layer = "" if block.content_layer == "body" else f" ({block.content_layer})"
    return f"[seq {block.sequence}] {block.label.upper()}{layer}: {block.text}"


def _render_table(table: TableEvidence) -> list[str]:
    header = (
        f"[seq {table.sequence}] TABLE table#{table.table_index} "
        f"({table.num_rows} rows x {table.num_cols} cols)"
    )
    lines = [header]
    if table.caption:
        lines.append(f"  caption: {table.caption}")

    for row in table.rows:
        marker = "HEADER" if row.is_header else f"row {row.row_index}"
        lines.append(f"  {marker} | " + " | ".join(row.cells))
    return lines


def build_repair_prompt(invalid_output: str, errors: str) -> str:
    """Ask for a corrected answer after schema validation failed.

    The model is asked to re-extract, not to patch: we never repair malformed
    output ourselves, and we do not want it inventing values to satisfy the
    schema either.
    """
    return (
        "Your previous response did not satisfy the required schema.\n\n"
        f"Validation errors:\n{errors}\n\n"
        "Previous response:\n"
        f"{invalid_output}\n\n"
        "Return a corrected response for the SAME evidence, matching the schema "
        "exactly. Do not invent values to satisfy the schema: if a required "
        "field genuinely cannot be extracted, omit that item and report it in "
        "`unresolved` instead."
    )
