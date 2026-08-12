"""Evidence package schema.

The evidence package is the contract between the *document* layer and the
*reasoning* layer. It is a deliberately small, explicit view of what Docling
found — not a copy of the Docling JSON, which is ~50x larger and full of
geometry the LLM cannot use.

Design decisions worth knowing:

* **Content lives in `text_blocks` and `tables`, not in `pages`.** `pages`
  carries page-level metadata only, so no piece of text is serialized twice.
* **`sequence` preserves document order** across both lists. A section heading
  that precedes a table is what tells the agent which section the table's rows
  belong to, so reading order is evidence in its own right.
* **`ref` preserves provenance** (the Docling `self_ref`, e.g. `#/tables/0`),
  and every table row keeps its original `row_index`. This is how a BOQ item
  can later be traced back to page/table/row in the extraction report.
* **Nothing is interpreted here.** No BOQ semantics, no item detection, no
  number parsing — cells stay strings exactly as the document had them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

EVIDENCE_VERSION = "1.0"


class DocumentEvidence(BaseModel):
    """Metadata about the source document."""

    filename: str
    page_count: int = Field(ge=0)
    extracted_at: datetime
    docling_status: str | None = None
    # The DoclingDocument *schema* version, not the docling package version.
    docling_schema_version: str | None = None


class PageEvidence(BaseModel):
    """Per-page metadata. Content is carried by text blocks and tables."""

    page_number: int
    width: float | None = None
    height: float | None = None
    text_block_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    picture_count: int = Field(default=0, ge=0)


class TextBlockEvidence(BaseModel):
    """One piece of non-table text, in document order."""

    sequence: int = Field(ge=0)
    ref: str
    page_number: int | None = None
    label: str
    content_layer: str
    text: str
    heading_level: int | None = None


class TableRowEvidence(BaseModel):
    """One table row. `row_index` is the row's position in the original grid."""

    row_index: int = Field(ge=0)
    is_header: bool = False
    cells: list[str]


class TableEvidence(BaseModel):
    """One detected table, with its grid flattened to rows of cell text."""

    sequence: int = Field(ge=0)
    ref: str
    table_index: int = Field(ge=0)
    page_number: int | None = None
    num_rows: int = Field(ge=0)
    num_cols: int = Field(ge=0)
    caption: str | None = None
    rows: list[TableRowEvidence] = Field(default_factory=list)

    @property
    def header_rows(self) -> list[TableRowEvidence]:
        return [row for row in self.rows if row.is_header]

    @property
    def data_rows(self) -> list[TableRowEvidence]:
        return [row for row in self.rows if not row.is_header]

    @property
    def column_headers(self) -> list[str]:
        """Cells of the first header row, or `[]` if none was detected."""
        headers = self.header_rows
        return headers[0].cells if headers else []


class EvidenceStatistics(BaseModel):
    """Counts used by the CLI summary and, later, the extraction report."""

    pages: int = Field(default=0, ge=0)
    text_blocks: int = Field(default=0, ge=0)
    tables: int = Field(default=0, ge=0)
    table_rows: int = Field(default=0, ge=0)
    pictures: int = Field(default=0, ge=0)


class EvidencePackage(BaseModel):
    """Everything the BOQ agent is allowed to reason from."""

    evidence_version: str = EVIDENCE_VERSION
    document: DocumentEvidence
    pages: list[PageEvidence] = Field(default_factory=list)
    text_blocks: list[TextBlockEvidence] = Field(default_factory=list)
    tables: list[TableEvidence] = Field(default_factory=list)
    statistics: EvidenceStatistics = Field(default_factory=EvidenceStatistics)

    @property
    def has_tables(self) -> bool:
        return bool(self.tables)

    def elements_in_document_order(self) -> list[TextBlockEvidence | TableEvidence]:
        """Text blocks and tables interleaved back into reading order."""
        elements: list[TextBlockEvidence | TableEvidence] = [*self.text_blocks, *self.tables]
        return sorted(elements, key=lambda element: element.sequence)
