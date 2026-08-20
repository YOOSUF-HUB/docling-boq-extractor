"""Splitting the user prompt into the parts that consume its tokens.

`build_user_prompt()` returns one string; to say *where* its tokens go, that
string has to be broken back into labelled pieces. This module mirrors the
structure of `build_user_prompt` and reuses its own element renderers, so the
per-element text can never drift from what is actually sent.

The frame around the elements (header, page markers, closing instruction) is
mirrored rather than shared, so `test_token_benchmark.py` asserts on every
available evidence package that the segments concatenate to *exactly*
`build_user_prompt(evidence)`. If the prompt is ever reworded, that test fails
rather than the benchmark quietly mis-attributing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.agent.prompts import _render_table, _render_text_block
from app.schemas.evidence import EvidencePackage, TableEvidence


class SegmentKind(StrEnum):
    """What a run of prompt lines is."""

    HEADER = "header"
    PAGE_MARKER = "page_marker"
    TEXT_BLOCK = "text_block"
    TABLE = "table"
    FOOTER = "footer"

    @property
    def is_document_content(self) -> bool:
        """True for segments whose size grows with the document."""
        return self in {SegmentKind.TEXT_BLOCK, SegmentKind.TABLE, SegmentKind.PAGE_MARKER}


@dataclass(frozen=True)
class PromptSegment:
    """One labelled run of consecutive prompt lines."""

    kind: SegmentKind
    label: str
    lines: list[str] = field(default_factory=list)
    sequence: int | None = None
    page_number: int | None = None
    row_count: int | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def split_user_prompt(evidence: EvidencePackage) -> list[PromptSegment]:
    """Rebuild `build_user_prompt(evidence)` as labelled segments, in order."""
    document = evidence.document
    segments = [
        PromptSegment(
            kind=SegmentKind.HEADER,
            label="prompt header (document facts + reading instructions)",
            lines=[
                "DOCUMENT EVIDENCE",
                f"filename: {document.filename}",
                f"pages: {document.page_count}",
                f"tables: {evidence.statistics.tables}",
                "",
                "Elements are listed in document reading order. `seq` is the reading",
                "position; use `table#N row M` when referring to evidence.",
                "",
            ],
        )
    ]

    current_page: Any = _NO_PAGE_YET
    for element in evidence.elements_in_document_order():
        if element.page_number != current_page:
            current_page = element.page_number
            page_label = current_page if current_page is not None else "unknown"
            segments.append(
                PromptSegment(
                    kind=SegmentKind.PAGE_MARKER,
                    label=f"page {page_label} marker",
                    lines=[f"--- PAGE {page_label} ---"],
                    page_number=element.page_number,
                )
            )

        if isinstance(element, TableEvidence):
            segments.append(
                PromptSegment(
                    kind=SegmentKind.TABLE,
                    label=(
                        f"table#{element.table_index} "
                        f"({len(element.rows)} rows x {element.num_cols} cols)"
                    ),
                    lines=_render_table(element),
                    sequence=element.sequence,
                    page_number=element.page_number,
                    row_count=len(element.rows),
                )
            )
        else:
            segments.append(
                PromptSegment(
                    kind=SegmentKind.TEXT_BLOCK,
                    label=f"{element.label}: {_preview(element.text)}",
                    lines=[_render_text_block(element)],
                    sequence=element.sequence,
                    page_number=element.page_number,
                )
            )

    segments.append(
        PromptSegment(
            kind=SegmentKind.FOOTER,
            label="closing instruction",
            lines=["", "Extract the BOQ from this evidence."],
        )
    )
    return segments


def segment_texts(segments: list[PromptSegment]) -> list[str]:
    """Segment strings that concatenate to exactly the rendered prompt.

    `build_user_prompt` joins its lines with newlines, so every segment but the
    last carries the separator that follows it.
    """
    texts = [segment.text for segment in segments]
    return [text + "\n" for text in texts[:-1]] + texts[-1:] if texts else []


#: Mirrors the sentinel in `app.agent.prompts`: the first element must emit a
#: page marker even when its page number is None.
_NO_PAGE_YET = object()


def _preview(text: str, limit: int = 48) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
