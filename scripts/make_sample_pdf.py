"""Generate the Phase 1 sample BOQ PDF fixture.

The repository does not ship binary fixtures; this script regenerates
`data/input/sample_boq.pdf` deterministically so anyone can reproduce the
pipeline locally.

Usage:
    python -m scripts.make_sample_pdf [--output data/input/sample_boq.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "input" / "sample_boq.pdf"

PROJECT_TITLE = "PROJECT: Sample Commercial Building"
DOCUMENT_TITLE = "BILL OF QUANTITIES"

HEADER_ROW = ["Item", "Description", "Unit", "Quantity"]

# (kind, cells) where kind is "section" or "item".
BOQ_ROWS: list[tuple[str, list[str]]] = [
    ("section", ["1", "EARTHWORKS", "", ""]),
    ("item", ["1.1", "Excavation for foundations", "m3", "125.50"]),
    ("item", ["1.2", "Filling with selected material", "m3", "80.00"]),
    ("section", ["2", "CONCRETE WORKS", "", ""]),
    ("item", ["2.1", "Blinding concrete", "m3", "15.00"]),
    ("item", ["2.2", "Reinforced concrete foundations", "m3", "50.00"]),
    ("item", ["2.3", "Reinforced concrete columns", "m3", "35.00"]),
    ("section", ["3", "MASONRY", "", ""]),
    ("item", ["3.1", "Brickwork in cement mortar", "m2", "250.00"]),
]


def build_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Sample BOQ",
    )

    table_data = [HEADER_ROW] + [cells for _, cells in BOQ_ROWS]

    style_commands = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (3, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    # Section rows are visually distinct headings inside the table.
    for offset, (kind, _) in enumerate(BOQ_ROWS, start=1):
        if kind == "section":
            style_commands += [
                ("FONTNAME", (0, offset), (-1, offset), "Helvetica-Bold"),
                ("BACKGROUND", (0, offset), (-1, offset), colors.HexColor("#F2F2F2")),
            ]

    table = Table(
        table_data,
        colWidths=[20 * mm, 90 * mm, 20 * mm, 30 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle(style_commands))

    story = [
        Paragraph(PROJECT_TITLE, styles["Heading2"]),
        Spacer(1, 6 * mm),
        Paragraph(DOCUMENT_TITLE, styles["Heading1"]),
        Spacer(1, 6 * mm),
        table,
    ]
    doc.build(story)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the sample BOQ PDF fixture.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    path = build_pdf(args.output)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
