# BOQ Test PDFs

1. `01_structured_boq.pdf` — clean baseline.
2. `02_multipage_boq.pdf` — multi-page continuation.
3. `03_reordered_columns_boq.pdf` — Quantity before Unit.
4. `04_missing_fields_boq.pdf` — missing units/quantities; do not hallucinate.
5. `05_totals_subtotals_boq.pdf` — totals/subtotals must not become items.
6. `06_nested_hierarchy_boq.pdf` — 3-level hierarchy and `level_path`.
7. `07_scanned_boq.pdf` — image-only/scanned-style PDF; tests OCR.
8. `08_rates_costs_boq.pdf` — rates, amounts, profit, summary rows.
9. `09_non_boq_document.pdf` — should not produce a fabricated BOQ.
10. `10_large_multipage_boq.pdf` — larger multi-page/context test.

Compare each `boq.json` and `extraction_report.json` with the source PDF.
