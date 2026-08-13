"""CLI tests: artifact writing, printed sections and exit codes.

The pipeline itself is stubbed out — what is under test here is the wiring
between a `BOQResult` and what the shell sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import main as cli
from app.errors import AIRequestError, ConfigurationError, DocumentError, ExtractionError
from app.schemas.boq import BOQDocument, BOQItem
from app.schemas.evidence import EvidencePackage
from app.schemas.report import (
    ExtractionIssue,
    ExtractionReport,
    ExtractionStatistics,
    ExtractionStatus,
    IssueType,
)
from app.services.extraction_service import BOQResult, EvidenceMetadata, LLMMetadata
from tests.test_extraction_service import StubProcessor

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def evidence() -> EvidencePackage:
    return EvidencePackage.model_validate_json(
        (FIXTURES / "evidence_sample_boq.json").read_text()
    )


@pytest.fixture(scope="module")
def docling_result():
    return StubProcessor().process("data/input/sample_boq.pdf")


def make_boq() -> BOQDocument:
    item = BOQItem(
        level_path=["EARTHWORKS"],
        boq_item_code="1.1",
        boq_item_name="Excavation for foundations",
        quantity=125.5,
        unit="m3",
    )
    return BOQDocument.assemble([item], filename="sample_boq.pdf")


def make_metadata() -> EvidenceMetadata:
    return EvidenceMetadata(
        filename="sample_boq.pdf",
        page_count=1,
        extracted_at="2026-08-12T09:14:00Z",
        evidence_version="1.0",
        statistics={},
    )


def make_result(status: ExtractionStatus = ExtractionStatus.SUCCESS) -> BOQResult:
    return BOQResult(
        boq=make_boq(),
        report=ExtractionReport(
            status=status,
            statistics=ExtractionStatistics(pages_processed=1, tables_detected=1, items_extracted=1),
        ),
        evidence_metadata=make_metadata(),
        llm=LLMMetadata(
            model="openai/gpt-oss-120b",
            attempts=1,
            duration_seconds=12.4,
            used_json_schema=True,
            prompt_tokens=1853,
            completion_tokens=1455,
        ),
    )


def make_failed_result() -> BOQResult:
    """A run that produced no BOQ at all — the model was never even called."""
    return BOQResult(
        boq=None,
        report=ExtractionReport(
            status=ExtractionStatus.FAILED,
            errors=[
                ExtractionIssue(type=IssueType.NO_BOQ_DETECTED, message="nothing to interpret")
            ],
        ),
        evidence_metadata=make_metadata(),
    )


def run_cli(monkeypatch, tmp_path, evidence, docling_result, result=None, error=None) -> int:
    """Run `main()` with both pipeline halves stubbed."""

    def fake_document_stage(path, **kwargs):
        return docling_result, evidence

    def fake_extract(package, **kwargs):
        if error is not None:
            raise error
        return result if result is not None else make_result()

    monkeypatch.setattr(cli, "run_document_stage", fake_document_stage)
    monkeypatch.setattr(cli, "extract_boq_from_evidence", fake_extract)
    return cli.main(["--input", "x.pdf", "--output-dir", str(tmp_path)])


# --- artifacts --------------------------------------------------------


def test_document_artifacts_are_written(tmp_path, docling_result, evidence):
    outputs = cli.write_outputs(docling_result, evidence, tmp_path)

    assert set(outputs) == {"markdown", "document_json", "evidence"}
    assert (tmp_path / "evidence.json").exists()
    assert (tmp_path / "docling_output.md").read_text().strip()


def test_boq_and_report_are_written_when_extraction_ran(tmp_path, docling_result, evidence):
    outputs = cli.write_outputs(docling_result, evidence, tmp_path, make_result())

    assert set(outputs) == {"markdown", "document_json", "evidence", "boq", "report"}
    boq = json.loads((tmp_path / "boq.json").read_text())
    report = json.loads((tmp_path / "extraction_report.json").read_text())
    assert boq["items"][0]["boq_item_code"] == "1.1"
    assert report["status"] == "success"


def test_a_failed_run_still_writes_its_report(tmp_path, docling_result, evidence):
    outputs = cli.write_outputs(docling_result, evidence, tmp_path, make_failed_result())

    assert "boq" not in outputs
    assert json.loads((tmp_path / "extraction_report.json").read_text())["status"] == "failed"


# --- printed output ---------------------------------------------------


def test_every_pipeline_stage_is_printed(monkeypatch, tmp_path, evidence, docling_result, capsys):
    run_cli(monkeypatch, tmp_path, evidence, docling_result)
    printed = capsys.readouterr().out

    sections = ["DOCLING", "EVIDENCE", "AI BOQ", "VALIDATION", "FINAL BOQ"]
    positions = [printed.index(f"========== {name} ==========") for name in sections]
    assert positions == sorted(positions)


def test_the_final_boq_section_shows_the_items(monkeypatch, tmp_path, evidence, docling_result, capsys):
    run_cli(monkeypatch, tmp_path, evidence, docling_result)
    printed = capsys.readouterr().out

    assert "Excavation for foundations" in printed
    assert "EARTHWORKS" in printed


def test_a_result_without_a_boq_says_so(monkeypatch, tmp_path, evidence, docling_result, capsys):
    run_cli(monkeypatch, tmp_path, evidence, docling_result, result=make_failed_result())

    assert "No canonical BOQ was produced" in capsys.readouterr().out


def test_evidence_only_stops_before_the_llm(monkeypatch, tmp_path, evidence, docling_result, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("--evidence-only must not call the model")

    monkeypatch.setattr(cli, "run_document_stage", lambda path, **kwargs: (docling_result, evidence))
    monkeypatch.setattr(cli, "extract_boq_from_evidence", fail)

    code = cli.main(["--evidence-only", "--output-dir", str(tmp_path)])
    printed = capsys.readouterr().out

    assert code == cli.EXIT_OK
    assert "========== EVIDENCE ==========" in printed
    assert "========== AI BOQ ==========" not in printed
    assert not (tmp_path / "boq.json").exists()


# --- exit codes -------------------------------------------------------


def report_with(*issues: ExtractionIssue) -> ExtractionReport:
    status = ExtractionStatus.FAILED if issues else ExtractionStatus.SUCCESS
    return ExtractionReport(status=status, errors=list(issues))


@pytest.mark.parametrize(
    ("issue_type", "expected"),
    [
        (IssueType.DOCUMENT_UNREADABLE, cli.EXIT_DOCUMENT_ERROR),
        (IssueType.EMPTY_DOCUMENT, cli.EXIT_DOCUMENT_ERROR),
        (IssueType.NO_BOQ_DETECTED, cli.EXIT_EXTRACTION_ERROR),
        (IssueType.NO_ITEMS_EXTRACTED, cli.EXIT_EXTRACTION_ERROR),
        (IssueType.LLM_INVALID_OUTPUT, cli.EXIT_EXTRACTION_ERROR),
        (IssueType.TOTAL_ROW_AS_ITEM, cli.EXIT_VALIDATION_ERROR),
        (IssueType.DUPLICATE_ITEM_CODE, cli.EXIT_VALIDATION_ERROR),
    ],
)
def test_exit_code_reflects_the_stage_that_failed(issue_type, expected):
    report = report_with(ExtractionIssue(type=issue_type, message="x"))

    assert cli.exit_code_for(report) == expected


def test_a_clean_report_exits_zero():
    assert cli.exit_code_for(report_with()) == cli.EXIT_OK


def test_warnings_alone_do_not_fail_the_run(monkeypatch, tmp_path, evidence, docling_result):
    partial = make_result(status=ExtractionStatus.PARTIAL)

    assert run_cli(monkeypatch, tmp_path, evidence, docling_result, result=partial) == cli.EXIT_OK


def test_extraction_failure_exit_code(monkeypatch, tmp_path, evidence, docling_result):
    code = run_cli(monkeypatch, tmp_path, evidence, docling_result, result=make_failed_result())

    assert code == cli.EXIT_EXTRACTION_ERROR


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DocumentError("not a PDF"), cli.EXIT_DOCUMENT_ERROR),
        (ExtractionError("Docling failed"), cli.EXIT_EXTRACTION_ERROR),
    ],
)
def test_document_stage_errors_exit_codes(monkeypatch, tmp_path, error, expected):
    def fail(path, **kwargs):
        raise error

    monkeypatch.setattr(cli, "run_document_stage", fail)

    assert cli.main(["--output-dir", str(tmp_path)]) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConfigurationError("GROQ_API_KEY is not set"), cli.EXIT_CONFIG_ERROR),
        (AIRequestError("connection refused"), cli.EXIT_AI_ERROR),
    ],
)
def test_agent_stage_errors_exit_codes(
    monkeypatch, tmp_path, evidence, docling_result, error, expected
):
    assert run_cli(monkeypatch, tmp_path, evidence, docling_result, error=error) == expected


# --- --validate-boq ---------------------------------------------------


def test_validate_boq_still_works_without_docling_or_the_llm(tmp_path, capsys):
    code = cli.main(["--validate-boq", str(FIXTURES / "valid_boq.json")])

    assert code == cli.EXIT_OK
    assert "Status        : success" in capsys.readouterr().out


def test_validate_boq_rejects_a_pms_field(tmp_path, capsys):
    payload = json.loads((FIXTURES / "valid_boq.json").read_text())
    payload["items"][0]["id"] = 42
    path = tmp_path / "with_id.json"
    path.write_text(json.dumps(payload))

    code = cli.main(["--validate-boq", str(path)])

    assert code == cli.EXIT_VALIDATION_ERROR
    assert "SCHEMA_VALIDATION_FAILED" in capsys.readouterr().out
