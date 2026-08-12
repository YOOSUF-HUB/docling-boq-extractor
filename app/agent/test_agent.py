"""Manual live check of the BOQ agent against a saved evidence package.

    python -m app.agent.test_agent
    python -m app.agent.test_agent --evidence data/output/evidence.json --show-prompt

This makes a real Groq API call and needs `GROQ_API_KEY` in `.env`. It is a
development tool, not part of the automated suite: `pytest` never runs it
(`testpaths = ["tests"]`), and the deterministic agent tests in
`tests/test_agent.py` use mocked responses instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.agent.boq_agent import BOQAgent
from app.agent.prompts import build_user_prompt
from app.config import get_settings
from app.errors import AIError, ConfigurationError
from app.logging_config import configure_logging
from app.schemas.boq import BOQDocument
from app.schemas.evidence import EvidencePackage
from app.validation.boq_validator import validate_boq

DEFAULT_EVIDENCE = Path("tests/fixtures/evidence_sample_boq.json")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.agent.test_agent",
        description="Run the BOQ agent against an evidence fixture (live Groq call).",
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--show-prompt", action="store_true", help="Print the user prompt.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not args.evidence.exists():
        print(f"Evidence file not found: {args.evidence}", file=sys.stderr)
        return 1

    evidence = EvidencePackage.model_validate_json(args.evidence.read_text(encoding="utf-8"))
    settings = get_settings()

    if args.show_prompt:
        print("========== PROMPT ==========")
        print(build_user_prompt(evidence))
        print()

    try:
        result = BOQAgent(settings=settings).extract(evidence)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except AIError as exc:
        print(f"AI error: {exc}", file=sys.stderr)
        for issue in getattr(exc, "issues", []):
            print(f"  {issue.type.value}: {issue.message}", file=sys.stderr)
        return 2

    print("========== AI BOQ ==========")
    print(f"Model         : {result.model}")
    print(f"Attempts      : {result.attempts}")
    print(f"Schema mode   : {'json_schema' if result.used_json_schema else 'json_object'}")
    print(f"Duration      : {result.duration_seconds:.2f}s")
    print(f"Tokens        : {result.prompt_tokens} in / {result.completion_tokens} out")
    print(f"Items         : {len(result.items)}")
    print(f"Unresolved    : {len(result.unresolved)}")
    print()

    for item in result.items:
        rate = "" if item.fixed_rate is None else f"  rate={item.fixed_rate}"
        print(
            f"  {' > '.join(item.level_path):<32} {item.boq_item_code:<8} "
            f"{item.boq_item_name:<34} {item.quantity:>10} {item.unit}{rate}"
        )

    for issue in result.warnings + result.errors:
        print(f"  [{issue.type.value}] {issue.message}")

    boq = BOQDocument.assemble(result.items, filename=evidence.document.filename)
    report = validate_boq(boq)

    print()
    print("========== VALIDATION ==========")
    print(f"Status        : {report.status.value}")
    for issue in report.errors + report.warnings:
        print(f"  {issue.type.value}: {issue.message}")

    print()
    print("========== CANONICAL BOQ ==========")
    print(json.dumps(json.loads(boq.model_dump_json()), indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
