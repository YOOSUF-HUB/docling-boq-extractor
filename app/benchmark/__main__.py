"""Token-usage benchmark CLI.

    python -m app.benchmark                                  # the sample PDF
    python -m app.benchmark --input data/input/big_boq.pdf
    python -m app.benchmark --evidence data/output/evidence.json
    python -m app.benchmark --corpus data/input             # every PDF in a folder
    python -m app.benchmark --live                          # make the real call

Static mode needs no API key and spends no tokens: it measures the request the
pipeline *would* send. `--live` additionally runs the extraction through a
recording client and reports what Groq charged.

Docling is the slow part, so evidence packages are cached under
`data/benchmark/evidence/` and reused; `--refresh` re-converts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.benchmark import report as report_renderer
from app.benchmark.metrics import TokenBenchmark
from app.benchmark.token_benchmark import (
    PromptDriftError,
    benchmark_evidence,
    benchmark_pdf,
    load_evidence,
)
from app.benchmark.tokenizer import TokenCounter, TokenizerUnavailableError
from app.config import Settings, get_settings
from app.errors import AIRequestError, ConfigurationError, DocumentError, ExtractionError
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_DOCUMENT_ERROR = 1
EXIT_BENCHMARK_ERROR = 2
EXIT_BUDGET_EXCEEDED = 3
EXIT_CONFIG_ERROR = 4
EXIT_AI_ERROR = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="app.benchmark",
        description="Measure where an extraction request's tokens are consumed.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        metavar="PDF",
        help="A PDF to measure. Repeatable. Defaults to the sample PDF.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=None,
        metavar="JSON",
        help="An evidence.json to measure directly, skipping Docling. Repeatable.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        metavar="DIR",
        help="Measure every PDF in a directory.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also call Groq and report charged usage (needs GROQ_API_KEY).",
    )
    parser.add_argument(
        "--envelope-tokens",
        type=int,
        help="Chat-format overhead measured by an earlier live run, applied to static results.",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=settings.groq_tpm_limit,
        help="Tokens-per-minute limit to check against (default: %(default)s)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=settings.groq_max_completion_tokens,
        help="Completion budget to model (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.benchmark_dir,
        help="Where benchmark JSON and cached evidence are written (default: %(default)s)",
    )
    parser.add_argument("--no-write", action="store_true", help="Print only; write nothing.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached evidence.")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Heaviest prompt components to list (0 disables, default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-exceed",
        action="store_true",
        help=f"Exit {EXIT_BUDGET_EXCEEDED} if any request exceeds the TPM limit.",
    )
    parser.add_argument("--log-level", default=settings.log_level)
    return parser.parse_args(argv)


def resolve_targets(args: argparse.Namespace, settings: Settings) -> tuple[list[Path], list[Path]]:
    """Which PDFs and which evidence files to measure."""
    pdfs = list(args.input or [])
    evidence = list(args.evidence or [])

    if args.corpus is not None:
        pdfs.extend(sorted(args.corpus.glob("*.pdf")))

    if not pdfs and not evidence:
        pdfs = [settings.default_input_pdf]
    return pdfs, evidence


def cached_evidence_path(output_dir: Path, pdf: Path) -> Path:
    return output_dir / "evidence" / f"{pdf.stem}.evidence.json"


def measure_pdf(
    pdf: Path,
    *,
    args: argparse.Namespace,
    settings: Settings,
    counter: TokenCounter,
) -> TokenBenchmark:
    """Measure one PDF, reusing cached evidence when Docling has already run."""
    cache = cached_evidence_path(args.output_dir, pdf)

    if cache.exists() and not args.refresh:
        logger.info("Using cached evidence for %s", pdf.name)
        benchmark = benchmark_evidence(
            load_evidence(cache),
            settings=settings,
            counter=counter,
            source=str(pdf),
            file_size_bytes=pdf.stat().st_size if pdf.exists() else None,
            live=args.live,
            envelope_tokens=args.envelope_tokens,
        )
        print("\n(evidence from cache — use --refresh for Docling JSON metrics)")
        return benchmark

    benchmark, evidence = benchmark_pdf(
        pdf,
        settings=settings,
        counter=counter,
        live=args.live,
        envelope_tokens=args.envelope_tokens,
    )
    if not args.no_write:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    return benchmark


def write_benchmark(benchmark: TokenBenchmark, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{Path(benchmark.source).stem or 'benchmark'}.benchmark.json"
    path.write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    settings = get_settings().model_copy(
        update={
            "groq_tpm_limit": args.tpm_limit,
            "groq_max_completion_tokens": args.max_completion_tokens,
        }
    )

    try:
        counter = TokenCounter(settings.groq_model)
    except TokenizerUnavailableError as exc:
        logger.error("%s", exc)
        return EXIT_CONFIG_ERROR

    pdfs, evidence_files = resolve_targets(args, settings)
    benchmarks: list[TokenBenchmark] = []
    written: list[Path] = []

    try:
        for pdf in pdfs:
            benchmarks.append(measure_pdf(pdf, args=args, settings=settings, counter=counter))
        for path in evidence_files:
            benchmarks.append(
                benchmark_evidence(
                    load_evidence(path),
                    settings=settings,
                    counter=counter,
                    source=str(path),
                    live=args.live,
                    envelope_tokens=args.envelope_tokens,
                )
            )
    except DocumentError as exc:
        logger.error("Document error: %s", exc)
        return EXIT_DOCUMENT_ERROR
    except ExtractionError as exc:
        logger.error("Extraction error: %s", exc)
        return EXIT_DOCUMENT_ERROR
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIG_ERROR
    except AIRequestError as exc:
        logger.error("The model could not be reached: %s", exc)
        return EXIT_AI_ERROR
    except PromptDriftError as exc:
        logger.error("%s", exc)
        return EXIT_BENCHMARK_ERROR
    except (OSError, ValueError) as exc:
        logger.error("Could not read the input: %s", exc)
        return EXIT_DOCUMENT_ERROR

    for benchmark in benchmarks:
        print(report_renderer.render(benchmark, top=args.top))
        if not args.no_write:
            written.append(write_benchmark(benchmark, args.output_dir))

    comparison = report_renderer.render_comparison(benchmarks)
    if comparison:
        print(comparison)

    if written:
        print("\nWritten:")
        for path in written:
            print(f"  {path}")
    print()

    if args.fail_on_exceed and any(b.budget.exceeds_limit for b in benchmarks):
        return EXIT_BUDGET_EXCEEDED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
