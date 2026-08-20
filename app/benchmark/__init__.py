"""Token-usage measurement for the extraction pipeline.

This package **observes** the pipeline; it never changes it. Nothing under
`app/agent`, `app/document`, `app/schemas`, `app/validation` or `app/services`
imports anything from here, and the benchmark reaches production code only
through public seams that already existed:

* `app.agent.prompts.SYSTEM_PROMPT` / `build_user_prompt()` — the exact strings
  the agent sends, called rather than re-implemented.
* `BOQAgent(client=...)` — a recording wrapper is injected in place of the Groq
  client, so a live run measures the real request without patching the agent.

The output is a `TokenBenchmark` record: a decomposition of one extraction
request into the components that consume its token budget.
"""

from app.benchmark.metrics import TokenBenchmark
from app.benchmark.token_benchmark import benchmark_evidence, benchmark_pdf, load_evidence
from app.benchmark.tokenizer import TokenCounter, TokenizerUnavailableError

__all__ = [
    "TokenBenchmark",
    "TokenCounter",
    "TokenizerUnavailableError",
    "benchmark_evidence",
    "benchmark_pdf",
    "load_evidence",
]
