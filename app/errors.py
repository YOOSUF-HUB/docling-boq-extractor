"""Error hierarchy for the extraction pipeline.

The pipeline deliberately distinguishes *where* a failure came from, so the
extraction report (and later the API layer) can tell a user whether the
document, the extraction, the LLM, or validation was at fault.
"""


class BOQExtractionError(Exception):
    """Base class for every error raised by this application."""


class DocumentError(BOQExtractionError):
    """The input document is missing, unreadable, or not a usable PDF."""


class ExtractionError(BOQExtractionError):
    """Docling ran but could not produce a usable document representation."""


class ConfigurationError(BOQExtractionError):
    """Required configuration (e.g. GROQ_API_KEY) is missing."""


class AIError(BOQExtractionError):
    """Base class for failures involving the LLM."""


class AIRequestError(AIError):
    """The request never produced a usable response: transport, auth,
    rate limit, timeout, or a server-side error."""


class AIResponseError(AIError):
    """The model answered, but the answer is not valid structured output.

    Carries the issues so the caller can build a failed extraction report
    instead of guessing what went wrong.
    """

    def __init__(self, message: str, issues: list | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []
