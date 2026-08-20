"""Token counting with the model's own tokenizer.

**No character heuristics.** `len(text) / 4` is not a token count; it is a guess
that varies by 40%+ between prose and the numeric, pipe-separated table rows a
BOQ prompt is mostly made of. If a real tokenizer cannot be loaded this module
raises `TokenizerUnavailableError` rather than returning an approximation, so a
benchmark result is never silently fabricated.

`openai/gpt-oss-120b` uses the **o200k_harmony** encoding, which `tiktoken`
ships. A model with no known encoding falls back to `o200k_base` and is flagged
`is_native=False` in the report — a documented proxy, not a silent substitution.

Attribution
-----------
`attribute()` splits a text's token count across consecutive segments *exactly*:
the per-segment counts always sum to the whole-text count. It does this by
walking the real token stream and assigning each token to the segment its first
byte falls in, rather than tokenizing segments separately — separate
tokenization overcounts, because a segment boundary forbids BPE merges that
happen in the real text (measured at +0.7% on the sample prompt).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.errors import ConfigurationError

#: Model-name fragment -> the encoding that model actually uses.
_NATIVE_ENCODINGS: tuple[tuple[str, str], ...] = (
    ("gpt-oss", "o200k_harmony"),
    ("gpt-4o", "o200k_base"),
    ("o200k", "o200k_base"),
)

#: Used when the model is unrecognised. Same BPE family as o200k_harmony, which
#: differs only by its added special tokens, so counts stay close for plain text.
_FALLBACK_ENCODING = "o200k_base"


class TokenizerUnavailableError(ConfigurationError):
    """No real tokenizer could be loaded, and estimating is not acceptable."""


@dataclass(frozen=True)
class TokenizerInfo:
    """Which tokenizer produced a set of counts, and how far to trust it."""

    encoding: str
    library: str
    library_version: str
    target_model: str
    is_native: bool

    @property
    def note(self) -> str:
        if self.is_native:
            return f"{self.encoding} is the encoding {self.target_model} itself uses."
        return (
            f"{self.target_model} has no known encoding here; counts use "
            f"{self.encoding} as a documented proxy and may differ slightly."
        )


@dataclass(frozen=True)
class TextMeasurement:
    """One measured string."""

    characters: int
    tokens: int

    @property
    def chars_per_token(self) -> float:
        return self.characters / self.tokens if self.tokens else 0.0


def encoding_for_model(model: str) -> tuple[str, bool]:
    """Return `(encoding_name, is_native)` for a model name."""
    lowered = model.lower()
    for fragment, encoding in _NATIVE_ENCODINGS:
        if fragment in lowered:
            return encoding, True
    return _FALLBACK_ENCODING, False


@lru_cache(maxsize=4)
def _load_encoding(name: str):
    """Load a tiktoken encoding, or explain precisely what is missing."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise TokenizerUnavailableError(
            "Token counting needs `tiktoken` (pip install tiktoken). "
            "The benchmark refuses to estimate tokens from character counts."
        ) from exc

    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise TokenizerUnavailableError(
            f"Could not load the '{name}' encoding: {exc}. tiktoken downloads "
            "encodings on first use, so this usually means no network access "
            "and no cached copy (see TIKTOKEN_CACHE_DIR)."
        ) from exc


def _library_version() -> str:
    try:
        import tiktoken

        return getattr(tiktoken, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        return "missing"


class TokenCounter:
    """Counts tokens the way the target model does."""

    def __init__(self, model: str, *, encoding: str | None = None) -> None:
        resolved, is_native = encoding_for_model(model)
        if encoding is not None:
            resolved, is_native = encoding, encoding == resolved
        self._encoding = _load_encoding(resolved)
        self._token_lengths: dict[int, int] = {}
        self.info = TokenizerInfo(
            encoding=resolved,
            library="tiktoken",
            library_version=_library_version(),
            target_model=model,
            is_native=is_native,
        )

    # -- counting ------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Tokenize text.

        `disallowed_special=()` matters: evidence comes from user-supplied PDFs,
        and a document containing the literal string `<|endoftext|>` would
        otherwise raise instead of being counted as the ordinary text it is.
        """
        return self._encoding.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def measure(self, text: str) -> TextMeasurement:
        return TextMeasurement(characters=len(text), tokens=self.count(text))

    # -- attribution ---------------------------------------------------

    def _byte_length(self, token: int) -> int:
        cached = self._token_lengths.get(token)
        if cached is None:
            cached = len(self._encoding.decode_single_token_bytes(token))
            self._token_lengths[token] = cached
        return cached

    def attribute(self, text: str, segments: list[str]) -> list[int]:
        """Split `text`'s tokens across `segments`, which must concatenate to it.

        A token straddling a boundary is charged to the segment containing its
        first byte. Counts therefore sum exactly to `count(text)`.
        """
        if "".join(segments) != text:
            raise ValueError("segments must concatenate to exactly the given text")

        tokens = self.encode(text)
        starts: list[int] = []
        offset = 0
        for token in tokens:
            starts.append(offset)
            offset += self._byte_length(token)

        counts: list[int] = []
        index = 0
        boundary = 0
        for segment in segments:
            boundary += len(segment.encode("utf-8"))
            taken = 0
            while index < len(starts) and starts[index] < boundary:
                taken += 1
                index += 1
            counts.append(taken)

        # Anything left over (possible only if byte lengths and the source text
        # disagree) is charged to the final segment rather than silently lost.
        if counts and index < len(starts):
            counts[-1] += len(starts) - index
        return counts
