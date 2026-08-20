"""Application configuration, loaded from the environment / `.env`.

Secrets are never hardcoded and never logged. `GROQ_API_KEY` is optional at this
stage because Phase 1 does not call the LLM.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    #: 0 keeps extraction as reproducible as the model allows.
    groq_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Counts towards Groq's tokens-per-minute budget *before* the call is made,
    #: so an oversized value gets a 413 on small tiers (free tier is 8k TPM).
    groq_max_completion_tokens: int = Field(default=4000, gt=0)
    groq_timeout_seconds: float = Field(default=120.0, gt=0)
    #: Transport-level retries (connection errors, 429, 5xx) done by the SDK.
    groq_max_retries: int = Field(default=2, ge=0)
    #: Attempts at getting schema-valid output; >1 enables one repair round.
    llm_max_attempts: int = Field(default=2, ge=1)
    #: Tokens-per-minute ceiling a request is measured against. Read only by
    #: `app.benchmark`; the real limit is enforced by Groq, and the extraction
    #: pipeline neither reads nor honours this value.
    groq_tpm_limit: int = Field(default=8000, gt=0)

    # --- Logging ---
    log_level: str = "INFO"

    # --- Docling ---
    docling_do_ocr: bool = True
    docling_do_table_structure: bool = True
    docling_table_cell_matching: bool = True

    # --- Limits ---
    max_pdf_size_mb: int = Field(default=50, gt=0)

    # --- Paths ---
    base_dir: Path = BASE_DIR
    input_dir: Path = BASE_DIR / "data" / "input"
    output_dir: Path = BASE_DIR / "data" / "output"
    debug_dir: Path = BASE_DIR / "data" / "debug"
    benchmark_dir: Path = BASE_DIR / "data" / "benchmark"

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def default_input_pdf(self) -> Path:
        return self.input_dir / "sample_boq.pdf"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
