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

    # --- LLM (Phase 4+) ---
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

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
