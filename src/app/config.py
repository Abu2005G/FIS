"""
Application settings, read once from environment variables (and a local .env).

Nothing secret has a default. The service refuses to start without an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

LLM_PROVIDERS = ("none", "ollama", "openai", "mock")

# Values copied from old examples that must never be accepted as a real key
_PLACEHOLDER_KEYS = {"my-secret-key-123", "test-key", "changeme", "your-api-key"}
MIN_API_KEY_LENGTH = 24


class ConfigurationError(RuntimeError):
    """The environment is missing or holds an invalid setting."""


@dataclass(frozen=True)
class Settings:
    environment: str
    api_key: str
    database_url: str
    log_level: str

    llm_provider: str
    llm_model: str
    ollama_base_url: str
    openai_api_key: Optional[str]
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_max_transactions: int
    llm_batch_size: int

    max_upload_bytes: int
    statement_day_first: bool

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none"


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a whole number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> Settings:
    """Build settings from the environment, validating as it goes."""
    load_dotenv()

    environment = os.getenv("ENVIRONMENT", "development").strip().lower()

    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "API_KEY is not set. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if api_key.lower() in _PLACEHOLDER_KEYS or len(api_key) < MIN_API_KEY_LENGTH:
        raise ConfigurationError(
            f"API_KEY must be a random secret of at least {MIN_API_KEY_LENGTH} characters"
        )

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ConfigurationError(
            f"LLM_PROVIDER must be one of {', '.join(LLM_PROVIDERS)}, got {provider!r}"
        )
    if provider == "mock" and environment == "production":
        raise ConfigurationError("LLM_PROVIDER=mock is not allowed in production")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    if provider == "openai" and not openai_api_key:
        raise ConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY")

    default_model = {"ollama": "llama3.2", "openai": "gpt-4o-mini"}.get(provider, "none")

    return Settings(
        environment=environment,
        api_key=api_key,
        database_url=os.getenv("DATABASE_URL", "sqlite:///./fis.db").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "").strip() or default_model,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        openai_api_key=openai_api_key,
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 120.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 2, minimum=0),
        llm_max_transactions=_int("LLM_MAX_TRANSACTIONS", 150),
        llm_batch_size=_int("LLM_BATCH_SIZE", 25),
        max_upload_bytes=_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        statement_day_first=_bool("STATEMENT_DAY_FIRST", True),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return load_settings()
