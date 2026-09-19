"""Environment-backed settings for document-ingestion task delivery."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()

DEFAULT_BROKER_URL = "amqp://guest:guest@localhost:5672//"
DEFAULT_INGESTION_QUEUE = "document_ingestion"


class CeleryConfigurationError(RuntimeError):
    """Raised when worker or retry configuration is invalid."""


def _integer_setting(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise CeleryConfigurationError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise CeleryConfigurationError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True)
class CelerySettings:
    broker_url: str = field(repr=False)
    ingestion_queue: str
    ingestion_max_retries: int
    retry_backoff_seconds: int
    retry_backoff_max_seconds: int

    @classmethod
    def from_environment(cls) -> "CelerySettings":
        broker_url = (os.getenv("CELERY_BROKER_URL") or DEFAULT_BROKER_URL).strip()
        ingestion_queue = (
            os.getenv("DOCUMENT_INGESTION_QUEUE") or DEFAULT_INGESTION_QUEUE
        ).strip()
        settings = cls(
            broker_url=broker_url,
            ingestion_queue=ingestion_queue,
            ingestion_max_retries=_integer_setting(
                "DOCUMENT_INGESTION_MAX_RETRIES", 3, minimum=0
            ),
            retry_backoff_seconds=_integer_setting(
                "DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS", 5, minimum=1
            ),
            retry_backoff_max_seconds=_integer_setting(
                "DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS", 300, minimum=1
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.broker_url:
            raise CeleryConfigurationError("CELERY_BROKER_URL must not be empty.")
        if not self.ingestion_queue:
            raise CeleryConfigurationError("DOCUMENT_INGESTION_QUEUE must not be empty.")
        if self.retry_backoff_max_seconds < self.retry_backoff_seconds:
            raise CeleryConfigurationError(
                "DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS must be greater than or "
                "equal to DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS."
            )


@lru_cache(maxsize=1)
def get_celery_settings() -> CelerySettings:
    return CelerySettings.from_environment()


def ingestion_retry_countdown(
    current_retry: int,
    *,
    settings: CelerySettings | None = None,
) -> int:
    """Return capped exponential backoff with full jitter for the next retry."""
    if current_retry < 0:
        raise ValueError("current_retry must not be negative.")
    selected = settings or get_celery_settings()
    upper_bound = min(
        selected.retry_backoff_max_seconds,
        selected.retry_backoff_seconds * (2**current_retry),
    )
    return random.randint(0, upper_bound)
