from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.celery_app import DOCUMENT_INGESTION_TASK_NAME, celery_app
from backend.app.celery_config import (
    CeleryConfigurationError,
    CelerySettings,
    ingestion_retry_countdown,
)


def test_celery_app_uses_safe_delivery_settings() -> None:
    assert celery_app.conf.task_serializer == "json"
    assert list(celery_app.conf.accept_content) == ["json"]
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_ignore_result is True
    assert celery_app.conf.task_store_errors_even_if_ignored is False
    assert celery_app.conf.result_backend is None
    assert celery_app.conf.broker_transport_options["confirm_publish"] is True
    assert celery_app.conf.task_routes[DOCUMENT_INGESTION_TASK_NAME]["queue"] == (
        celery_app.conf.task_default_queue
    )


def test_registered_ingestion_task_inherits_delivery_and_retry_limits() -> None:
    celery_app.loader.import_default_modules()
    task = celery_app.tasks[DOCUMENT_INGESTION_TASK_NAME]

    assert task.ignore_result is True
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True
    assert task.max_retries == 3


def test_default_retry_count_means_three_retries_after_initial_attempt(monkeypatch) -> None:
    for name in (
        "DOCUMENT_INGESTION_MAX_RETRIES",
        "DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS",
        "DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = CelerySettings.from_environment()

    assert settings.ingestion_max_retries == 3


def test_retry_countdown_uses_capped_full_jitter(monkeypatch) -> None:
    settings = CelerySettings(
        broker_url="amqp://example.test//",
        ingestion_queue="documents",
        ingestion_max_retries=3,
        retry_backoff_seconds=5,
        retry_backoff_max_seconds=12,
    )
    observed_bounds: list[tuple[int, int]] = []

    def choose_upper_bound(lower: int, upper: int) -> int:
        observed_bounds.append((lower, upper))
        return upper

    monkeypatch.setattr("backend.app.celery_config.random.randint", choose_upper_bound)

    assert ingestion_retry_countdown(0, settings=settings) == 5
    assert ingestion_retry_countdown(1, settings=settings) == 10
    assert ingestion_retry_countdown(2, settings=settings) == 12
    assert observed_bounds == [(0, 5), (0, 10), (0, 12)]


def test_retry_configuration_rejects_an_invalid_cap(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS", "10")
    monkeypatch.setenv("DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS", "5")

    with pytest.raises(CeleryConfigurationError, match="greater than or equal"):
        CelerySettings.from_environment()


def test_worker_task_module_imports_without_fastapi_or_main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    command = (
        "import sys; "
        "import backend.app.tasks.ingest_document; "
        "assert 'main' not in sys.modules; "
        "assert 'fastapi' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
