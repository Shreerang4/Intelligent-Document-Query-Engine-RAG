from __future__ import annotations

import inspect
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.app.services.document_ingestion import (
    DocumentIngestionNotFoundError,
    DocumentIngestionResult,
    RetryableDocumentIngestionError,
)
from backend.app.tasks import ingest_document as task_module
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
)


class RetryRequested(Exception):
    pass


def _run_with_retry_count(retry_count: int, document_id: str) -> None:
    task_module.ingest_document_task.push_request(retries=retry_count)
    try:
        task_module.ingest_document_task.run(document_id)
    finally:
        task_module.ingest_document_task.pop_request()


def test_task_signature_receives_only_document_id() -> None:
    assert list(inspect.signature(task_module.ingest_document_task.run).parameters) == [
        "document_id"
    ]


def test_successful_task_calls_ingestion_once(monkeypatch) -> None:
    document_id = str(uuid4())
    calls: list[str] = []

    def ingest(selected_document_id: str) -> DocumentIngestionResult:
        calls.append(selected_document_id)
        return DocumentIngestionResult(selected_document_id, DOCUMENT_STATUS_READY, 2)

    monkeypatch.setattr(task_module.ingestion_service, "ingest_document", ingest)

    _run_with_retry_count(0, document_id)

    assert calls == [document_id]


def test_permanent_failed_result_does_not_retry(monkeypatch) -> None:
    document_id = str(uuid4())
    monkeypatch.setattr(
        task_module.ingestion_service,
        "ingest_document",
        lambda selected: DocumentIngestionResult(selected, DOCUMENT_STATUS_FAILED, 0),
    )
    monkeypatch.setattr(
        task_module.ingest_document_task,
        "retry",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not retry")),
    )

    _run_with_retry_count(0, document_id)


def test_missing_document_is_permanent_and_does_not_retry(monkeypatch) -> None:
    def missing(_document_id: str):
        raise DocumentIngestionNotFoundError("missing")

    monkeypatch.setattr(task_module.ingestion_service, "ingest_document", missing)
    monkeypatch.setattr(
        task_module.ingest_document_task,
        "retry",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not retry")),
    )

    _run_with_retry_count(0, str(uuid4()))


@pytest.mark.parametrize("retry_count", [0, 1, 2])
def test_retryable_failure_requeues_then_requests_celery_retry(
    monkeypatch,
    retry_count,
) -> None:
    document_id = str(uuid4())
    events: list[tuple[str, object]] = []

    def transient(_document_id: str):
        raise RetryableDocumentIngestionError("temporary")

    def prepare(selected: str) -> DocumentIngestionResult:
        events.append(("prepare", selected))
        return DocumentIngestionResult(selected, DOCUMENT_STATUS_QUEUED, 0)

    def retry(**kwargs):
        events.append(("retry", kwargs))
        raise RetryRequested

    monkeypatch.setattr(task_module.ingestion_service, "ingest_document", transient)
    monkeypatch.setattr(task_module.ingestion_service, "prepare_document_ingestion_retry", prepare)
    monkeypatch.setattr(task_module, "ingestion_retry_countdown", lambda *_args, **_kwargs: 17)
    monkeypatch.setattr(task_module.ingest_document_task, "retry", retry)

    with pytest.raises(RetryRequested):
        _run_with_retry_count(retry_count, document_id)

    assert events[0] == ("prepare", document_id)
    assert events[1][0] == "retry"
    assert events[1][1]["countdown"] == 17
    assert events[1][1]["max_retries"] == 3


def test_retry_state_write_failure_does_not_suppress_retry(monkeypatch, caplog) -> None:
    def transient(_document_id: str):
        raise RetryableDocumentIngestionError("temporary")

    monkeypatch.setattr(task_module.ingestion_service, "ingest_document", transient)
    monkeypatch.setattr(
        task_module.ingestion_service,
        "prepare_document_ingestion_retry",
        lambda _document_id: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(task_module, "ingestion_retry_countdown", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        task_module.ingest_document_task,
        "retry",
        lambda **_kwargs: (_ for _ in ()).throw(RetryRequested()),
    )

    with pytest.raises(RetryRequested):
        _run_with_retry_count(1, str(uuid4()))

    assert "error_type=RuntimeError" in caplog.text
    assert "database unavailable" not in caplog.text


@pytest.mark.parametrize("terminal_status", [DOCUMENT_STATUS_READY, DOCUMENT_STATUS_FAILED])
def test_terminal_race_during_retry_state_update_does_not_schedule_retry(
    monkeypatch,
    terminal_status,
) -> None:
    document_id = str(uuid4())
    monkeypatch.setattr(
        task_module.ingestion_service,
        "ingest_document",
        lambda _document_id: (_ for _ in ()).throw(
            RetryableDocumentIngestionError("losing duplicate")
        ),
    )
    monkeypatch.setattr(
        task_module.ingestion_service,
        "prepare_document_ingestion_retry",
        lambda selected: DocumentIngestionResult(selected, terminal_status, 0),
    )
    monkeypatch.setattr(
        task_module.ingest_document_task,
        "retry",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not retry")),
    )

    _run_with_retry_count(1, document_id)


def test_fourth_total_attempt_explicitly_finalizes_without_calling_retry(monkeypatch) -> None:
    document_id = str(uuid4())
    finalized: list[str] = []
    monkeypatch.setattr(
        task_module.ingestion_service,
        "ingest_document",
        lambda _document_id: (_ for _ in ()).throw(
            RetryableDocumentIngestionError("still unavailable")
        ),
    )
    monkeypatch.setattr(
        task_module.ingestion_service,
        "finalize_document_ingestion_retry_exhaustion",
        lambda selected: finalized.append(selected),
    )
    monkeypatch.setattr(
        task_module.ingest_document_task,
        "retry",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("budget is exhausted")),
    )

    _run_with_retry_count(3, document_id)

    assert finalized == [document_id]


def test_terminal_state_write_failure_remains_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        task_module.ingestion_service,
        "ingest_document",
        lambda _document_id: (_ for _ in ()).throw(
            RetryableDocumentIngestionError("still unavailable")
        ),
    )
    monkeypatch.setattr(
        task_module.ingestion_service,
        "finalize_document_ingestion_retry_exhaustion",
        lambda _document_id: (_ for _ in ()).throw(RuntimeError("terminal DB write failed")),
    )

    with pytest.raises(RuntimeError, match="terminal DB write failed"):
        _run_with_retry_count(3, str(uuid4()))
