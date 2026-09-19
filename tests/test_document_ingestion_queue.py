from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.app.services import document_ingestion_queue


def test_enqueue_helper_publishes_only_the_document_id(monkeypatch) -> None:
    document_id = str(uuid4())
    calls: list[tuple[tuple, dict]] = []

    def send_task(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(id="celery-task-id")

    monkeypatch.setattr(document_ingestion_queue.celery_app, "send_task", send_task)

    task_id = document_ingestion_queue.enqueue_document_ingestion(document_id)

    assert task_id == "celery-task-id"
    assert calls == [
        (
            (document_ingestion_queue.DOCUMENT_INGESTION_TASK_NAME,),
            {
                "args": [document_id],
                "kwargs": {},
                "queue": document_ingestion_queue.get_celery_settings().ingestion_queue,
            },
        )
    ]


def test_enqueue_helper_rejects_non_uuid_without_publishing(monkeypatch) -> None:
    published = False

    def send_task(*_args, **_kwargs):
        nonlocal published
        published = True

    monkeypatch.setattr(document_ingestion_queue.celery_app, "send_task", send_task)

    with pytest.raises(ValueError):
        document_ingestion_queue.enqueue_document_ingestion("not-a-document-id")

    assert published is False


def test_enqueue_helper_does_not_hide_publish_failures(monkeypatch) -> None:
    def fail_publish(*_args, **_kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(document_ingestion_queue.celery_app, "send_task", fail_publish)

    with pytest.raises(RuntimeError, match="broker unavailable"):
        document_ingestion_queue.enqueue_document_ingestion(str(uuid4()))
