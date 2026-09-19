"""Application-side publishing boundary for document ingestion."""

from __future__ import annotations

from uuid import UUID

from backend.app.celery_app import DOCUMENT_INGESTION_TASK_NAME, celery_app
from backend.app.celery_config import get_celery_settings


def enqueue_document_ingestion(document_id: str) -> str:
    """Publish exactly one document ID and surface normal broker failures."""
    normalized_document_id = str(UUID(str(document_id)))
    result = celery_app.send_task(
        DOCUMENT_INGESTION_TASK_NAME,
        args=[normalized_document_id],
        kwargs={},
        queue=get_celery_settings().ingestion_queue,
    )
    return str(result.id)


__all__ = ["enqueue_document_ingestion"]
