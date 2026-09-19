"""Celery application for queue-delivered background work.

This module deliberately has no dependency on FastAPI or ``main.py``.
"""

from celery import Celery

from backend.app.celery_config import get_celery_settings


DOCUMENT_INGESTION_TASK_NAME = "documents.ingest"

settings = get_celery_settings()
celery_app = Celery(
    "intelligent_document_query_engine",
    broker=settings.broker_url,
    backend=None,
    include=("backend.app.tasks.ingest_document",),
)
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_store_errors_even_if_ignored=False,
    result_backend=None,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"confirm_publish": True},
    task_default_queue=settings.ingestion_queue,
    task_routes={
        DOCUMENT_INGESTION_TASK_NAME: {"queue": settings.ingestion_queue},
    },
)

# Celery's ``-A backend.app.celery_app`` discovery looks for ``app``.
app = celery_app

__all__ = ["DOCUMENT_INGESTION_TASK_NAME", "app", "celery_app"]
