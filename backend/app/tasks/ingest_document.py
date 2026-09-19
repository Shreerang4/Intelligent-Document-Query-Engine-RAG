"""Thin Celery delivery wrapper for queue-independent document ingestion."""

from __future__ import annotations

from celery.utils.log import get_task_logger

from backend.app.celery_app import DOCUMENT_INGESTION_TASK_NAME, celery_app
from backend.app.celery_config import get_celery_settings, ingestion_retry_countdown
from backend.app.services import document_ingestion as ingestion_service
from persistence.document_ingestion import DOCUMENT_STATUS_FAILED, DOCUMENT_STATUS_READY


logger = get_task_logger(__name__)
settings = get_celery_settings()


@celery_app.task(
    bind=True,
    name=DOCUMENT_INGESTION_TASK_NAME,
    ignore_result=True,
    max_retries=settings.ingestion_max_retries,
)
def ingest_document_task(self, document_id: str) -> None:
    """Deliver one opaque document identifier to the ingestion service."""
    try:
        result = ingestion_service.ingest_document(document_id)
    except ingestion_service.DocumentIngestionNotFoundError:
        logger.warning("Ignoring ingestion task for a document that no longer exists.")
        return None
    except ingestion_service.RetryableDocumentIngestionError as exc:
        current_retry = int(self.request.retries or 0)
        if current_retry >= settings.ingestion_max_retries:
            # This is intentionally not best-effort. If the terminal database
            # update fails, the task must remain visibly failed.
            ingestion_service.finalize_document_ingestion_retry_exhaustion(document_id)
            return None

        should_retry = True
        try:
            retry_state = ingestion_service.prepare_document_ingestion_retry(document_id)
            if retry_state.status in {DOCUMENT_STATUS_READY, DOCUMENT_STATUS_FAILED}:
                should_retry = False
        except ingestion_service.DocumentIngestionNotFoundError:
            logger.warning("Document disappeared before retry state could be recorded.")
            should_retry = False
        except Exception as state_exc:
            # A transient state-write failure must not suppress the broker retry.
            # Log only the exception type, not DB/storage details or credentials.
            logger.warning(
                "Could not persist queued retry state; scheduling retry anyway "
                "(error_type=%s).",
                type(state_exc).__name__,
            )

        if not should_retry:
            return None

        countdown = ingestion_retry_countdown(current_retry, settings=settings)
        raise self.retry(
            exc=exc,
            countdown=countdown,
            max_retries=settings.ingestion_max_retries,
        )

    return None


__all__ = ["ingest_document_task"]
