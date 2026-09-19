"""Application-level service entrypoints."""

from backend.app.services.document_ingestion import (
    DocumentIngestionResult,
    RETRY_EXHAUSTED_FAILURE_MESSAGE,
    RetryableDocumentIngestionError,
    finalize_document_ingestion_retry_exhaustion,
    ingest_document,
    prepare_document_ingestion_retry,
)

__all__ = [
    "DocumentIngestionResult",
    "RETRY_EXHAUSTED_FAILURE_MESSAGE",
    "RetryableDocumentIngestionError",
    "finalize_document_ingestion_retry_exhaustion",
    "ingest_document",
    "prepare_document_ingestion_retry",
]
