"""Application-level service entrypoints."""

from backend.app.services.document_ingestion import (
    DocumentIngestionResult,
    RetryableDocumentIngestionError,
    ingest_document,
)

__all__ = [
    "DocumentIngestionResult",
    "RetryableDocumentIngestionError",
    "ingest_document",
]
