"""Queue-independent source-document ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from backend.app.rag.config import get_embedding_model_name
from backend.app.rag.embeddings import (
    create_chunk_embeddings,
    get_embedding_input_format_version,
    get_embedding_model,
)
from backend.app.rag.ingestion import ChunkRecord, PdfContentError, parse_and_chunk_pdf_bytes
from backend.app.storage import ObjectStorage, ObjectStorageError, get_object_storage
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DocumentIngestionNotFoundError,
    DocumentIngestionStateError,
    mark_document_ingestion_failed,
    persist_document_ingestion_atomic,
    start_document_ingestion,
)


class RetryableDocumentIngestionError(RuntimeError):
    """Raised for infrastructure/model failures that a future task may retry."""


@dataclass(frozen=True)
class DocumentIngestionResult:
    document_id: str
    status: str
    chunk_count: int


def _record_permanent_failure(document_id: str, message: str) -> DocumentIngestionResult:
    try:
        write_result = mark_document_ingestion_failed(
            document_id=document_id,
            error_message=message,
        )
    except Exception as exc:
        raise RetryableDocumentIngestionError(
            "Failed to record the permanent ingestion failure."
        ) from exc
    return DocumentIngestionResult(
        document_id=write_result.document_id,
        status=write_result.status,
        chunk_count=write_result.chunk_count,
    )


def ingest_document(
    document_id: str,
    *,
    object_storage: Optional[ObjectStorage] = None,
    embedding_model: Any = None,
    pdf_parser: Optional[Callable[[bytes], list[ChunkRecord]]] = None,
    embedding_builder: Optional[Callable[[list[ChunkRecord], Any], np.ndarray]] = None,
) -> DocumentIngestionResult:
    """Ingest one stored PDF; suitable for direct use by a future Celery task."""
    try:
        start = start_document_ingestion(document_id=document_id)
    except DocumentIngestionNotFoundError:
        raise
    except DocumentIngestionStateError as exc:
        return _record_permanent_failure(document_id, str(exc))
    except Exception as exc:
        raise RetryableDocumentIngestionError("Failed to start document ingestion.") from exc

    if start.status == DOCUMENT_STATUS_READY:
        return DocumentIngestionResult(document_id=start.document_id, status=start.status, chunk_count=0)
    if start.status == DOCUMENT_STATUS_FAILED:
        return DocumentIngestionResult(document_id=start.document_id, status=start.status, chunk_count=0)
    if start.status != DOCUMENT_STATUS_PROCESSING or start.object_metadata is None:
        return _record_permanent_failure(document_id, "Document ingestion state is invalid.")

    try:
        storage = object_storage if object_storage is not None else get_object_storage()
        pdf_bytes = storage.get(start.object_metadata.object_key)
    except ObjectStorageError as exc:
        raise RetryableDocumentIngestionError("Failed to load the source document.") from exc
    except Exception as exc:
        raise RetryableDocumentIngestionError("Unexpected source storage failure.") from exc

    if len(pdf_bytes) != start.object_metadata.byte_size:
        return _record_permanent_failure(
            document_id,
            "Stored source size does not match document metadata.",
        )

    parser = pdf_parser or parse_and_chunk_pdf_bytes
    try:
        chunks = parser(pdf_bytes)
    except PdfContentError as exc:
        return _record_permanent_failure(document_id, str(exc))
    except Exception as exc:
        raise RetryableDocumentIngestionError("Unexpected PDF processing failure.") from exc

    selected_embedding_model = embedding_model
    if selected_embedding_model is None:
        try:
            selected_embedding_model = get_embedding_model()
        except Exception as exc:
            raise RetryableDocumentIngestionError("Failed to load the embedding model.") from exc

    builder = embedding_builder or create_chunk_embeddings
    try:
        embedding_matrix = builder(chunks, selected_embedding_model)
    except Exception as exc:
        raise RetryableDocumentIngestionError("Failed to embed document chunks.") from exc

    model_name = get_embedding_model_name()
    try:
        write_result = persist_document_ingestion_atomic(
            document_id=document_id,
            chunks=chunks,
            embedding_matrix=embedding_matrix,
            embedding_model=model_name,
            embedding_format=get_embedding_input_format_version(model_name),
        )
    except Exception as exc:
        raise RetryableDocumentIngestionError("Failed to persist document artifacts.") from exc

    return DocumentIngestionResult(
        document_id=write_result.document_id,
        status=write_result.status,
        chunk_count=write_result.chunk_count,
    )
