"""Transactional lifecycle operations for queue-independent document ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sqlalchemy import delete, func, select

from persistence.document_artifacts import ArtifactValidationError, serialize_embedding
from persistence.document_objects import (
    DocumentObjectMetadataError,
    StoredDocumentObjectMetadata,
    document_object_metadata_from_row,
)


DOCUMENT_STATUS_QUEUED = "queued"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_READY = "ready"
DOCUMENT_STATUS_FAILED = "failed"
LEGACY_DOCUMENT_STATUS_INGESTED = "ingested"


class DocumentIngestionNotFoundError(LookupError):
    """Raised when an internal ingestion invocation references no document."""


class DocumentIngestionStateError(RuntimeError):
    """Raised when a document cannot enter or complete ingestion."""


@dataclass(frozen=True)
class DocumentIngestionStart:
    document_id: str
    status: str
    object_metadata: Optional[StoredDocumentObjectMetadata]


@dataclass(frozen=True)
class DocumentIngestionWriteResult:
    document_id: str
    status: str
    chunk_count: int


def _locked_document(session: Any, document_id: str) -> Any:
    from persistence.models import Document

    document = session.execute(
        select(Document).where(Document.id == document_id).with_for_update()
    ).scalar_one_or_none()
    if document is None:
        raise DocumentIngestionNotFoundError("Document not found for ingestion.")
    return document


def start_document_ingestion(*, document_id: str) -> DocumentIngestionStart:
    """Move a queued document to processing and return its private object metadata."""
    from persistence.db import SessionLocal

    with SessionLocal() as session:
        try:
            document = _locked_document(session, document_id)
            if document.status in {DOCUMENT_STATUS_READY, LEGACY_DOCUMENT_STATUS_INGESTED}:
                return DocumentIngestionStart(
                    document_id=str(document.id),
                    status=DOCUMENT_STATUS_READY,
                    object_metadata=None,
                )
            if document.status == DOCUMENT_STATUS_FAILED:
                return DocumentIngestionStart(
                    document_id=str(document.id),
                    status=DOCUMENT_STATUS_FAILED,
                    object_metadata=None,
                )
            if document.status not in {DOCUMENT_STATUS_QUEUED, DOCUMENT_STATUS_PROCESSING}:
                raise DocumentIngestionStateError(
                    f"Document has unsupported ingestion status {document.status!r}."
                )

            try:
                metadata = document_object_metadata_from_row(document)
            except DocumentObjectMetadataError as exc:
                raise DocumentIngestionStateError("Document object metadata is invalid.") from exc
            if metadata is None:
                raise DocumentIngestionStateError("Document has no source object metadata.")

            document.status = DOCUMENT_STATUS_PROCESSING
            document.error_message = None
            session.commit()
            return DocumentIngestionStart(
                document_id=str(document.id),
                status=DOCUMENT_STATUS_PROCESSING,
                object_metadata=metadata,
            )
        except Exception:
            session.rollback()
            raise


def mark_document_ingestion_failed(
    *,
    document_id: str,
    error_message: str,
) -> DocumentIngestionWriteResult:
    """Record a failure unless another invocation has already made the document ready."""
    from persistence.db import SessionLocal
    from persistence.models import Chunk

    safe_message = " ".join(str(error_message).split()).strip()
    if not safe_message:
        safe_message = "Document ingestion failed."
    safe_message = safe_message[:2000]

    with SessionLocal() as session:
        try:
            document = _locked_document(session, document_id)
            if document.status in {DOCUMENT_STATUS_READY, LEGACY_DOCUMENT_STATUS_INGESTED}:
                chunk_count = int(
                    session.scalar(
                        select(func.count()).select_from(Chunk).where(
                            Chunk.user_id == document.user_id,
                            Chunk.document_id == document.id,
                        )
                    )
                    or 0
                )
                return DocumentIngestionWriteResult(
                    document_id=str(document.id),
                    status=DOCUMENT_STATUS_READY,
                    chunk_count=chunk_count,
                )

            document.status = DOCUMENT_STATUS_FAILED
            document.error_message = safe_message
            session.commit()
            return DocumentIngestionWriteResult(
                document_id=str(document.id),
                status=DOCUMENT_STATUS_FAILED,
                chunk_count=0,
            )
        except Exception:
            session.rollback()
            raise


def persist_document_ingestion_atomic(
    *,
    document_id: str,
    chunks: Sequence[Mapping[str, Any]],
    embedding_matrix: np.ndarray,
    embedding_model: str,
    embedding_format: str,
) -> DocumentIngestionWriteResult:
    """Atomically replace document artifacts and make ready the final committed state."""
    matrix = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
        raise ArtifactValidationError("Embedding matrix shape does not match the chunk sequence.")
    if not chunks:
        raise ArtifactValidationError("A ready document requires at least one chunk.")

    from persistence.db import SessionLocal
    from persistence.models import Chunk

    with SessionLocal() as session:
        try:
            document = _locked_document(session, document_id)
            if document.status in {DOCUMENT_STATUS_READY, LEGACY_DOCUMENT_STATUS_INGESTED}:
                chunk_count = int(
                    session.scalar(
                        select(func.count()).select_from(Chunk).where(
                            Chunk.user_id == document.user_id,
                            Chunk.document_id == document.id,
                        )
                    )
                    or 0
                )
                return DocumentIngestionWriteResult(
                    document_id=str(document.id),
                    status=DOCUMENT_STATUS_READY,
                    chunk_count=chunk_count,
                )

            session.execute(
                delete(Chunk).where(
                    Chunk.user_id == document.user_id,
                    Chunk.document_id == document.id,
                )
            )
            chunk_rows = []
            for chunk_index, (chunk, vector) in enumerate(zip(chunks, matrix)):
                chunk_text = str(chunk["text"])
                blob, dimension, dtype = serialize_embedding(vector)
                chunk_rows.append(
                    Chunk(
                        user_id=document.user_id,
                        document_id=document.id,
                        chunk_id=int(chunk["chunk_id"]),
                        chunk_index=chunk_index,
                        page_number=int(chunk["page"]),
                        text=chunk_text,
                        text_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        char_count=len(chunk_text),
                        embedding_blob=blob,
                        embedding_dimension=dimension,
                        embedding_dtype=dtype,
                    )
                )
            session.add_all(chunk_rows)
            document.embedding_model = embedding_model
            document.embedding_format = embedding_format
            document.status = DOCUMENT_STATUS_READY
            document.error_message = None
            session.commit()
            return DocumentIngestionWriteResult(
                document_id=str(document.id),
                status=DOCUMENT_STATUS_READY,
                chunk_count=len(chunk_rows),
            )
        except Exception:
            session.rollback()
            raise
