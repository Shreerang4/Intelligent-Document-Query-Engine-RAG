"""Owned persistence operations for queued asynchronous document uploads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import case, select
from sqlalchemy.exc import IntegrityError

from backend.app.storage.base import document_pdf_object_key, validate_object_key
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
)
from persistence.ownership import require_existing_user, require_owned_document


DOCUMENT_LIFECYCLE_STATUSES = {
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_FAILED,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UploadRequestConflictError(ValueError):
    """Raised when an owned upload request UUID is reused for different bytes."""


class DocumentUploadStateError(RuntimeError):
    """Raised when persisted document state cannot be safely exposed or recovered."""


@dataclass(frozen=True)
class StoredDocumentStatus:
    document_id: str
    user_id: str
    filename: Optional[str]
    source_hash: Optional[str]
    status: str
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class QueuedDocumentWriteResult:
    document: StoredDocumentStatus
    created: bool


def _safe_error_message(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized[:2000] or None


def _project_document(document: object) -> StoredDocumentStatus:
    status = str(getattr(document, "status"))
    if status not in DOCUMENT_LIFECYCLE_STATUSES:
        raise DocumentUploadStateError("Document has an unsupported lifecycle status.")
    raw_source_hash = getattr(document, "source_hash", None)
    source_hash = str(raw_source_hash) if raw_source_hash is not None else None
    return StoredDocumentStatus(
        document_id=str(getattr(document, "id")),
        user_id=str(getattr(document, "user_id")),
        filename=getattr(document, "filename", None),
        source_hash=source_hash,
        status=status,
        error_message=_safe_error_message(getattr(document, "error_message", None)),
        created_at=getattr(document, "created_at"),
        updated_at=getattr(document, "updated_at"),
    )


def _matching_upload_request(
    session: object,
    *,
    user_id: str,
    upload_request_id: str,
):
    from persistence.models import Document

    return session.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.upload_request_id == upload_request_id,
        )
    ).scalar_one_or_none()


def recover_owned_upload_request(
    *,
    user_id: str,
    upload_request_id: str,
    source_hash: str,
) -> Optional[StoredDocumentStatus]:
    """Recover an owned upload before any new object is written."""
    from persistence.db import SessionLocal

    with SessionLocal() as session:
        document = _matching_upload_request(
            session,
            user_id=user_id,
            upload_request_id=upload_request_id,
        )
        if document is None:
            return None
        if document.source_hash is None:
            raise DocumentUploadStateError("Document source hash is unavailable.")
        if document.source_hash != source_hash:
            raise UploadRequestConflictError(
                "upload_request_id was already used with a different document."
            )
        return _project_document(document)


def find_owned_usable_duplicate(*, user_id: str, source_hash: str) -> Optional[StoredDocumentStatus]:
    """Prefer the newest active match, then the newest ready match for this owner."""
    from persistence.db import SessionLocal
    from persistence.models import Document

    active_statuses = (DOCUMENT_STATUS_QUEUED, DOCUMENT_STATUS_PROCESSING)
    with SessionLocal() as session:
        document = session.execute(
            select(Document)
            .where(
                Document.user_id == user_id,
                Document.source_hash == source_hash,
                Document.status.in_((*active_statuses, DOCUMENT_STATUS_READY)),
            )
            .order_by(
                case((Document.status.in_(active_statuses), 0), else_=1),
                Document.created_at.desc(),
                Document.id.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        return _project_document(document) if document is not None else None


def create_or_recover_queued_upload(
    *,
    user_id: str,
    document_id: str,
    upload_request_id: Optional[str],
    filename: str,
    source_hash: str,
    object_key: str,
    content_type: str,
    byte_size: int,
    embedding_model: str,
    embedding_format: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
) -> QueuedDocumentWriteResult:
    """Create one queued document or recover the winner of an idempotency race."""
    from persistence.db import SessionLocal
    from persistence.models import Document

    validated_key = validate_object_key(object_key)
    if validated_key != document_pdf_object_key(document_id):
        raise ValueError("Document object key does not match the generated document ID.")
    if _SHA256_PATTERN.fullmatch(source_hash) is None:
        raise ValueError("source_hash must be a lowercase SHA-256 digest.")
    if byte_size <= 0:
        raise ValueError("byte_size must be positive.")

    with SessionLocal() as session:
        try:
            require_existing_user(session, user_id)
            document = Document(
                id=document_id,
                user_id=user_id,
                source_type="upload",
                filename=filename,
                source_hash=source_hash,
                cache_key=f"async-upload:{document_id}",
                object_key=validated_key,
                content_type=content_type,
                byte_size=byte_size,
                upload_request_id=upload_request_id,
                status=DOCUMENT_STATUS_QUEUED,
                error_message=None,
                embedding_model=embedding_model,
                embedding_format=embedding_format,
                retrieval_mode=retrieval_mode,
                reranker_model=reranker_model,
                k_initial=k_initial,
                k_final=k_final,
            )
            session.add(document)
            session.flush()
            session.refresh(document)
            projected = _project_document(document)
            session.commit()
            return QueuedDocumentWriteResult(document=projected, created=True)
        except IntegrityError:
            session.rollback()
            if upload_request_id is None:
                raise
            winner = _matching_upload_request(
                session,
                user_id=user_id,
                upload_request_id=upload_request_id,
            )
            if winner is None:
                raise
            if winner.source_hash != source_hash:
                raise UploadRequestConflictError(
                    "upload_request_id was already used with a different document."
                )
            return QueuedDocumentWriteResult(
                document=_project_document(winner),
                created=False,
            )
        except Exception:
            session.rollback()
            raise


def get_owned_document_status(*, user_id: str, document_id: str) -> StoredDocumentStatus:
    """Return one document only after ownership is enforced by the query."""
    from persistence.db import SessionLocal

    with SessionLocal() as session:
        document = require_owned_document(
            session,
            user_id=user_id,
            document_id=document_id,
        )
        return _project_document(document)


def should_enqueue_owned_document(*, user_id: str, document_id: str) -> StoredDocumentStatus:
    """Lock and return current state before an idempotent re-publication decision."""
    from persistence.db import SessionLocal
    from persistence.models import Document

    with SessionLocal() as session:
        document = session.execute(
            select(Document)
            .where(Document.id == document_id, Document.user_id == user_id)
            .with_for_update()
        ).scalar_one_or_none()
        if document is None:
            from persistence.ownership import OwnedResourceNotFoundError

            raise OwnedResourceNotFoundError("Document not found.")
        return _project_document(document)
