"""Ownership-safe persistence helpers for source-document object metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select

from backend.app.storage.base import document_pdf_object_key, validate_object_key
from persistence.ownership import require_owned_document


class DocumentObjectMetadataError(ValueError):
    """Raised when document object metadata is invalid or internally inconsistent."""


@dataclass(frozen=True)
class StoredDocumentObjectMetadata:
    document_id: str
    user_id: str
    object_key: str
    content_type: str
    byte_size: int


def _validate_metadata(
    *,
    document_id: str,
    object_key: str,
    content_type: str,
    byte_size: int,
) -> tuple[str, str, int]:
    validated_key = validate_object_key(object_key)
    expected_key = document_pdf_object_key(document_id)
    if validated_key != expected_key:
        raise DocumentObjectMetadataError(
            "Document object key does not match the document's generated storage key."
        )

    normalized_content_type = content_type.strip() if isinstance(content_type, str) else ""
    if not normalized_content_type or len(normalized_content_type) > 255:
        raise DocumentObjectMetadataError("Document content type is invalid.")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise DocumentObjectMetadataError("Document byte size must be a positive integer.")
    return validated_key, normalized_content_type, byte_size


def document_object_metadata_from_row(document: object) -> Optional[StoredDocumentObjectMetadata]:
    """Validate and project object metadata from an already-loaded document row."""
    values = (
        getattr(document, "object_key", None),
        getattr(document, "content_type", None),
        getattr(document, "byte_size", None),
    )
    if values == (None, None, None):
        return None
    if any(value is None for value in values):
        raise DocumentObjectMetadataError("Document object metadata is incomplete.")

    object_key, content_type, byte_size = _validate_metadata(
        document_id=str(getattr(document, "id")),
        object_key=str(values[0]),
        content_type=str(values[1]),
        byte_size=values[2],
    )
    return StoredDocumentObjectMetadata(
        document_id=str(getattr(document, "id")),
        user_id=str(getattr(document, "user_id")),
        object_key=object_key,
        content_type=content_type,
        byte_size=byte_size,
    )


def attach_owned_document_object_metadata(
    *,
    user_id: str,
    document_id: str,
    object_key: str,
    content_type: str,
    byte_size: int,
) -> StoredDocumentObjectMetadata:
    """Attach generated object metadata after enforcing document ownership."""
    from persistence.db import SessionLocal

    with SessionLocal() as session:
        try:
            document = require_owned_document(
                session,
                user_id=user_id,
                document_id=document_id,
            )
            validated_key, normalized_content_type, validated_byte_size = _validate_metadata(
                document_id=str(document.id),
                object_key=object_key,
                content_type=content_type,
                byte_size=byte_size,
            )
            document.object_key = validated_key
            document.content_type = normalized_content_type
            document.byte_size = validated_byte_size
            session.commit()
            return StoredDocumentObjectMetadata(
                document_id=str(document.id),
                user_id=str(document.user_id),
                object_key=validated_key,
                content_type=normalized_content_type,
                byte_size=validated_byte_size,
            )
        except Exception:
            session.rollback()
            raise


def get_owned_document_object_metadata(
    *,
    user_id: str,
    document_id: str,
) -> Optional[StoredDocumentObjectMetadata]:
    """Load metadata only after resolving a document in the user's namespace."""
    from persistence.db import SessionLocal

    with SessionLocal() as session:
        document = require_owned_document(
            session,
            user_id=user_id,
            document_id=document_id,
        )
        return document_object_metadata_from_row(document)


def get_document_object_metadata(
    *,
    document_id: str,
) -> Optional[StoredDocumentObjectMetadata]:
    """Internal source-metadata lookup by document ID, without an HTTP projection."""
    from persistence.db import SessionLocal
    from persistence.models import Document

    if not isinstance(document_id, str) or not document_id.strip():
        raise DocumentObjectMetadataError("Document ID must not be empty.")

    with SessionLocal() as session:
        document = session.execute(
            select(Document).where(Document.id == document_id)
        ).scalar_one_or_none()
        if document is None:
            return None
        return document_object_metadata_from_row(document)
