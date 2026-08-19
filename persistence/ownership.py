"""Shared ownership guards for user-scoped persistence operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session


class PersistenceUserNotFoundError(LookupError):
    """Raised when a persistence caller supplies no existing user."""


class OwnedResourceNotFoundError(LookupError):
    """Raised for both missing and differently-owned resources."""


def require_existing_user(session: Session, user_id: str) -> str:
    """Return an existing explicit user ID without inventing an account."""
    from persistence.models import User

    if not isinstance(user_id, str) or not user_id.strip():
        raise PersistenceUserNotFoundError("Persistence requires an existing user.")
    existing_id = session.execute(
        select(User.id).where(User.id == user_id)
    ).scalar_one_or_none()
    if existing_id is None:
        raise PersistenceUserNotFoundError("Persistence requires an existing user.")
    return str(existing_id)


def require_owned_document(session: Session, *, user_id: str, document_id: str):
    """Resolve a document with ownership enforced in the database query."""
    from persistence.models import Document

    document = session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
    ).scalar_one_or_none()
    if document is None:
        raise OwnedResourceNotFoundError("Document not found.")
    return document
