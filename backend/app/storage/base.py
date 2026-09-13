"""Common object-storage contracts and object-key validation."""

from __future__ import annotations

import re
import uuid
from typing import Protocol, runtime_checkable


MAX_OBJECT_KEY_LENGTH = 512
_KEY_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ObjectStorageError(RuntimeError):
    """Base error for object-storage operations."""


class ObjectNotFoundError(ObjectStorageError):
    """Raised when an object key does not exist."""


class ObjectStorageConfigurationError(ObjectStorageError):
    """Raised when object-storage configuration is incomplete or invalid."""


class InvalidObjectKeyError(ValueError):
    """Raised when an object key is unsafe or outside the supported format."""


def validate_object_key(object_key: str) -> str:
    """Validate the portable, relative key format accepted by every backend."""
    if not isinstance(object_key, str) or not object_key:
        raise InvalidObjectKeyError("Object key must be a non-empty string.")
    if len(object_key) > MAX_OBJECT_KEY_LENGTH:
        raise InvalidObjectKeyError("Object key is too long.")
    if object_key.startswith("/") or object_key.endswith("/"):
        raise InvalidObjectKeyError("Object key must be a relative object path.")
    if "\\" in object_key or "\x00" in object_key:
        raise InvalidObjectKeyError("Object key contains an unsupported character.")

    parts = object_key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InvalidObjectKeyError("Object key contains an unsafe path segment.")
    if any(_KEY_SEGMENT_PATTERN.fullmatch(part) is None for part in parts):
        raise InvalidObjectKeyError("Object key contains an unsupported path segment.")
    return object_key


def document_pdf_object_key(document_id: str) -> str:
    """Build the sole object-key format used for persisted source PDFs."""
    try:
        normalized_document_id = str(uuid.UUID(str(document_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise InvalidObjectKeyError("Document ID must be a valid UUID.") from exc
    return validate_object_key(f"documents/{normalized_document_id}/source.pdf")


@runtime_checkable
class ObjectStorage(Protocol):
    """Minimal storage API used by document ingestion and future viewing."""

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        """Create or replace a private object."""

    def get(self, object_key: str) -> bytes:
        """Return an object's bytes or raise ObjectNotFoundError."""

    def delete(self, object_key: str) -> None:
        """Delete an object. Missing objects are treated as already deleted."""
