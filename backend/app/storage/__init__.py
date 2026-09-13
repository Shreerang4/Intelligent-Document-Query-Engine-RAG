"""Private object-storage interfaces and implementations."""

from backend.app.storage.base import (
    InvalidObjectKeyError,
    ObjectNotFoundError,
    ObjectStorage,
    ObjectStorageConfigurationError,
    ObjectStorageError,
    document_pdf_object_key,
    validate_object_key,
)
from backend.app.storage.config import ObjectStorageSettings, get_object_storage
from backend.app.storage.local import LocalObjectStorage
from backend.app.storage.s3 import S3ObjectStorage

__all__ = [
    "InvalidObjectKeyError",
    "LocalObjectStorage",
    "ObjectNotFoundError",
    "ObjectStorage",
    "ObjectStorageConfigurationError",
    "ObjectStorageError",
    "ObjectStorageSettings",
    "S3ObjectStorage",
    "document_pdf_object_key",
    "get_object_storage",
    "validate_object_key",
]
