"""Environment-backed construction for object-storage implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from backend.app.storage.base import ObjectStorage, ObjectStorageConfigurationError
from backend.app.storage.local import LocalObjectStorage
from backend.app.storage.s3 import S3ObjectStorage


SUPPORTED_BACKENDS = {"local", "s3"}
SUPPORTED_ADDRESSING_STYLES = {"auto", "path", "virtual"}


def _optional_environment_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True)
class ObjectStorageSettings:
    backend: str
    local_root: str
    s3_endpoint_url: str | None
    s3_bucket: str | None
    s3_region: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_session_token: str | None
    s3_addressing_style: str

    @classmethod
    def from_environment(cls) -> "ObjectStorageSettings":
        backend = (os.getenv("OBJECT_STORAGE_BACKEND") or "local").strip().lower()
        local_root = (os.getenv("OBJECT_STORAGE_LOCAL_ROOT") or "./uploads/object-storage").strip()
        addressing_style = (
            os.getenv("OBJECT_STORAGE_S3_ADDRESSING_STYLE") or "auto"
        ).strip().lower()
        settings = cls(
            backend=backend,
            local_root=local_root,
            s3_endpoint_url=_optional_environment_value("OBJECT_STORAGE_S3_ENDPOINT_URL"),
            s3_bucket=_optional_environment_value("OBJECT_STORAGE_S3_BUCKET"),
            s3_region=_optional_environment_value("OBJECT_STORAGE_S3_REGION"),
            s3_access_key_id=_optional_environment_value("OBJECT_STORAGE_S3_ACCESS_KEY_ID"),
            s3_secret_access_key=_optional_environment_value("OBJECT_STORAGE_S3_SECRET_ACCESS_KEY"),
            s3_session_token=_optional_environment_value("OBJECT_STORAGE_S3_SESSION_TOKEN"),
            s3_addressing_style=addressing_style,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ObjectStorageConfigurationError(
                f"Unsupported OBJECT_STORAGE_BACKEND: {self.backend!r}."
            )
        if self.backend == "local" and not self.local_root:
            raise ObjectStorageConfigurationError("OBJECT_STORAGE_LOCAL_ROOT must not be empty.")
        if self.s3_addressing_style not in SUPPORTED_ADDRESSING_STYLES:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_S3_ADDRESSING_STYLE must be auto, path, or virtual."
            )
        if self.backend != "s3":
            return
        if not self.s3_bucket:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_S3_BUCKET is required for the s3 backend."
            )
        if bool(self.s3_access_key_id) != bool(self.s3_secret_access_key):
            raise ObjectStorageConfigurationError(
                "Explicit S3 access key ID and secret access key must be configured together."
            )
        if self.s3_session_token and not self.s3_access_key_id:
            raise ObjectStorageConfigurationError(
                "An explicit S3 session token requires explicit access credentials."
            )


def _build_s3_client(
    settings: ObjectStorageSettings,
    *,
    boto3_module: Any = None,
    config_factory: Callable[..., object] | None = None,
) -> object:
    if boto3_module is None or config_factory is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ObjectStorageConfigurationError(
                "The s3 storage backend requires the boto3 package."
            ) from exc
        boto3_module = boto3
        config_factory = Config

    client_arguments: dict[str, object] = {
        "config": config_factory(s3={"addressing_style": settings.s3_addressing_style}),
    }
    if settings.s3_endpoint_url:
        client_arguments["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_region:
        client_arguments["region_name"] = settings.s3_region

    # Omitting these arguments intentionally delegates credentials to boto3's
    # standard provider chain (IAM roles, workload identity, shared config, etc.).
    if settings.s3_access_key_id and settings.s3_secret_access_key:
        client_arguments["aws_access_key_id"] = settings.s3_access_key_id
        client_arguments["aws_secret_access_key"] = settings.s3_secret_access_key
        if settings.s3_session_token:
            client_arguments["aws_session_token"] = settings.s3_session_token

    return boto3_module.client("s3", **client_arguments)


@lru_cache(maxsize=1)
def get_object_storage() -> ObjectStorage:
    """Build the configured backend lazily and cache it per process."""
    settings = ObjectStorageSettings.from_environment()
    if settings.backend == "local":
        return LocalObjectStorage(settings.local_root)
    return S3ObjectStorage(
        bucket=settings.s3_bucket or "",
        client=_build_s3_client(settings),
    )
