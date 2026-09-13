"""Private S3-compatible object storage."""

from __future__ import annotations

from typing import Any

from backend.app.storage.base import ObjectNotFoundError, ObjectStorageError, validate_object_key


_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def _is_not_found_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return str(error.get("Code", "")) in _NOT_FOUND_CODES


class S3ObjectStorage:
    """Store private objects in an S3 bucket without constructing public URLs."""

    def __init__(self, *, bucket: str, client: Any) -> None:
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("S3 bucket must not be empty.")
        if client is None:
            raise ValueError("S3 client must not be None.")
        self.bucket = bucket.strip()
        self._client = client

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        validated_key = validate_object_key(object_key)
        if not isinstance(data, bytes):
            raise TypeError("Object data must be bytes.")
        normalized_content_type = content_type.strip() if isinstance(content_type, str) else ""
        if not normalized_content_type:
            raise ValueError("Object content type must not be empty.")

        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=validated_key,
                Body=data,
                ContentType=normalized_content_type,
            )
        except Exception as exc:
            raise ObjectStorageError("Failed to store S3 object.") from exc

    def get(self, object_key: str) -> bytes:
        validated_key = validate_object_key(object_key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=validated_key)
            body = response["Body"]
            try:
                data = body.read()
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
            if not isinstance(data, bytes):
                data = bytes(data)
            return data
        except Exception as exc:
            if _is_not_found_error(exc):
                raise ObjectNotFoundError("Stored object was not found.") from exc
            raise ObjectStorageError("Failed to read S3 object.") from exc

    def delete(self, object_key: str) -> None:
        validated_key = validate_object_key(object_key)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=validated_key)
        except Exception as exc:
            raise ObjectStorageError("Failed to delete S3 object.") from exc
