"""Private local-filesystem object storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from backend.app.storage.base import (
    ObjectNotFoundError,
    ObjectStorageError,
    validate_object_key,
)


class LocalObjectStorage:
    """Store objects under one private root shared by API and worker processes."""

    def __init__(self, root: str | Path) -> None:
        if not str(root).strip():
            raise ValueError("Local object-storage root must not be empty.")
        self.root = Path(root).expanduser().resolve(strict=False)

    def _resolve_object_path(self, object_key: str) -> Path:
        validated_key = validate_object_key(object_key)
        candidate = self.root.joinpath(*validated_key.split("/")).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Object key resolves outside the configured storage root.") from exc
        return candidate

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        destination = self._resolve_object_path(object_key)
        if not isinstance(data, bytes):
            raise TypeError("Object data must be bytes.")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError("Object content type must not be empty.")

        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = destination.parent.resolve(strict=True)
            try:
                resolved_parent.relative_to(self.root)
            except ValueError as exc:
                raise ObjectStorageError(
                    "Object destination escapes the configured storage root."
                ) from exc

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=resolved_parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            temporary_path.chmod(0o600)
            os.replace(temporary_path, destination)
            temporary_path = None
        except ObjectStorageError:
            raise
        except Exception as exc:
            raise ObjectStorageError("Failed to store local object.") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def get(self, object_key: str) -> bytes:
        object_path = self._resolve_object_path(object_key)
        try:
            return object_path.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise ObjectNotFoundError("Stored object was not found.") from exc
        except OSError as exc:
            raise ObjectStorageError("Failed to read local object.") from exc

    def delete(self, object_key: str) -> None:
        object_path = self._resolve_object_path(object_key)
        try:
            object_path.unlink(missing_ok=True)
        except IsADirectoryError as exc:
            raise ObjectNotFoundError("Stored object was not found.") from exc
        except OSError as exc:
            raise ObjectStorageError("Failed to delete local object.") from exc
