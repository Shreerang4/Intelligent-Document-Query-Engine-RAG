from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.storage import (
    InvalidObjectKeyError,
    LocalObjectStorage,
    ObjectNotFoundError,
    ObjectStorageConfigurationError,
    ObjectStorageSettings,
    S3ObjectStorage,
    document_pdf_object_key,
)
from backend.app.storage.config import _build_s3_client


class FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    def get_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise FakeS3Error("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[key])}

    def delete_object(self, **kwargs):
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)


@pytest.fixture(params=["local", "s3"])
def object_storage(request, tmp_path):
    if request.param == "local":
        return LocalObjectStorage(tmp_path / "private-objects")
    return S3ObjectStorage(bucket="private-documents", client=FakeS3Client())


def test_storage_contract_put_get_overwrite_delete_and_missing(object_storage) -> None:
    object_key = document_pdf_object_key(str(uuid.uuid4()))

    object_storage.put(object_key, b"first", content_type="application/pdf")
    assert object_storage.get(object_key) == b"first"

    object_storage.put(object_key, b"second", content_type="application/pdf")
    assert object_storage.get(object_key) == b"second"

    object_storage.delete(object_key)
    object_storage.delete(object_key)
    with pytest.raises(ObjectNotFoundError):
        object_storage.get(object_key)


@pytest.mark.parametrize(
    "object_key",
    [
        "",
        "/documents/file.pdf",
        "../documents/file.pdf",
        "documents/../file.pdf",
        "documents\\file.pdf",
        "documents//file.pdf",
        "documents/file name.pdf",
    ],
)
def test_storage_contract_rejects_unsafe_object_keys(object_storage, object_key: str) -> None:
    with pytest.raises(InvalidObjectKeyError):
        object_storage.put(object_key, b"pdf", content_type="application/pdf")


def test_document_key_is_canonical_and_contains_no_user_metadata() -> None:
    document_id = str(uuid.uuid4())
    object_key = document_pdf_object_key(document_id.upper())

    assert object_key == f"documents/{document_id}/source.pdf"
    assert "@" not in object_key


def test_document_key_rejects_non_uuid_document_id() -> None:
    with pytest.raises(InvalidObjectKeyError, match="valid UUID"):
        document_pdf_object_key("../../../user@example.com")


def test_local_storage_commits_with_atomic_replace(monkeypatch, tmp_path) -> None:
    storage = LocalObjectStorage(tmp_path / "private-objects")
    object_key = document_pdf_object_key(str(uuid.uuid4()))
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("backend.app.storage.local.os.replace", recording_replace)
    storage.put(object_key, b"private pdf", content_type="application/pdf")

    assert len(replace_calls) == 1
    temporary_path, destination = replace_calls[0]
    assert temporary_path.parent == destination.parent
    assert not temporary_path.exists()
    assert destination.read_bytes() == b"private pdf"


def test_local_storage_does_not_create_root_until_first_write(tmp_path) -> None:
    root = tmp_path / "not-created-on-import-or-construction"
    storage = LocalObjectStorage(root)

    assert not root.exists()
    with pytest.raises(ObjectNotFoundError):
        storage.get(document_pdf_object_key(str(uuid.uuid4())))
    assert not root.exists()


def test_s3_upload_uses_no_acl_and_exposes_no_url() -> None:
    client = FakeS3Client()
    storage = S3ObjectStorage(bucket="private-documents", client=client)
    object_key = document_pdf_object_key(str(uuid.uuid4()))

    storage.put(object_key, b"private pdf", content_type="application/pdf")

    assert client.put_calls == [
        {
            "Bucket": "private-documents",
            "Key": object_key,
            "Body": b"private pdf",
            "ContentType": "application/pdf",
        }
    ]
    assert "ACL" not in client.put_calls[0]
    assert not hasattr(storage, "public_url")


def _s3_settings(**overrides) -> ObjectStorageSettings:
    values = {
        "backend": "s3",
        "local_root": "./uploads/object-storage",
        "s3_endpoint_url": "https://objects.example.test",
        "s3_bucket": "private-documents",
        "s3_region": "test-1",
        "s3_access_key_id": None,
        "s3_secret_access_key": None,
        "s3_session_token": None,
        "s3_addressing_style": "path",
    }
    values.update(overrides)
    return ObjectStorageSettings(**values)


def test_s3_client_uses_standard_credential_chain_when_explicit_credentials_are_absent() -> None:
    calls: list[tuple[str, dict]] = []
    boto3_module = SimpleNamespace(
        client=lambda service, **kwargs: calls.append((service, kwargs)) or object()
    )
    config_factory = lambda **kwargs: ("config", kwargs)

    _build_s3_client(
        _s3_settings(),
        boto3_module=boto3_module,
        config_factory=config_factory,
    )

    service, arguments = calls[0]
    assert service == "s3"
    assert "aws_access_key_id" not in arguments
    assert "aws_secret_access_key" not in arguments
    assert "aws_session_token" not in arguments


def test_s3_client_passes_explicit_credentials_when_configured() -> None:
    calls: list[dict] = []
    boto3_module = SimpleNamespace(
        client=lambda _service, **kwargs: calls.append(kwargs) or object()
    )

    _build_s3_client(
        _s3_settings(
            s3_access_key_id="access-key",
            s3_secret_access_key="secret-key",
            s3_session_token="session-token",
        ),
        boto3_module=boto3_module,
        config_factory=lambda **kwargs: kwargs,
    )

    assert calls[0]["aws_access_key_id"] == "access-key"
    assert calls[0]["aws_secret_access_key"] == "secret-key"
    assert calls[0]["aws_session_token"] == "session-token"


def test_s3_settings_require_both_explicit_credential_parts() -> None:
    settings = _s3_settings(s3_access_key_id="access-key")
    with pytest.raises(ObjectStorageConfigurationError, match="configured together"):
        settings.validate()
