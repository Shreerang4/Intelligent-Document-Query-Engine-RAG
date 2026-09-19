from __future__ import annotations

import importlib
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db
from backend.app.auth.tokens import create_access_token
from backend.app.documents.router import router
from backend.app.services.document_upload import create_document_upload
from persistence.db import Base
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
)
from persistence.models import Document, User


document_router_module = importlib.import_module("backend.app.documents.router")
TEST_SECRET = "async-document-api-test-secret-that-is-not-for-production"
PDF_BYTES = b"%PDF-1.7\nasync API upload"


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        assert content_type == "application/pdf"
        self.objects[object_key] = data

    def get(self, object_key: str) -> bytes:
        return self.objects[object_key]

    def delete(self, object_key: str) -> None:
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


@pytest.fixture
def application(monkeypatch):
    monkeypatch.setenv("ACCESS_JWT_SECRET", TEST_SECRET)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    with factory() as session:
        user_a = User(id=str(uuid.uuid4()), email="async-a@example.test")
        user_b = User(id=str(uuid.uuid4()), email="async-b@example.test")
        session.add_all([user_a, user_b])
        session.commit()
        user_ids = (user_a.id, user_b.id)

    storage = MemoryStorage()
    enqueued: list[str] = []

    def invoke_service(**kwargs):
        return create_document_upload(
            **kwargs,
            object_storage=storage,
            enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
        )

    monkeypatch.setattr(document_router_module, "create_document_upload", invoke_service)
    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            yield client, factory, user_ids, storage, enqueued
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _headers(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def _upload(client, user_id: str, *, request_id: str | None = None, data: bytes = PDF_BYTES):
    form = {"upload_request_id": request_id} if request_id is not None else {}
    return client.post(
        "/documents/upload",
        headers=_headers(user_id),
        files={"file": ("statement.pdf", data, "application/pdf")},
        data=form,
    )


def test_upload_requires_existing_jwt_authentication(application) -> None:
    client, _factory, _users, _storage, _enqueued = application

    response = client.post(
        "/documents/upload",
        files={"file": ("statement.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 401


def test_authenticated_upload_returns_202_and_only_safe_fields(application) -> None:
    client, factory, (user_id, _), storage, enqueued = application

    response = _upload(client, user_id, request_id=str(uuid.uuid4()))

    assert response.status_code == 202
    assert set(response.json()) == {"document_id", "filename", "status"}
    assert response.json()["status"] == DOCUMENT_STATUS_QUEUED
    assert "object" not in response.text.lower()
    assert "bucket" not in response.text.lower()
    assert enqueued == [response.json()["document_id"]]
    with factory() as session:
        document = session.get(Document, response.json()["document_id"])
        assert document.user_id == user_id
        assert document.object_key in storage.objects


@pytest.mark.parametrize(
    ("filename", "content_type", "data"),
    [
        ("statement.txt", "application/pdf", PDF_BYTES),
        ("statement.pdf", "text/plain", PDF_BYTES),
        ("statement.pdf", "application/pdf", b""),
        ("statement.pdf", "application/pdf", b"not-pdf"),
    ],
)
def test_invalid_upload_creates_no_artifacts(
    application,
    filename,
    content_type,
    data,
) -> None:
    client, factory, (user_id, _), storage, enqueued = application

    response = client.post(
        "/documents/upload",
        headers=_headers(user_id),
        files={"file": (filename, data, content_type)},
    )

    assert response.status_code == 400
    assert storage.objects == {}
    assert enqueued == []
    with factory() as session:
        assert session.query(Document).count() == 0


def test_oversized_upload_creates_no_artifacts(application, monkeypatch) -> None:
    client, factory, (user_id, _), storage, enqueued = application
    monkeypatch.setenv("MAX_PDF_BYTES", "8")

    response = _upload(client, user_id)

    assert response.status_code == 400
    assert storage.objects == {}
    assert enqueued == []
    with factory() as session:
        assert session.query(Document).count() == 0


def test_publish_failure_returns_structured_503_and_keeps_queued_document(
    application,
    monkeypatch,
) -> None:
    client, factory, (user_id, _), storage, _enqueued = application

    def invoke_with_failed_publish(**kwargs):
        return create_document_upload(
            **kwargs,
            object_storage=storage,
            enqueue=lambda _document_id: (_ for _ in ()).throw(RuntimeError("broker down")),
        )

    monkeypatch.setattr(
        document_router_module,
        "create_document_upload",
        invoke_with_failed_publish,
    )

    response = _upload(client, user_id, request_id=str(uuid.uuid4()))

    assert response.status_code == 503
    assert set(response.json()) == {"document_id", "status", "message"}
    assert response.json()["status"] == DOCUMENT_STATUS_QUEUED
    with factory() as session:
        document = session.get(Document, response.json()["document_id"])
        assert document.status == DOCUMENT_STATUS_QUEUED
        assert document.error_message is None
        assert document.object_key in storage.objects


@pytest.mark.parametrize(
    "lifecycle_status",
    [
        DOCUMENT_STATUS_QUEUED,
        DOCUMENT_STATUS_PROCESSING,
        DOCUMENT_STATUS_READY,
        DOCUMENT_STATUS_FAILED,
    ],
)
def test_status_endpoint_returns_exact_lifecycle_and_enforces_safe_fields(
    application,
    lifecycle_status,
) -> None:
    client, factory, (user_id, _), _storage, _enqueued = application
    upload = _upload(client, user_id, request_id=str(uuid.uuid4()))
    document_id = upload.json()["document_id"]
    with factory() as session:
        document = session.get(Document, document_id)
        document.status = lifecycle_status
        document.error_message = "Safe ingestion failure." if lifecycle_status == "failed" else None
        session.commit()

    response = client.get(f"/documents/{document_id}", headers=_headers(user_id))

    assert response.status_code == 200
    assert response.json()["status"] == lifecycle_status
    assert set(response.json()) == {
        "document_id",
        "filename",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    }
    assert "object_key" not in response.text
    assert response.json()["error_message"] == (
        "Safe ingestion failure." if lifecycle_status == DOCUMENT_STATUS_FAILED else None
    )


def test_status_endpoint_hides_missing_and_other_users_documents(application) -> None:
    client, _factory, (owner_id, other_id), _storage, _enqueued = application
    upload = _upload(client, owner_id)
    document_id = upload.json()["document_id"]

    missing = client.get(f"/documents/{uuid.uuid4()}", headers=_headers(other_id))
    forbidden = client.get(f"/documents/{document_id}", headers=_headers(other_id))

    assert missing.status_code == forbidden.status_code == 404
    assert missing.json() == forbidden.json() == {"detail": "Document not found."}


@pytest.mark.parametrize(
    ("lifecycle_status", "expected_http_status"),
    [
        (DOCUMENT_STATUS_QUEUED, 202),
        (DOCUMENT_STATUS_PROCESSING, 202),
        (DOCUMENT_STATUS_READY, 200),
        (DOCUMENT_STATUS_FAILED, 200),
    ],
)
def test_same_request_recovery_returns_actual_status_and_sensible_http_code(
    application,
    lifecycle_status,
    expected_http_status,
) -> None:
    client, factory, (user_id, _), _storage, enqueued = application
    request_id = str(uuid.uuid4())
    first = _upload(client, user_id, request_id=request_id)
    document_id = first.json()["document_id"]
    with factory() as session:
        document = session.get(Document, document_id)
        document.status = lifecycle_status
        session.commit()
    enqueued.clear()

    recovered = _upload(client, user_id, request_id=request_id)

    assert recovered.status_code == expected_http_status
    assert recovered.json()["document_id"] == document_id
    assert recovered.json()["status"] == lifecycle_status
    assert enqueued == ([document_id] if lifecycle_status == DOCUMENT_STATUS_QUEUED else [])
