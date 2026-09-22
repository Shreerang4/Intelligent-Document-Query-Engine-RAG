from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db
from backend.app.rag.config import (
    get_embedding_model_name,
    get_reranker_model_name,
    get_retrieval_k_final,
    get_retrieval_k_initial,
    get_retrieval_mode,
)
from backend.app.rag.embeddings import get_embedding_input_format_version
from backend.app.services import document_upload as upload_service
from backend.app.services.document_upload import (
    DocumentUploadConflictError,
    DocumentUploadUnavailableError,
    DuplicateDocumentError,
    InvalidDocumentUploadError,
    create_document_upload,
)
from persistence.db import Base
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
)
from persistence.document_uploads import QueuedDocumentWriteResult, StoredDocumentStatus
from persistence.models import Document, User


PDF_BYTES = b"%PDF-1.7\nasync service upload"


class RecordingStorage:
    def __init__(self, events: list[tuple] | None = None) -> None:
        self.events = events if events is not None else []
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        self.events.append(("put", object_key, content_type))
        self.objects[object_key] = data

    def get(self, object_key: str) -> bytes:
        return self.objects[object_key]

    def delete(self, object_key: str) -> None:
        self.events.append(("delete", object_key))
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


@pytest.fixture
def session_factory(monkeypatch):
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
        user = User(id=str(uuid.uuid4()), email="upload-service@example.test")
        session.add(user)
        session.commit()
        user_id = user.id
    try:
        yield factory, user_id
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_successful_upload_orders_storage_before_db_before_publish(
    session_factory,
    monkeypatch,
) -> None:
    factory, user_id = session_factory
    events: list[tuple] = []
    storage = RecordingStorage(events)
    real_create = upload_service.upload_persistence.create_or_recover_queued_upload

    def record_create(**kwargs):
        events.append(("db", kwargs["document_id"]))
        return real_create(**kwargs)

    monkeypatch.setattr(
        upload_service.upload_persistence,
        "create_or_recover_queued_upload",
        record_create,
    )

    result = create_document_upload(
        user_id=user_id,
        filename="report.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=str(uuid.uuid4()),
        object_storage=storage,
        enqueue=lambda document_id: events.append(("enqueue", document_id)) or "task-id",
    )

    assert result.http_status_code == 202
    assert result.document.status == DOCUMENT_STATUS_QUEUED
    assert [event[0] for event in events] == ["put", "db", "enqueue"]
    with factory() as session:
        document = session.get(Document, result.document.document_id)
        assert document.user_id == user_id
        assert document.object_key in storage.objects
        assert document.embedding_model == get_embedding_model_name()
        assert document.embedding_format == get_embedding_input_format_version(
            get_embedding_model_name()
        )
        assert document.retrieval_mode == get_retrieval_mode()
        assert document.reranker_model == get_reranker_model_name()
        assert document.k_initial == get_retrieval_k_initial()
        assert document.k_final == get_retrieval_k_final()


@pytest.mark.parametrize(
    ("filename", "content_type", "data"),
    [
        ("report.txt", "application/pdf", PDF_BYTES),
        ("report.pdf", "text/plain", PDF_BYTES),
        ("report.pdf", "application/pdf", b""),
        ("report.pdf", "application/pdf", b"not a pdf"),
    ],
)
def test_invalid_upload_has_no_storage_db_or_queue_side_effects(
    session_factory,
    filename,
    content_type,
    data,
) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    enqueued: list[str] = []

    with pytest.raises(InvalidDocumentUploadError):
        create_document_upload(
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            pdf_bytes=data,
            upload_request_id=str(uuid.uuid4()),
            object_storage=storage,
            enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
        )

    assert storage.objects == {}
    assert enqueued == []
    with factory() as session:
        assert session.query(Document).count() == 0


def test_oversized_upload_has_no_side_effects(session_factory, monkeypatch) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    monkeypatch.setenv("MAX_PDF_BYTES", "8")

    with pytest.raises(InvalidDocumentUploadError, match="maximum"):
        create_document_upload(
            user_id=user_id,
            filename="report.pdf",
            content_type="application/pdf",
            pdf_bytes=PDF_BYTES,
            upload_request_id=None,
            object_storage=storage,
            enqueue=lambda _document_id: "task-id",
        )

    assert storage.objects == {}
    with factory() as session:
        assert session.query(Document).count() == 0


def test_invalid_upload_request_id_has_no_side_effects(session_factory) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    enqueued: list[str] = []

    with pytest.raises(InvalidDocumentUploadError, match="valid UUID"):
        create_document_upload(
            user_id=user_id,
            filename="report.pdf",
            content_type="application/pdf",
            pdf_bytes=PDF_BYTES,
            upload_request_id="not-a-uuid",
            object_storage=storage,
            enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
        )

    assert storage.objects == {}
    assert enqueued == []
    with factory() as session:
        assert session.query(Document).count() == 0


def test_db_failure_after_put_deletes_only_the_new_object(session_factory, monkeypatch) -> None:
    _factory, user_id = session_factory
    storage = RecordingStorage()
    monkeypatch.setattr(
        upload_service.upload_persistence,
        "create_or_recover_queued_upload",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(DocumentUploadUnavailableError, match="metadata"):
        create_document_upload(
            user_id=user_id,
            filename="report.pdf",
            content_type="application/pdf",
            pdf_bytes=PDF_BYTES,
            upload_request_id=None,
            object_storage=storage,
        )

    assert len(storage.deleted) == 1
    assert storage.objects == {}


def test_publish_failure_returns_recoverable_error_and_leaves_row_queued(
    session_factory,
) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()

    with pytest.raises(DocumentUploadUnavailableError) as raised:
        create_document_upload(
            user_id=user_id,
            filename="report.pdf",
            content_type="application/pdf",
            pdf_bytes=PDF_BYTES,
            upload_request_id=str(uuid.uuid4()),
            object_storage=storage,
            enqueue=lambda _document_id: (_ for _ in ()).throw(
                RuntimeError("ambiguous broker failure")
            ),
        )

    assert raised.value.document_id is not None
    assert raised.value.status == DOCUMENT_STATUS_QUEUED
    assert storage.deleted == []
    with factory() as session:
        document = session.get(Document, raised.value.document_id)
        assert document.status == DOCUMENT_STATUS_QUEUED
        assert document.error_message is None
        assert document.object_key in storage.objects


def test_same_request_recovery_republishes_queued_without_new_object(session_factory) -> None:
    _factory, user_id = session_factory
    storage = RecordingStorage()
    request_id = str(uuid.uuid4())
    enqueued: list[str] = []

    first = create_document_upload(
        user_id=user_id,
        filename="first-name.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )
    second = create_document_upload(
        user_id=user_id,
        filename="renamed.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )

    assert second.recovered is True
    assert second.document.document_id == first.document.document_id
    assert second.document.filename == "first-name.pdf"
    assert second.document.status == DOCUMENT_STATUS_QUEUED
    assert len(storage.objects) == 1
    assert enqueued == [first.document.document_id, first.document.document_id]


def test_same_request_with_different_pdf_conflicts_before_new_object(session_factory) -> None:
    _factory, user_id = session_factory
    storage = RecordingStorage()
    request_id = str(uuid.uuid4())
    create_document_upload(
        user_id=user_id,
        filename="report.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda _document_id: "task-id",
    )

    with pytest.raises(DocumentUploadConflictError):
        create_document_upload(
            user_id=user_id,
            filename="report.pdf",
            content_type="application/pdf",
            pdf_bytes=b"%PDF-1.7\ndifferent bytes",
            upload_request_id=request_id,
            object_storage=storage,
            enqueue=lambda _document_id: "task-id",
        )

    assert len(storage.objects) == 1


@pytest.mark.parametrize(
    "terminal_status",
    [DOCUMENT_STATUS_PROCESSING, DOCUMENT_STATUS_READY, DOCUMENT_STATUS_FAILED],
)
def test_recovery_returns_actual_nonqueued_status_without_republishing(
    session_factory,
    terminal_status,
) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    request_id = str(uuid.uuid4())
    first = create_document_upload(
        user_id=user_id,
        filename="report.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda _document_id: "task-id",
    )
    with factory() as session:
        document = session.get(Document, first.document.document_id)
        document.status = terminal_status
        session.commit()
    enqueued: list[str] = []

    recovered = create_document_upload(
        user_id=user_id,
        filename="report.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )

    assert recovered.document.status == terminal_status
    assert recovered.http_status_code == (
        202 if terminal_status == DOCUMENT_STATUS_PROCESSING else 200
    )
    assert enqueued == []


def test_unique_race_deletes_losing_object_and_uses_winner(monkeypatch, session_factory) -> None:
    _factory, user_id = session_factory
    storage = RecordingStorage()
    now = datetime.now(timezone.utc)
    winner = StoredDocumentStatus(
        document_id=str(uuid.uuid4()),
        user_id=user_id,
        filename="winner.pdf",
        source_hash=hashlib.sha256(PDF_BYTES).hexdigest(),
        status=DOCUMENT_STATUS_QUEUED,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(
        upload_service.upload_persistence,
        "recover_owned_upload_request",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        upload_service.upload_persistence,
        "create_or_recover_queued_upload",
        lambda **_kwargs: QueuedDocumentWriteResult(document=winner, created=False),
    )
    monkeypatch.setattr(
        upload_service.upload_persistence,
        "should_enqueue_owned_document",
        lambda **_kwargs: winner,
    )
    enqueued: list[str] = []

    result = create_document_upload(
        user_id=user_id,
        filename="loser.pdf",
        content_type="application/pdf",
        pdf_bytes=PDF_BYTES,
        upload_request_id=str(uuid.uuid4()),
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )

    assert result.recovered is True
    assert result.document.document_id == winner.document_id
    assert len(storage.deleted) == 1
    assert storage.objects == {}
    assert enqueued == [winner.document_id]


@pytest.mark.parametrize("status", [DOCUMENT_STATUS_READY, DOCUMENT_STATUS_QUEUED, DOCUMENT_STATUS_PROCESSING])
def test_same_owner_duplicate_has_no_new_side_effects(session_factory, status) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    enqueued: list[str] = []
    first = create_document_upload(
        user_id=user_id, filename="original.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=str(uuid.uuid4()),
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )
    with factory() as session:
        session.get(Document, first.document.document_id).status = status
        session.commit()
    storage.events.clear()
    enqueued.clear()

    with pytest.raises(DuplicateDocumentError) as raised:
        create_document_upload(
            user_id=user_id, filename="renamed.pdf", content_type="application/pdf",
            pdf_bytes=PDF_BYTES, upload_request_id=str(uuid.uuid4()),
            object_storage=storage,
            enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
        )

    assert raised.value.document.document_id == first.document.document_id
    assert raised.value.document.filename == "original.pdf"
    assert raised.value.document.status == status
    assert storage.events == []
    assert enqueued == []
    with factory() as session:
        assert session.query(Document).count() == 1


def test_failed_match_and_other_users_hash_do_not_block_upload(session_factory) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    first = create_document_upload(
        user_id=user_id, filename="failed.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=None, object_storage=storage,
        enqueue=lambda _document_id: "task-id",
    )
    with factory() as session:
        session.get(Document, first.document.document_id).status = DOCUMENT_STATUS_FAILED
        other = User(id=str(uuid.uuid4()), email="other-duplicate@example.test")
        session.add(other)
        session.commit()
        other_id = other.id

    retry = create_document_upload(
        user_id=user_id, filename="retry.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=None, object_storage=storage,
        enqueue=lambda _document_id: "task-id",
    )
    other_upload = create_document_upload(
        user_id=other_id, filename="other.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=None, object_storage=storage,
        enqueue=lambda _document_id: "task-id",
    )
    assert len({first.document.document_id, retry.document.document_id, other_upload.document.document_id}) == 3
    with factory() as session:
        assert session.query(Document).count() == 3


def test_allow_duplicate_creates_new_document_but_same_request_still_recovers(session_factory) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    enqueued: list[str] = []
    first = create_document_upload(
        user_id=user_id, filename="first.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=str(uuid.uuid4()), object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )
    request_id = str(uuid.uuid4())
    second = create_document_upload(
        user_id=user_id, filename="second.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=request_id, allow_duplicate=True,
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )
    recovered = create_document_upload(
        user_id=user_id, filename="retry.pdf", content_type="application/pdf",
        pdf_bytes=PDF_BYTES, upload_request_id=request_id,
        object_storage=storage,
        enqueue=lambda document_id: enqueued.append(document_id) or "task-id",
    )

    assert second.document.document_id != first.document.document_id
    assert recovered.document.document_id == second.document.document_id
    assert recovered.recovered is True
    assert len(storage.objects) == 2
    with factory() as session:
        assert session.query(Document).count() == 2


def test_historical_matches_prefer_newest_active_then_newest_ready(session_factory) -> None:
    factory, user_id = session_factory
    storage = RecordingStorage()
    ids = []
    for status in (DOCUMENT_STATUS_READY, DOCUMENT_STATUS_PROCESSING, DOCUMENT_STATUS_QUEUED, DOCUMENT_STATUS_READY):
        result = create_document_upload(
            user_id=user_id, filename=f"{status}.pdf", content_type="application/pdf",
            pdf_bytes=PDF_BYTES, upload_request_id=None, allow_duplicate=True,
            object_storage=storage, enqueue=lambda _document_id: "task-id",
        )
        ids.append(result.document.document_id)
        with factory() as session:
            document = session.get(Document, result.document.document_id)
            document.status = status
            document.created_at = datetime(2026, 1, len(ids), tzinfo=timezone.utc)
            session.commit()

    source_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    selected = upload_service.upload_persistence.find_owned_usable_duplicate(
        user_id=user_id, source_hash=source_hash,
    )
    assert selected.document_id == ids[2]
    with factory() as session:
        session.get(Document, ids[1]).status = DOCUMENT_STATUS_FAILED
        session.get(Document, ids[2]).status = DOCUMENT_STATUS_FAILED
        session.commit()
    selected_ready = upload_service.upload_persistence.find_owned_usable_duplicate(
        user_id=user_id, source_hash=source_hash,
    )
    assert selected_ready.document_id == ids[3]
