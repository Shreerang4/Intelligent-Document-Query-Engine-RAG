from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db
from backend.app.storage import document_pdf_object_key
from persistence.db import Base
from persistence.document_ingestion import (
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
)
from persistence.document_uploads import (
    UploadRequestConflictError,
    create_or_recover_queued_upload,
    get_owned_document_status,
    recover_owned_upload_request,
    should_enqueue_owned_document,
)
from persistence.models import Document, User
from persistence.ownership import OwnedResourceNotFoundError


PDF_BYTES = b"%PDF-1.7 persistence upload"
SOURCE_HASH = hashlib.sha256(PDF_BYTES).hexdigest()
MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "mysql"
    / "005_document_upload_idempotency.sql"
)


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
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _add_user(session_factory) -> str:
    user_id = str(uuid.uuid4())
    with session_factory() as session:
        session.add(User(id=user_id, email=f"{user_id}@example.test"))
        session.commit()
    return user_id


def _create(session_factory, *, user_id: str, request_id: str, source_hash: str = SOURCE_HASH):
    document_id = str(uuid.uuid4())
    return create_or_recover_queued_upload(
        user_id=user_id,
        document_id=document_id,
        upload_request_id=request_id,
        filename="report.pdf",
        source_hash=source_hash,
        object_key=document_pdf_object_key(document_id),
        content_type="application/pdf",
        byte_size=len(PDF_BYTES),
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
    )


def test_create_queued_upload_persists_owner_object_and_configuration(session_factory) -> None:
    user_id = _add_user(session_factory)
    request_id = str(uuid.uuid4())

    result = _create(session_factory, user_id=user_id, request_id=request_id)

    assert result.created is True
    assert result.document.user_id == user_id
    assert result.document.status == DOCUMENT_STATUS_QUEUED
    with session_factory() as session:
        document = session.get(Document, result.document.document_id)
        assert document.upload_request_id == request_id
        assert document.object_key == document_pdf_object_key(document.id)
        assert document.source_hash == SOURCE_HASH
        assert document.content_type == "application/pdf"
        assert document.byte_size == len(PDF_BYTES)
        assert document.embedding_model == "intfloat/e5-small-v2"
        assert document.embedding_format == "e5-query-passage-v1"
        assert document.retrieval_mode == "faiss_reranker"
        assert document.reranker_model == "cross-encoder/test"
        assert document.k_initial == 20
        assert document.k_final == 8


def test_upload_idempotency_migration_is_narrow_and_repeatable() -> None:
    migration = MIGRATION_PATH.read_text(encoding="utf-8").lower()

    assert "add column `upload_request_id` varchar(36) null" in migration
    assert "(`user_id`, `upload_request_id`)" in migration
    assert "uq_documents_user_upload_request_id" in migration
    assert "information_schema.columns" in migration
    assert "information_schema.statistics" in migration
    assert "object_key" not in migration


def test_owned_request_recovery_and_hash_conflict(session_factory) -> None:
    user_id = _add_user(session_factory)
    request_id = str(uuid.uuid4())
    created = _create(session_factory, user_id=user_id, request_id=request_id)

    recovered = recover_owned_upload_request(
        user_id=user_id,
        upload_request_id=request_id,
        source_hash=SOURCE_HASH,
    )

    assert recovered is not None
    assert recovered.document_id == created.document.document_id
    with pytest.raises(UploadRequestConflictError):
        recover_owned_upload_request(
            user_id=user_id,
            upload_request_id=request_id,
            source_hash="0" * 64,
        )


def test_unique_constraint_loser_recovers_one_authoritative_document(session_factory) -> None:
    user_id = _add_user(session_factory)
    request_id = str(uuid.uuid4())
    winner = _create(session_factory, user_id=user_id, request_id=request_id)

    loser = _create(session_factory, user_id=user_id, request_id=request_id)

    assert loser.created is False
    assert loser.document.document_id == winner.document.document_id
    with session_factory() as session:
        rows = session.query(Document).filter_by(
            user_id=user_id,
            upload_request_id=request_id,
        ).all()
        assert [row.id for row in rows] == [winner.document.document_id]


def test_unique_constraint_loser_with_different_hash_conflicts(session_factory) -> None:
    user_id = _add_user(session_factory)
    request_id = str(uuid.uuid4())
    _create(session_factory, user_id=user_id, request_id=request_id)

    with pytest.raises(UploadRequestConflictError):
        _create(
            session_factory,
            user_id=user_id,
            request_id=request_id,
            source_hash="0" * 64,
        )


def test_upload_request_id_is_scoped_to_owner(session_factory) -> None:
    first_user_id = _add_user(session_factory)
    second_user_id = _add_user(session_factory)
    request_id = str(uuid.uuid4())

    first = _create(
        session_factory,
        user_id=first_user_id,
        request_id=request_id,
    )
    second = _create(
        session_factory,
        user_id=second_user_id,
        request_id=request_id,
    )

    assert first.created is True
    assert second.created is True
    assert first.document.document_id != second.document.document_id


def test_status_lookup_enforces_ownership_and_preserves_current_state(session_factory) -> None:
    owner_id = _add_user(session_factory)
    other_id = _add_user(session_factory)
    created = _create(
        session_factory,
        user_id=owner_id,
        request_id=str(uuid.uuid4()),
    )
    with session_factory() as session:
        document = session.get(Document, created.document.document_id)
        document.status = DOCUMENT_STATUS_PROCESSING
        session.commit()

    current = get_owned_document_status(
        user_id=owner_id,
        document_id=created.document.document_id,
    )

    assert current.status == DOCUMENT_STATUS_PROCESSING
    assert should_enqueue_owned_document(
        user_id=owner_id,
        document_id=created.document.document_id,
    ).status == DOCUMENT_STATUS_PROCESSING
    with pytest.raises(OwnedResourceNotFoundError):
        get_owned_document_status(
            user_id=other_id,
            document_id=created.document.document_id,
        )
