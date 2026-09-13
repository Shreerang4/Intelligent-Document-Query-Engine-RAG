from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db
from backend.app.storage import document_pdf_object_key
from persistence.db import Base
from persistence.document_objects import (
    DocumentObjectMetadataError,
    attach_owned_document_object_metadata,
    get_document_object_metadata,
    get_owned_document_object_metadata,
)
from persistence.models import Document, User
from persistence.ownership import OwnedResourceNotFoundError


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _add_document(session_factory, *, user_id: str, document_id: str) -> None:
    with session_factory() as session:
        session.add(User(id=user_id, email=f"{user_id}@example.test"))
        session.add(
            Document(
                id=document_id,
                user_id=user_id,
                source_type="upload",
                filename="original private report.pdf",
                source_hash=f"hash-{document_id}",
                cache_key=f"upload:{document_id}",
                status="ready",
                embedding_model="intfloat/e5-small-v2",
                embedding_format="e5-query-passage-v1",
                retrieval_mode="faiss_reranker",
                reranker_model="cross-encoder/test",
                k_initial=20,
                k_final=8,
            )
        )
        session.commit()


def test_existing_document_metadata_is_nullable_and_status_is_unchanged(session_factory) -> None:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    _add_document(session_factory, user_id=user_id, document_id=document_id)

    assert get_owned_document_object_metadata(
        user_id=user_id,
        document_id=document_id,
    ) is None

    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        assert document.status == "ready"
        assert document.object_key is None
        assert document.content_type is None
        assert document.byte_size is None


def test_attach_and_load_document_object_metadata(session_factory) -> None:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    object_key = document_pdf_object_key(document_id)
    _add_document(session_factory, user_id=user_id, document_id=document_id)

    attached = attach_owned_document_object_metadata(
        user_id=user_id,
        document_id=document_id,
        object_key=object_key,
        content_type=" application/pdf ",
        byte_size=1234,
    )
    owned = get_owned_document_object_metadata(user_id=user_id, document_id=document_id)
    internal = get_document_object_metadata(document_id=document_id)

    assert attached == owned == internal
    assert attached.object_key == object_key
    assert attached.content_type == "application/pdf"
    assert attached.byte_size == 1234
    assert attached.user_id == user_id


def test_document_object_metadata_lookup_enforces_ownership(session_factory) -> None:
    owner_id = str(uuid.uuid4())
    other_user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    _add_document(session_factory, user_id=owner_id, document_id=document_id)
    with session_factory() as session:
        session.add(User(id=other_user_id, email=f"{other_user_id}@example.test"))
        session.commit()

    with pytest.raises(OwnedResourceNotFoundError, match="Document not found"):
        get_owned_document_object_metadata(
            user_id=other_user_id,
            document_id=document_id,
        )
    with pytest.raises(OwnedResourceNotFoundError, match="Document not found"):
        attach_owned_document_object_metadata(
            user_id=other_user_id,
            document_id=document_id,
            object_key=document_pdf_object_key(document_id),
            content_type="application/pdf",
            byte_size=1234,
        )


@pytest.mark.parametrize("byte_size", [0, -1, True, 1.5])
def test_document_object_metadata_rejects_invalid_byte_size(session_factory, byte_size) -> None:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    _add_document(session_factory, user_id=user_id, document_id=document_id)

    with pytest.raises(DocumentObjectMetadataError, match="positive integer"):
        attach_owned_document_object_metadata(
            user_id=user_id,
            document_id=document_id,
            object_key=document_pdf_object_key(document_id),
            content_type="application/pdf",
            byte_size=byte_size,
        )


def test_document_object_metadata_rejects_key_for_another_document(session_factory) -> None:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    _add_document(session_factory, user_id=user_id, document_id=document_id)

    with pytest.raises(DocumentObjectMetadataError, match="does not match"):
        attach_owned_document_object_metadata(
            user_id=user_id,
            document_id=document_id,
            object_key=document_pdf_object_key(str(uuid.uuid4())),
            content_type="application/pdf",
            byte_size=1234,
        )
