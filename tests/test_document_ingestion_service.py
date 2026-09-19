from __future__ import annotations

import uuid

import fitz
import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import persistence.db as persistence_db
import persistence.document_ingestion as ingestion_persistence
from backend.app.rag.embeddings import create_chunk_embeddings
from backend.app.rag.ingestion import (
    InvalidPdfError,
    NoMeaningfulTextError,
    parse_and_chunk_pdf_bytes,
)
from backend.app.services.document_ingestion import (
    RETRY_EXHAUSTED_FAILURE_MESSAGE,
    RetryableDocumentIngestionError,
    finalize_document_ingestion_retry_exhaustion,
    ingest_document,
    prepare_document_ingestion_retry,
)
from backend.app.storage import ObjectNotFoundError, document_pdf_object_key
from persistence.db import Base
from persistence.document_artifacts import load_document_artifacts
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
    mark_document_ingestion_failed,
    mark_document_ingestion_queued_for_retry,
    persist_document_ingestion_atomic,
)
from persistence.models import Chunk, Document, User


PDF_BYTES = b"%PDF-1.7 stored source bytes"
CHUNKS = [
    {"text": "The first meaningful stored passage.", "page": 1, "chunk_id": 0},
    {"text": "The second meaningful stored passage.", "page": 2, "chunk_id": 1},
]
MATRIX = np.ascontiguousarray(
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
)


class RecordingObjectStorage:
    def __init__(self, data: bytes = PDF_BYTES, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.get_calls: list[str] = []

    def put(self, object_key: str, data: bytes, *, content_type: str) -> None:
        raise AssertionError("Ingestion must not write the source object.")

    def get(self, object_key: str) -> bytes:
        self.get_calls.append(object_key)
        if self.error is not None:
            raise self.error
        return self.data

    def delete(self, object_key: str) -> None:
        raise AssertionError("Ingestion must retain the source object.")


class RecordingEmbeddingModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed_documents(self, texts):
        self.inputs.append(list(texts))
        return MATRIX.tolist()


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


def _add_document(
    session_factory,
    *,
    status: str = DOCUMENT_STATUS_QUEUED,
    with_object: bool = True,
) -> tuple[str, str, str]:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    object_key = document_pdf_object_key(document_id)
    with session_factory() as session:
        session.add(User(id=user_id, email=f"{user_id}@example.test"))
        session.add(
            Document(
                id=document_id,
                user_id=user_id,
                source_type="upload",
                filename="customer filename.pdf",
                source_hash=f"hash-{document_id}",
                cache_key=f"upload:{document_id}",
                object_key=object_key if with_object else None,
                content_type="application/pdf" if with_object else None,
                byte_size=len(PDF_BYTES) if with_object else None,
                status=status,
                embedding_model="pending-model",
                embedding_format="pending-format",
                retrieval_mode="faiss_reranker",
                reranker_model="cross-encoder/test",
                k_initial=20,
                k_final=8,
            )
        )
        session.commit()
    return user_id, document_id, object_key


def test_new_document_model_defaults_to_queued(session_factory) -> None:
    user_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    with session_factory() as session:
        session.add(User(id=user_id, email=f"{user_id}@example.test"))
        document = Document(
            id=document_id,
            user_id=user_id,
            source_type="upload",
            embedding_model="intfloat/e5-small-v2",
            embedding_format="e5-query-passage-v1",
            retrieval_mode="faiss_reranker",
            reranker_model="cross-encoder/test",
            k_initial=20,
            k_final=8,
        )
        session.add(document)
        session.commit()
        assert document.status == DOCUMENT_STATUS_QUEUED


def test_ingestion_loads_object_transitions_and_persists_complete_artifacts(
    session_factory,
    monkeypatch,
) -> None:
    user_id, document_id, object_key = _add_document(session_factory)
    storage = RecordingObjectStorage()
    observed_statuses: list[str] = []

    def parser(pdf_bytes: bytes):
        assert pdf_bytes == PDF_BYTES
        with session_factory() as session:
            observed_statuses.append(session.get(Document, document_id).status)
        return CHUNKS

    result = ingest_document(
        document_id,
        object_storage=storage,
        embedding_model=object(),
        pdf_parser=parser,
        embedding_builder=lambda chunks, _model: MATRIX,
    )

    assert result.status == DOCUMENT_STATUS_READY
    assert result.chunk_count == 2
    assert storage.get_calls == [object_key]
    assert observed_statuses == [DOCUMENT_STATUS_PROCESSING]

    repeated = ingest_document(
        document_id,
        object_storage=storage,
        embedding_model=object(),
        pdf_parser=lambda _data: (_ for _ in ()).throw(AssertionError("parser must not rerun")),
        embedding_builder=lambda *_args: (_ for _ in ()).throw(
            AssertionError("embedding must not rerun")
        ),
    )
    assert repeated.status == DOCUMENT_STATUS_READY
    assert storage.get_calls == [object_key]

    with session_factory() as session:
        document = session.get(Document, document_id)
        rows = session.execute(
            select(Chunk)
            .where(Chunk.user_id == user_id, Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).scalars().all()
        assert document.status == DOCUMENT_STATUS_READY
        assert document.error_message is None
        assert document.embedding_model == "intfloat/e5-small-v2"
        assert document.embedding_format == "e5-query-passage-v1"
        assert [(row.chunk_id, row.chunk_index, row.page_number, row.text) for row in rows] == [
            (0, 0, 1, CHUNKS[0]["text"]),
            (1, 1, 2, CHUNKS[1]["text"]),
        ]
        assert all(row.embedding_blob is not None for row in rows)


def test_ready_document_returns_immediately_without_object_or_model_access(session_factory) -> None:
    _, document_id, _ = _add_document(session_factory, status=DOCUMENT_STATUS_READY)
    storage = RecordingObjectStorage(error=AssertionError("storage must not be called"))

    result = ingest_document(
        document_id,
        object_storage=storage,
        embedding_model=object(),
        pdf_parser=lambda _data: (_ for _ in ()).throw(AssertionError("parser must not run")),
        embedding_builder=lambda *_args: (_ for _ in ()).throw(
            AssertionError("embedding must not run")
        ),
    )

    assert result.status == DOCUMENT_STATUS_READY
    assert storage.get_calls == []


@pytest.mark.parametrize(
    "content_error",
    [InvalidPdfError("Failed to parse PDF document."), NoMeaningfulTextError("No meaningful text found.")],
)
def test_permanent_pdf_content_failure_marks_document_failed_without_embeddings(
    session_factory,
    content_error,
) -> None:
    _, document_id, object_key = _add_document(session_factory)
    storage = RecordingObjectStorage()
    embedding_called = False

    def fail_embedding(*_args):
        nonlocal embedding_called
        embedding_called = True
        return MATRIX

    result = ingest_document(
        document_id,
        object_storage=storage,
        embedding_model=object(),
        pdf_parser=lambda _data: (_ for _ in ()).throw(content_error),
        embedding_builder=fail_embedding,
    )

    assert result.status == DOCUMENT_STATUS_FAILED
    assert storage.get_calls == [object_key]
    assert embedding_called is False
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_FAILED
        assert document.error_message == str(content_error)
        assert session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all() == []


def test_storage_failure_is_retryable_and_leaves_document_processing(session_factory) -> None:
    _, document_id, _ = _add_document(session_factory)
    storage = RecordingObjectStorage(error=ObjectNotFoundError("temporarily unavailable"))

    with pytest.raises(RetryableDocumentIngestionError, match="load the source"):
        ingest_document(document_id, object_storage=storage, embedding_model=object())

    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_PROCESSING
        assert document.error_message is None


def test_missing_object_metadata_is_a_terminal_state_failure(session_factory) -> None:
    _, document_id, _ = _add_document(session_factory, with_object=False)

    result = ingest_document(document_id, object_storage=RecordingObjectStorage())

    assert result.status == DOCUMENT_STATUS_FAILED
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_FAILED
        assert document.error_message == "Document has no source object metadata."


def test_later_duplicate_failure_cannot_overwrite_ready(session_factory) -> None:
    _, document_id, _ = _add_document(session_factory, status=DOCUMENT_STATUS_PROCESSING)
    completed = persist_document_ingestion_atomic(
        document_id=document_id,
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    losing_failure = mark_document_ingestion_failed(
        document_id=document_id,
        error_message="A duplicate failed after the winner committed.",
    )

    assert completed.status == DOCUMENT_STATUS_READY
    assert losing_failure.status == DOCUMENT_STATUS_READY
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_READY
        assert document.error_message is None
        assert len(session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()) == 2


def test_retry_countdown_state_moves_processing_back_to_queued(session_factory) -> None:
    _, document_id, _ = _add_document(
        session_factory,
        status=DOCUMENT_STATUS_PROCESSING,
    )

    result = prepare_document_ingestion_retry(document_id)

    assert result.status == DOCUMENT_STATUS_QUEUED
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_QUEUED
        assert document.error_message is None


def test_retry_countdown_state_cannot_overwrite_ready(session_factory) -> None:
    _, document_id, _ = _add_document(
        session_factory,
        status=DOCUMENT_STATUS_PROCESSING,
    )
    persist_document_ingestion_atomic(
        document_id=document_id,
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    result = mark_document_ingestion_queued_for_retry(document_id=document_id)

    assert result.status == DOCUMENT_STATUS_READY
    assert result.chunk_count == 2
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document.status == DOCUMENT_STATUS_READY


def test_exhausted_retry_finalizer_is_sanitized_and_ready_safe(session_factory) -> None:
    _, failed_document_id, _ = _add_document(
        session_factory,
        status=DOCUMENT_STATUS_PROCESSING,
    )
    failed = finalize_document_ingestion_retry_exhaustion(failed_document_id)
    assert failed.status == DOCUMENT_STATUS_FAILED
    with session_factory() as session:
        document = session.get(Document, failed_document_id)
        assert document.error_message == RETRY_EXHAUSTED_FAILURE_MESSAGE

    _, ready_document_id, _ = _add_document(
        session_factory,
        status=DOCUMENT_STATUS_PROCESSING,
    )
    persist_document_ingestion_atomic(
        document_id=ready_document_id,
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    losing_finalizer = finalize_document_ingestion_retry_exhaustion(ready_document_id)

    assert losing_finalizer.status == DOCUMENT_STATUS_READY
    with session_factory() as session:
        document = session.get(Document, ready_document_id)
        assert document.status == DOCUMENT_STATUS_READY
        assert document.error_message is None


def test_failed_finalization_rolls_back_chunk_replacement_and_status(
    session_factory,
    monkeypatch,
) -> None:
    user_id, document_id, _ = _add_document(session_factory, status=DOCUMENT_STATUS_PROCESSING)
    with session_factory() as session:
        session.add(
            Chunk(
                user_id=user_id,
                document_id=document_id,
                chunk_id=99,
                chunk_index=0,
                page_number=9,
                text="Previously committed partial artifact.",
                text_hash="old-hash",
                char_count=38,
            )
        )
        session.commit()

    real_serialize = ingestion_persistence.serialize_embedding
    calls = 0

    def fail_second_vector(vector):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected serialization failure")
        return real_serialize(vector)

    monkeypatch.setattr(ingestion_persistence, "serialize_embedding", fail_second_vector)
    with pytest.raises(RuntimeError, match="injected"):
        persist_document_ingestion_atomic(
            document_id=document_id,
            chunks=CHUNKS,
            embedding_matrix=MATRIX,
            embedding_model="intfloat/e5-small-v2",
            embedding_format="e5-query-passage-v1",
        )

    with session_factory() as session:
        document = session.get(Document, document_id)
        rows = session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()
        assert document.status == DOCUMENT_STATUS_PROCESSING
        assert [(row.chunk_id, row.text) for row in rows] == [
            (99, "Previously committed partial artifact.")
        ]


def test_ready_embeddings_follow_existing_mysql_to_faiss_recovery_path(
    session_factory,
) -> None:
    user_id, document_id, _ = _add_document(session_factory, status=DOCUMENT_STATUS_PROCESSING)
    with session_factory() as session:
        source_hash = session.get(Document, document_id).source_hash
    persist_document_ingestion_atomic(
        document_id=document_id,
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    artifacts = load_document_artifacts(
        user_id=user_id,
        source_hash=source_hash,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    assert artifacts is not None
    assert artifacts.document_id == document_id
    assert artifacts.needs_embedding_backfill is False
    np.testing.assert_array_equal(artifacts.embedding_matrix, MATRIX)
    index = main.build_faiss_index(artifacts.embedding_matrix)
    assert index.ntotal == 2


def test_shared_e5_embedding_helper_formats_passages_once(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
    model = RecordingEmbeddingModel()

    matrix = create_chunk_embeddings(CHUNKS, model)

    assert model.inputs == [[f"passage: {chunk['text']}" for chunk in CHUNKS]]
    np.testing.assert_array_equal(matrix, MATRIX)


def test_main_pdf_wrapper_delegates_to_shared_parser(monkeypatch) -> None:
    monkeypatch.setattr(main, "parse_and_chunk_pdf_bytes", lambda data: CHUNKS if data == PDF_BYTES else [])

    assert main.load_and_chunk_pdf_bytes(PDF_BYTES) == CHUNKS


def test_shared_pdf_parser_classifies_invalid_and_empty_documents() -> None:
    with pytest.raises(InvalidPdfError):
        parse_and_chunk_pdf_bytes(b"not a PDF")

    empty_document = fitz.open()
    empty_document.new_page()
    empty_pdf = empty_document.tobytes()
    empty_document.close()
    with pytest.raises(NoMeaningfulTextError):
        parse_and_chunk_pdf_bytes(empty_pdf)


def test_synchronous_retrieval_configuration_delegates_to_shared_settings(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_K_INITIAL", "13")
    monkeypatch.setenv("RETRIEVAL_K_FINAL", "4")
    monkeypatch.setenv("RETRIEVAL_MODE", "e5")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "cross-encoder/shared-test")

    assert main.get_retrieval_k_initial() == 13
    assert main.get_retrieval_k_final() == 4
    assert main.get_retrieval_mode() == "faiss_reranker"
    assert main.get_reranker_model_name() == "cross-encoder/shared-test"

    monkeypatch.setenv("RETRIEVAL_MODE", "")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "")
    assert main.get_retrieval_mode() == "faiss_reranker"
    assert (
        main.get_reranker_model_name()
        == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )
