from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import uuid

import numpy as np
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db

import main
from backend.app.schemas import AnswerItem, SourceReference
from main import ProcessedQuestionResult
from persistence.db import Base
from persistence.document_artifacts import (
    deserialize_embedding,
    load_document_artifacts,
    load_or_backfill_document_embeddings,
    serialize_embedding,
)
from persistence.models import Chunk, Citation, Document, Query, User
from persistence.upload_results import (
    RequestConflictError,
    persist_upload_result_atomic,
    recover_upload_request,
)


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _embedding_rows() -> np.ndarray:
    return np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
    )


def _chunks() -> list[dict]:
    return [
        {"text": "first stored chunk", "page": 1, "chunk_id": 10},
        {"text": "second stored chunk", "page": 2, "chunk_id": 20},
    ]


def _question_result(question: str = "What happened?") -> ProcessedQuestionResult:
    return ProcessedQuestionResult(
        answer_item=AnswerItem(
            question=question,
            answer="The first event happened.",
            status="ok",
            sources=[SourceReference(page=1, chunk_id=10, excerpt="first stored chunk")],
        ),
        source_chunks=[{"text": "first stored chunk", "page": 1, "chunk_id": 10}],
        latency_ms=12.5,
    )


def _ensure_user(user_id: str = "user-1") -> None:
    with persistence_db.SessionLocal() as session:
        if session.get(User, user_id) is None:
            session.add(User(id=user_id))
            session.commit()


def _persist_complete_request(request_id: str | None = None) -> str:
    _ensure_user()
    return persist_upload_result_atomic(
        user_id="user-1",
        proposed_document_id=str(uuid.uuid4()),
        source_hash="pdf-source-hash",
        filename="report.pdf",
        cache_key="upload:test",
        chunks=_chunks(),
        embedding_matrix=_embedding_rows(),
        question_results=[_question_result()],
        request_id=request_id,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
    )


def test_embedding_blob_round_trip_preserves_float32_values() -> None:
    original = np.array([1.25, -2.5, 3.75], dtype=np.float32)

    blob, dimension, dtype = serialize_embedding(original)
    restored = deserialize_embedding(blob, dimension, dtype)

    assert dimension == 3
    assert dtype == "float32"
    assert restored.dtype == np.float32
    assert restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, original)


def test_atomic_full_miss_stores_document_chunks_embeddings_queries_and_citations(session_factory) -> None:
    document_id = _persist_complete_request(request_id=str(uuid.uuid4()))

    with session_factory() as session:
        document = session.get(Document, document_id)
        stored_chunks = session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        ).scalars().all()
        queries = session.execute(select(Query).where(Query.document_id == document_id)).scalars().all()
        citations = session.execute(select(Citation).where(Citation.document_id == document_id)).scalars().all()

    assert document is not None
    assert len(stored_chunks) == 2
    assert all(chunk.embedding_blob for chunk in stored_chunks)
    assert [chunk.embedding_dimension for chunk in stored_chunks] == [3, 3]
    assert [chunk.embedding_dtype for chunk in stored_chunks] == ["float32", "float32"]
    assert len(queries) == 1
    assert queries[0].request_index == 0
    assert len(citations) == 1
    assert citations[0].chunk_db_id == stored_chunks[0].id


def test_atomic_persistence_does_not_load_orm_state_after_commit(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=True, future=True)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    _ensure_user()
    committed = False

    def mark_committed(_session) -> None:
        nonlocal committed
        committed = True

    def reject_post_commit_sql(*_args, **_kwargs) -> None:
        if committed:
            raise AssertionError("ORM state triggered database access after commit")

    event.listen(factory.class_, "after_commit", mark_committed)
    event.listen(engine, "before_cursor_execute", reject_post_commit_sql)
    try:
        document_id = _persist_complete_request(request_id=str(uuid.uuid4()))
    finally:
        event.remove(factory.class_, "after_commit", mark_committed)
        event.remove(engine, "before_cursor_execute", reject_post_commit_sql)
        engine.dispose()

    assert document_id


def test_stored_chunks_and_embeddings_load_in_chunk_index_order(session_factory) -> None:
    with session_factory() as session:
        session.add(User(id="user-1"))
        session.add(
            Document(
                id="doc-ordered",
                user_id="user-1",
                source_type="upload",
                filename="ordered.pdf",
                source_hash="ordered-hash",
                status="ingested",
                embedding_model="intfloat/e5-small-v2",
                embedding_format="e5-query-passage-v1",
                retrieval_mode="faiss_reranker",
                reranker_model="cross-encoder/test",
                k_initial=20,
                k_final=8,
            )
        )
        for chunk_index, text_value, vector in [
            (1, "second", np.array([4.0, 5.0], dtype=np.float32)),
            (0, "first", np.array([1.0, 2.0], dtype=np.float32)),
        ]:
            blob, dimension, dtype = serialize_embedding(vector)
            session.add(
                Chunk(
                    user_id="user-1",
                    document_id="doc-ordered",
                    chunk_id=chunk_index,
                    chunk_index=chunk_index,
                    page_number=chunk_index + 1,
                    text=text_value,
                    embedding_blob=blob,
                    embedding_dimension=dimension,
                    embedding_dtype=dtype,
                )
            )
        session.commit()

    artifacts = load_document_artifacts(
        user_id="user-1",
        source_hash="ordered-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    assert artifacts is not None
    assert [chunk["text"] for chunk in artifacts.chunks] == ["first", "second"]
    np.testing.assert_array_equal(
        artifacts.embedding_matrix,
        np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32),
    )


def test_legacy_chunks_can_be_backfilled_once(session_factory) -> None:
    with session_factory() as session:
        session.add(User(id="user-1"))
        session.add(
            Document(
                id="legacy-doc",
                user_id="user-1",
                source_type="upload",
                filename="legacy.pdf",
                source_hash="legacy-hash",
                status="ingested",
                embedding_model="intfloat/e5-small-v2",
                embedding_format="e5-query-passage-v1",
                retrieval_mode="faiss_reranker",
                reranker_model="cross-encoder/test",
                k_initial=20,
                k_final=8,
            )
        )
        for index, chunk in enumerate(_chunks()):
            session.add(
                Chunk(
                    user_id="user-1",
                    document_id="legacy-doc",
                    chunk_id=chunk["chunk_id"],
                    chunk_index=index,
                    page_number=chunk["page"],
                    text=chunk["text"],
                )
            )
        session.commit()

    before = load_document_artifacts(
        user_id="user-1",
        source_hash="legacy-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )
    assert before is not None
    assert before.needs_embedding_backfill is True
    assert before.embedding_matrix is None

    repaired = load_or_backfill_document_embeddings(
        user_id="user-1",
        document_id="legacy-doc",
        source_hash="legacy-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
        embedding_generator=lambda _chunks: _embedding_rows(),
    )
    after = load_document_artifacts(
        user_id="user-1",
        source_hash="legacy-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    assert after is not None
    assert repaired.needs_embedding_backfill is False
    assert after.needs_embedding_backfill is False
    np.testing.assert_array_equal(after.embedding_matrix, _embedding_rows())


def test_concurrent_legacy_backfill_embeds_document_once(tmp_path, monkeypatch) -> None:
    database_path = (tmp_path / "backfill.db").as_posix()
    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=True, future=True)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)

    with factory() as session:
        session.add(User(id="user-1"))
        session.add(
            Document(
                id="legacy-concurrent-doc",
                user_id="user-1",
                source_type="upload",
                filename="legacy.pdf",
                source_hash="legacy-concurrent-hash",
                status="ingested",
                embedding_model="intfloat/e5-small-v2",
                embedding_format="e5-query-passage-v1",
                retrieval_mode="faiss_reranker",
                reranker_model="cross-encoder/test",
                k_initial=20,
                k_final=8,
            )
        )
        for index, chunk in enumerate(_chunks()):
            session.add(
                Chunk(
                    user_id="user-1",
                    document_id="legacy-concurrent-doc",
                    chunk_id=chunk["chunk_id"],
                    chunk_index=index,
                    page_number=chunk["page"],
                    text=chunk["text"],
                )
            )
        session.commit()

    document_lock = threading.Lock()

    @contextmanager
    def simulated_mysql_lock(*_args, **_kwargs):
        with document_lock:
            yield

    monkeypatch.setattr(persistence_db, "mysql_document_lock", simulated_mysql_lock)
    start = threading.Barrier(2)
    call_count = 0
    count_lock = threading.Lock()

    def generate_embeddings(chunks):
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.05)
        assert [chunk["chunk_id"] for chunk in chunks] == [10, 20]
        return _embedding_rows()

    def attempt_backfill():
        start.wait()
        return load_or_backfill_document_embeddings(
            user_id="user-1",
            document_id="legacy-concurrent-doc",
            source_hash="legacy-concurrent-hash",
            embedding_model="intfloat/e5-small-v2",
            embedding_format="e5-query-passage-v1",
            embedding_generator=generate_embeddings,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: attempt_backfill(), range(2)))
    finally:
        engine.dispose()

    assert call_count == 1
    for result in results:
        np.testing.assert_array_equal(result.embedding_matrix, _embedding_rows())


def test_atomic_persistence_rolls_back_every_table_on_failure(session_factory) -> None:
    _ensure_user()
    bad_result = _question_result()
    bad_result.answer_item.sources = [
        SourceReference(page=1, chunk_id=10, excerpt="valid"),
        {"page": 2, "chunk_id": "not-an-integer", "excerpt": "invalid"},
    ]

    with pytest.raises((TypeError, ValueError)):
        persist_upload_result_atomic(
            user_id="user-1",
            proposed_document_id="rollback-doc",
            source_hash="rollback-hash",
            filename="rollback.pdf",
            cache_key="upload:rollback",
            chunks=_chunks(),
            embedding_matrix=_embedding_rows(),
            question_results=[bad_result],
            request_id=str(uuid.uuid4()),
            embedding_model="intfloat/e5-small-v2",
            embedding_format="e5-query-passage-v1",
            retrieval_mode="faiss_reranker",
            reranker_model="cross-encoder/test",
            k_initial=20,
            k_final=8,
        )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(User)) == 1
        assert session.scalar(select(func.count()).select_from(Document)) == 0
        assert session.scalar(select(func.count()).select_from(Chunk)) == 0
        assert session.scalar(select(func.count()).select_from(Query)) == 0
        assert session.scalar(select(func.count()).select_from(Citation)) == 0


def test_recovery_returns_committed_answer_and_rejects_different_input(session_factory) -> None:
    request_id = str(uuid.uuid4())
    document_id = _persist_complete_request(request_id=request_id)

    recovered = recover_upload_request(
        user_id="user-1",
        request_id=request_id,
        source_hash="pdf-source-hash",
        questions=["What happened?"],
    )

    assert recovered is not None
    assert recovered.document_id == document_id
    assert recovered.response_payload["answers"][0]["answer"] == "The first event happened."
    assert recovered.response_payload["answers"][0]["sources"][0]["chunk_id"] == 10

    with pytest.raises(RequestConflictError):
        recover_upload_request(
            user_id="user-1",
            request_id=request_id,
            source_hash="pdf-source-hash",
            questions=["A different question?"],
        )

    with pytest.raises(RequestConflictError):
        recover_upload_request(
            user_id="user-1",
            request_id=request_id,
            source_hash="different-document-hash",
            questions=["What happened?"],
        )


@pytest.mark.asyncio
async def test_existing_history_endpoints_still_return_atomic_upload_rows(
    session_factory,
) -> None:
    document_id = _persist_complete_request(request_id=str(uuid.uuid4()))

    documents = await main.list_history_documents(limit=100, user_id="user-1")
    queries = await main.list_history_document_queries(
        document_id=document_id,
        limit=100,
        user_id="user-1",
    )
    citations = await main.list_history_query_citations(
        query_id=queries.queries[0].id,
        limit=100,
        user_id="user-1",
    )

    assert documents.documents[0].id == document_id
    assert documents.documents[0].chunk_count == 2
    assert documents.documents[0].query_count == 1
    assert queries.queries[0].answer == "The first event happened."
    assert citations.citations[0].chunk_id == 10
