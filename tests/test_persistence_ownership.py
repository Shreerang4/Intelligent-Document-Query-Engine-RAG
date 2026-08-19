from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import persistence.db as persistence_db
from persistence.db import Base
from persistence.document_artifacts import (
    ArtifactValidationError,
    load_document_artifacts,
    load_or_backfill_document_embeddings,
)
from persistence.history import (
    list_user_document_queries,
    list_user_documents,
    list_user_query_citations,
)
from persistence.ingestion import persist_ingested_document_best_effort
from persistence.models import Citation, Document, Query, User
from persistence.ownership import OwnedResourceNotFoundError, PersistenceUserNotFoundError
from persistence.query_history import persist_query_result_best_effort
from persistence.upload_results import persist_upload_result_atomic, recover_upload_request


USER_A = "user-a"
USER_B = "user-b"
CHUNKS = [{"text": "owned stored passage", "page": 1, "chunk_id": 7}]
MATRIX = np.ascontiguousarray(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    with factory() as session:
        session.add_all([User(id=USER_A), User(id=USER_B)])
        session.commit()
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _question_result(question: str = "Who owns this?") -> dict:
    return {
        "answer_item": {
            "question": question,
            "answer": "The current user owns it.",
            "status": "ok",
            "sources": [{"page": 1, "chunk_id": 7, "excerpt": "owned stored passage"}],
            "claim_verifications": [],
        },
        "source_chunks": [
            {
                "text": "owned stored passage",
                "page": 1,
                "chunk_id": 7,
                "retrieval_score": 0.8,
                "reranker_score": 0.9,
            }
        ],
        "is_abstained": False,
        "latency_ms": 4.5,
    }


def _persist_upload(
    *,
    user_id: str,
    document_id: str,
    source_hash: str = "same-pdf-hash",
    request_id: str | None = None,
    question: str = "Who owns this?",
) -> str:
    return persist_upload_result_atomic(
        user_id=user_id,
        proposed_document_id=document_id,
        source_hash=source_hash,
        filename="same.pdf",
        cache_key="upload:same",
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        question_results=[_question_result(question)],
        request_id=request_id,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
    )


def test_same_pdf_creates_separate_owned_documents_and_artifacts(session_factory) -> None:
    document_a = _persist_upload(user_id=USER_A, document_id="document-a")

    artifacts_a = load_document_artifacts(
        user_id=USER_A,
        source_hash="same-pdf-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )
    artifacts_b_before = load_document_artifacts(
        user_id=USER_B,
        source_hash="same-pdf-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    assert artifacts_a is not None and artifacts_a.document_id == document_a
    assert artifacts_b_before is None
    with pytest.raises(ArtifactValidationError, match="disappeared"):
        load_or_backfill_document_embeddings(
            user_id=USER_B,
            document_id=document_a,
            source_hash="same-pdf-hash",
            embedding_model="intfloat/e5-small-v2",
            embedding_format="e5-query-passage-v1",
            embedding_generator=lambda _chunks: MATRIX,
        )

    document_b = _persist_upload(user_id=USER_B, document_id="document-b")
    artifacts_b = load_document_artifacts(
        user_id=USER_B,
        source_hash="same-pdf-hash",
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
    )

    assert document_b != document_a
    assert artifacts_b is not None and artifacts_b.document_id == document_b
    np.testing.assert_array_equal(artifacts_b.embedding_matrix, MATRIX)
    with session_factory() as session:
        owners = session.execute(
            select(Document.id, Document.user_id).order_by(Document.id)
        ).all()
    assert owners == [("document-a", USER_A), ("document-b", USER_B)]


def test_history_queries_and_citations_are_not_recoverable_by_another_user(
    session_factory,
) -> None:
    document_a = _persist_upload(user_id=USER_A, document_id="history-document-a")

    documents_a = list_user_documents(user_id=USER_A, limit=100)
    documents_b = list_user_documents(user_id=USER_B, limit=100)
    queries_a = list_user_document_queries(
        user_id=USER_A,
        document_id=document_a,
        limit=100,
    )
    query_a = queries_a[0]
    citations_a = list_user_query_citations(
        user_id=USER_A,
        query_id=query_a.id,
        limit=100,
    )

    assert [document.id for document in documents_a] == [document_a]
    assert documents_b == []
    assert len(queries_a) == 1
    assert len(citations_a) == 1
    with pytest.raises(OwnedResourceNotFoundError, match="Document not found"):
        list_user_document_queries(user_id=USER_B, document_id=document_a, limit=100)
    with pytest.raises(OwnedResourceNotFoundError, match="Query not found"):
        list_user_query_citations(user_id=USER_B, query_id=query_a.id, limit=100)

    unowned_query = persist_query_result_best_effort(
        user_id=USER_B,
        document_id=document_a,
        question="Cross-user attempt",
        answer="Must not persist.",
        status="ok",
        is_abstained=False,
        claim_verifications=[],
        sources=[{"page": 1, "chunk_id": 7, "excerpt": "owned"}],
        source_chunks=CHUNKS,
        embedding_model="intfloat/e5-small-v2",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
        latency_ms=1.0,
    )
    assert unowned_query is None
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(Query).where(Query.user_id == USER_B)
        ) == 0
        assert session.scalar(
            select(func.count()).select_from(Citation).where(Citation.user_id == USER_B)
        ) == 0


def test_request_id_is_namespaced_by_user_and_can_be_reused_independently(
    session_factory,
) -> None:
    request_id = "shared-request-id"
    document_a = _persist_upload(
        user_id=USER_A,
        document_id="request-document-a",
        request_id=request_id,
        question="Question A?",
    )

    assert recover_upload_request(
        user_id=USER_B,
        request_id=request_id,
        source_hash="same-pdf-hash",
        questions=["Question A?"],
    ) is None

    document_b = _persist_upload(
        user_id=USER_B,
        document_id="request-document-b",
        request_id=request_id,
        question="Question B?",
    )
    recovered_a = recover_upload_request(
        user_id=USER_A,
        request_id=request_id,
        source_hash="same-pdf-hash",
        questions=["Question A?"],
    )
    recovered_b = recover_upload_request(
        user_id=USER_B,
        request_id=request_id,
        source_hash="same-pdf-hash",
        questions=["Question B?"],
    )

    assert recovered_a is not None and recovered_a.document_id == document_a
    assert recovered_b is not None and recovered_b.document_id == document_b


def test_url_ingestion_and_query_history_use_the_explicit_owner(session_factory) -> None:
    common_arguments = {
        "source_type": "url",
        "source_url": "https://example.com/shared.pdf",
        "cache_key": "url:shared-cache-key",
        "chunks": CHUNKS,
        "embedding_model": "intfloat/e5-small-v2",
        "embedding_format": "e5-query-passage-v1",
        "retrieval_mode": "faiss_reranker",
        "reranker_model": "cross-encoder/test",
        "k_initial": 20,
        "k_final": 8,
    }
    document_a = persist_ingested_document_best_effort(user_id=USER_A, **common_arguments)
    document_b = persist_ingested_document_best_effort(user_id=USER_B, **common_arguments)

    assert document_a is not None
    assert document_b is not None
    assert document_b != document_a

    query_a = persist_query_result_best_effort(
        user_id=USER_A,
        document_id=document_a,
        question="URL question?",
        answer="URL answer.",
        status="ok",
        is_abstained=False,
        claim_verifications=[],
        sources=[{"page": 1, "chunk_id": 7, "excerpt": "owned stored passage"}],
        source_chunks=CHUNKS,
        embedding_model="intfloat/e5-small-v2",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
        latency_ms=2.0,
    )
    assert query_a is not None
    assert len(list_user_document_queries(user_id=USER_A, document_id=document_a, limit=100)) == 1
    with pytest.raises(OwnedResourceNotFoundError):
        list_user_document_queries(user_id=USER_B, document_id=document_a, limit=100)


def test_missing_user_is_not_auto_created_by_upload_or_url_persistence(session_factory) -> None:
    with pytest.raises(PersistenceUserNotFoundError):
        _persist_upload(user_id="missing-user", document_id="missing-user-document")

    url_result = persist_ingested_document_best_effort(
        user_id="another-missing-user",
        source_type="url",
        source_url="https://example.com/missing.pdf",
        cache_key="url:missing",
        chunks=CHUNKS,
        embedding_model="intfloat/e5-small-v2",
        embedding_format="e5-query-passage-v1",
        retrieval_mode="faiss_reranker",
        reranker_model="cross-encoder/test",
        k_initial=20,
        k_final=8,
    )
    assert url_result is None
    with session_factory() as session:
        assert session.get(User, "missing-user") is None
        assert session.get(User, "another-missing-user") is None


def test_owned_persistence_entrypoints_require_user_id_and_have_no_global_context() -> None:
    for function in (
        persist_upload_result_atomic,
        recover_upload_request,
        persist_ingested_document_best_effort,
        persist_query_result_best_effort,
        load_document_artifacts,
        load_or_backfill_document_embeddings,
        list_user_documents,
        list_user_document_queries,
        list_user_query_citations,
    ):
        user_parameter = inspect.signature(function).parameters["user_id"]
        assert user_parameter.default is inspect.Parameter.empty

    persistence_root = Path(__file__).resolve().parents[1] / "persistence"
    persistence_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in persistence_root.glob("*.py")
    )
    assert "get_current_user_id" not in persistence_source
    assert "local-dev-user" not in persistence_source
