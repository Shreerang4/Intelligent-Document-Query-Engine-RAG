"""Document-scoped queries use owned persisted vectors and the shared RAG path."""

import importlib
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import persistence.db as persistence_db
from backend.app.auth.tokens import create_access_token
from backend.app.rag.embeddings import get_embedding_input_format_version
from persistence.db import Base
from persistence.document_artifacts import serialize_embedding
from persistence.models import Chunk, Document, Query, User


class QueryOnlyEmbeddingModel:
    def __init__(self):
        self.queries = []

    def embed_query(self, question):
        self.queries.append(question)
        return [1.0, 0.0]

    def embed_documents(self, _texts):
        raise AssertionError("Document embeddings must come from persistence")


class Reranker:
    def predict(self, pairs):
        return [float(len(pairs) - index) for index in range(len(pairs))]


@pytest.fixture
def query_app(monkeypatch):
    monkeypatch.setenv("ACCESS_JWT_SECRET", "document-query-tests-secret-long-enough-for-access-tokens")
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)
    main.document_cache.clear()
    model = QueryOnlyEmbeddingModel()
    monkeypatch.setattr(main, "get_embedding_model", lambda: model)
    monkeypatch.setattr(main, "get_reranker_model", lambda: Reranker())
    monkeypatch.setattr(main, "get_groq_client", lambda: object())
    monkeypatch.setattr(main, "generate_answer", lambda _question, context, _client: f"Found: {context}")
    monkeypatch.setattr(main, "verify_answer_claims", lambda *_args: [])
    monkeypatch.setattr(main, "create_vector_store", lambda *_args: (_ for _ in ()).throw(AssertionError("No document re-embedding")))
    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", lambda *_args: (_ for _ in ()).throw(AssertionError("No PDF parsing")))
    with factory() as session:
        owners = [str(uuid.uuid4()), str(uuid.uuid4())]
        session.add_all([User(id=owner, email=f"{i}@example.test") for i, owner in enumerate(owners)])
        session.commit()
    with TestClient(main.app) as client:
        yield client, factory, owners, model
    main.document_cache.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


def _headers(user_id):
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def _document(factory, owner, status="ready", with_chunks=True):
    document_id = str(uuid.uuid4())
    with factory() as session:
        session.add(Document(
            id=document_id, user_id=owner, source_type="upload", filename="source.pdf",
            status=status, error_message="private SQL detail" if status == "failed" else None,
            embedding_model=main.get_embedding_model_name(),
            embedding_format=get_embedding_input_format_version(),
            retrieval_mode=main.get_retrieval_mode(),
            reranker_model=main.get_reranker_model_name(),
            k_initial=main.get_retrieval_k_initial(), k_final=main.get_retrieval_k_final(),
        ))
        if with_chunks:
            for index, vector in enumerate(([1.0, 0.0], [0.0, 1.0])):
                blob, dimension, dtype = serialize_embedding(np.array(vector, dtype=np.float32))
                session.add(Chunk(
                    user_id=owner, document_id=document_id, chunk_id=index + 1,
                    chunk_index=index, page_number=index + 1, text=f"Evidence {index + 1}",
                    embedding_blob=blob, embedding_dimension=dimension, embedding_dtype=dtype,
                ))
        session.commit()
    return document_id


def test_owned_ready_document_uses_stored_vectors_and_shared_pipeline(query_app, monkeypatch):
    client, factory, (owner, _other), model = query_app
    document_id = _document(factory, owner)
    calls = []
    original_runner = main._run_questions

    async def observe_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return await original_runner(*args, **kwargs)

    monkeypatch.setattr(main, "_run_questions", observe_runner)
    response = client.post(
        f"/documents/{document_id}/queries", json={"question": "  What happened?  "},
        headers=_headers(owner),
    )
    assert response.status_code == 200
    assert response.headers["X-Document-Cache"] == "MISS"
    assert response.json()["answers"][0]["question"] == "What happened?"
    assert response.json()["answers"][0]["status"] == "ok"
    assert response.json()["answers"][0]["sources"][0]["chunk_id"] == 1
    assert model.queries[0] == "query: What happened?"
    assert calls[0][1]["document_id"] == document_id
    assert calls[0][1]["user_id"] == owner
    with factory() as session:
        assert len(session.execute(select(Query).where(Query.document_id == document_id)).scalars().all()) == 1


def test_cache_hit_skips_reconstruction(query_app, monkeypatch):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner)
    first = client.post(f"/documents/{document_id}/queries", json={"question": "First?"}, headers=_headers(owner))
    assert first.status_code == 200
    monkeypatch.setattr(main, "load_owned_document_query_artifacts", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Cache miss")))
    second = client.post(f"/documents/{document_id}/queries", json={"question": "Second?"}, headers=_headers(owner))
    assert second.status_code == 200
    assert second.headers["X-Document-Cache"] == "HIT"


@pytest.mark.parametrize("state", ["queued", "processing", "failed"])
def test_nonready_document_returns_safe_conflict(query_app, state):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner, status=state, with_chunks=False)
    response = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(owner))
    assert response.status_code == 409
    assert "private SQL detail" not in response.text


def test_missing_and_other_users_document_are_identical(query_app):
    client, factory, (owner, other), _model = query_app
    document_id = _document(factory, owner)
    missing = client.post(f"/documents/{uuid.uuid4()}/queries", json={"question": "Why?"}, headers=_headers(other))
    forbidden = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(other))
    assert missing.status_code == forbidden.status_code == 404
    assert missing.json() == forbidden.json() == {"detail": "Document not found."}


def test_auth_and_empty_question(query_app):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner)
    assert client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}).status_code == 401
    response = client.post(f"/documents/{document_id}/queries", json={"question": "  \n "}, headers=_headers(owner))
    assert response.status_code == 400


def test_model_failure_is_safe_and_does_not_change_document_status(query_app, monkeypatch):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner)
    monkeypatch.setattr(main, "get_embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("secret model path")))
    response = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(owner))
    assert response.status_code == 503
    assert "secret model path" not in response.text
    with factory() as session:
        assert session.get(Document, document_id).status == "ready"


def test_llm_failure_is_safe(query_app, monkeypatch):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner)
    monkeypatch.setattr(main, "generate_answer", lambda *_args: (_ for _ in ()).throw(RuntimeError("secret groq token")))
    response = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(owner))
    assert response.status_code == 503
    assert "secret groq token" not in response.text
    with factory() as session:
        assert session.get(Document, document_id).status == "ready"
        assert session.execute(select(Query).where(Query.document_id == document_id)).first() is None


def test_missing_stored_embedding_fails_without_backfill(query_app):
    client, factory, (owner, _other), model = query_app
    document_id = _document(factory, owner)
    with factory() as session:
        chunk = session.execute(select(Chunk).where(Chunk.document_id == document_id)).scalars().first()
        chunk.embedding_blob = None
        session.commit()
    response = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(owner))
    assert response.status_code == 503
    assert model.queries == []
    with factory() as session:
        assert session.get(Document, document_id).status == "ready"


def test_database_failure_is_safe(query_app, monkeypatch):
    client, factory, (owner, _other), _model = query_app
    document_id = _document(factory, owner)
    document_router = importlib.import_module("backend.app.documents.router")

    monkeypatch.setattr(document_router, "get_document_status", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("SELECT secret")))
    response = client.post(f"/documents/{document_id}/queries", json={"question": "Why?"}, headers=_headers(owner))
    assert response.status_code == 503
    assert "SELECT secret" not in response.text
