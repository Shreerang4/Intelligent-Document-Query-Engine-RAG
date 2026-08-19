from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault(
    "ACCESS_JWT_SECRET",
    "phase-6b2-test-secret-that-is-not-used-in-production",
)

import main
import persistence.db as persistence_db
from backend.app.auth.config import ACCESS_JWT_AUDIENCE, ACCESS_JWT_ISSUER
from backend.app.auth.tokens import create_access_token
from backend.app.schemas import AnswerItem
from persistence.db import Base
from persistence.models import Document, Query, User
from persistence.upload_results import recover_upload_request


TEST_SECRET = "phase-6b2-test-secret-that-is-not-used-in-production"
TEST_OPERATIONAL_TOKEN = "test-operational-api-token"
PDF_BYTES = b"%PDF-1.7 phase 6b2 identical upload bytes"
URL = "https://example.com/shared-phase-6b2.pdf"
CHUNKS = [{"text": "owned route passage", "page": 1, "chunk_id": 0}]


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[float(index + 1), 2.0, 3.0] for index, _text in enumerate(texts)]


@pytest.fixture(autouse=True)
def jwt_environment(monkeypatch):
    monkeypatch.setenv("ACCESS_JWT_SECRET", TEST_SECRET)
    main.document_cache.clear()
    yield
    main.document_cache.clear()


@pytest.fixture
def owned_application(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr(persistence_db, "SessionLocal", factory)

    with factory() as session:
        user_a = User(
            email="route-a@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        user_b = User(
            email="route-b@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        session.add_all([user_a, user_b])
        session.commit()
        user_ids = (user_a.id, user_b.id)

    async def fake_url_loader(_url):
        return [dict(CHUNKS[0])]

    async def fake_question_result(question, *_args, **_kwargs):
        return main.ProcessedQuestionResult(
            answer_item=AnswerItem(
                question=question,
                answer="A deterministic owned answer.",
                status="ok",
                sources=[
                    {
                        "page": 1,
                        "chunk_id": 0,
                        "excerpt": "owned route passage",
                    }
                ],
            ),
            source_chunks=[
                {
                    **CHUNKS[0],
                    "retrieval_score": 0.8,
                    "reranker_score": 0.9,
                }
            ],
            latency_ms=1.0,
        )

    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", lambda _pdf: [dict(CHUNKS[0])])
    monkeypatch.setattr(main, "load_and_chunk_pdf", fake_url_loader)
    monkeypatch.setattr(main, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(main, "get_reranker_model", lambda: object())
    monkeypatch.setattr(main, "get_groq_client", lambda: object())
    monkeypatch.setattr(main, "_process_question_result", fake_question_result)

    try:
        with TestClient(main.app) as client:
            yield client, factory, user_ids
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _authorization(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def _expired_token(user_id: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": user_id,
            "iat": now - 700,
            "exp": now - 100,
            "jti": str(uuid.uuid4()),
            "iss": ACCESS_JWT_ISSUER,
            "aud": ACCESS_JWT_AUDIENCE,
        },
        TEST_SECRET,
        algorithm="HS256",
    )


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        (
            "post",
            "/hackrx/run",
            {"json": {"documents": URL, "questions": ["Who owns this?"]}},
        ),
        (
            "post",
            "/hackrx/upload-run",
            {
                "files": {"file": ("same.pdf", PDF_BYTES, "application/pdf")},
                "data": {"questions_json": '["Who owns this?"]'},
            },
        ),
        ("get", "/history/documents", {}),
        ("get", f"/history/documents/{uuid.uuid4()}/queries", {}),
        ("get", f"/history/queries/{uuid.uuid4()}/citations", {}),
    ],
)
def test_user_facing_routes_reject_missing_invalid_expired_and_shared_tokens(
    owned_application,
    method,
    path,
    request_kwargs,
) -> None:
    client, _factory, (user_a_id, _user_b_id) = owned_application
    tokens = [
        None,
        "not-a-jwt",
        _expired_token(user_a_id),
        TEST_OPERATIONAL_TOKEN,
    ]

    for token in tokens:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = client.request(method, path, headers=headers, **request_kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_two_jwt_users_are_isolated_across_upload_url_history_and_ram(
    owned_application,
) -> None:
    client, factory, (user_a_id, user_b_id) = owned_application
    token_a = _authorization(user_a_id)
    token_b = _authorization(user_b_id)
    shared_request_id = str(uuid.uuid4())
    upload_data = {
        "questions_json": '["Who owns this?"]',
        "request_id": shared_request_id,
    }

    upload_a = client.post(
        "/hackrx/upload-run",
        headers=token_a,
        files={"file": ("same.pdf", PDF_BYTES, "application/pdf")},
        data=upload_data,
    )
    upload_b = client.post(
        "/hackrx/upload-run",
        headers=token_b,
        files={"file": ("same.pdf", PDF_BYTES, "application/pdf")},
        data=upload_data,
    )
    url_a = client.post(
        "/hackrx/run",
        headers=token_a,
        json={"documents": URL, "questions": ["URL owner?"]},
    )
    url_b = client.post(
        "/hackrx/run",
        headers=token_b,
        json={"documents": URL, "questions": ["URL owner?"]},
    )

    assert upload_a.status_code == upload_b.status_code == 200
    assert url_a.status_code == url_b.status_code == 200
    assert upload_a.headers["X-Document-Cache"] == "FULL_MISS"
    assert upload_b.headers["X-Document-Cache"] == "FULL_MISS"

    upload_key_a = main._upload_cache_key(user_a_id, PDF_BYTES)
    upload_key_b = main._upload_cache_key(user_b_id, PDF_BYTES)
    url_key_a = main._url_cache_key(user_a_id, URL)
    url_key_b = main._url_cache_key(user_b_id, URL)
    upload_entry_a = main.document_cache[upload_key_a]
    upload_entry_b = main.document_cache[upload_key_b]
    assert upload_key_a.resource_key == upload_key_b.resource_key
    assert upload_key_a != upload_key_b
    assert upload_entry_a.user_id == user_a_id
    assert upload_entry_b.user_id == user_b_id
    assert upload_entry_a.document_id != upload_entry_b.document_id
    assert upload_entry_a.faiss_index is not upload_entry_b.faiss_index
    assert url_key_a in main.document_cache
    assert url_key_b in main.document_cache
    assert main.document_cache[url_key_a].faiss_index is not main.document_cache[url_key_b].faiss_index

    history_a = client.get("/history/documents", headers=token_a)
    history_b = client.get("/history/documents", headers=token_b)
    assert history_a.status_code == history_b.status_code == 200
    documents_a = history_a.json()["documents"]
    documents_b = history_b.json()["documents"]
    ids_a = {document["id"] for document in documents_a}
    ids_b = {document["id"] for document in documents_b}
    assert len(ids_a) == len(ids_b) == 2
    assert ids_a.isdisjoint(ids_b)

    with factory() as session:
        upload_document_a = session.execute(
            select(Document).where(
                Document.user_id == user_a_id,
                Document.source_type == "upload",
            )
        ).scalar_one()
        upload_document_b = session.execute(
            select(Document).where(
                Document.user_id == user_b_id,
                Document.source_type == "upload",
            )
        ).scalar_one()
        query_a = session.execute(
            select(Query).where(
                Query.user_id == user_a_id,
                Query.document_id == upload_document_a.id,
            )
        ).scalar_one()
        query_b = session.execute(
            select(Query).where(
                Query.user_id == user_b_id,
                Query.document_id == upload_document_b.id,
            )
        ).scalar_one()
        assert query_a.request_id == query_b.request_id == shared_request_id

    own_queries = client.get(
        f"/history/documents/{upload_document_a.id}/queries",
        headers=token_a,
    )
    cross_user_queries = client.get(
        f"/history/documents/{upload_document_a.id}/queries",
        headers=token_b,
    )
    own_citations = client.get(
        f"/history/queries/{query_a.id}/citations",
        headers=token_a,
    )
    cross_user_citations = client.get(
        f"/history/queries/{query_a.id}/citations",
        headers=token_b,
    )

    assert own_queries.status_code == 200
    assert own_citations.status_code == 200
    assert own_citations.json()["citations"][0]["chunk_id"] == 0
    assert cross_user_queries.status_code == 404
    assert cross_user_queries.json() == {"detail": "Document not found."}
    assert cross_user_citations.status_code == 404
    assert cross_user_citations.json() == {"detail": "Query not found."}

    assert recover_upload_request(
        user_id=user_a_id,
        request_id=shared_request_id,
        source_hash=upload_document_a.source_hash,
        questions=["Who owns this?"],
    ).document_id == upload_document_a.id
    assert recover_upload_request(
        user_id=user_b_id,
        request_id=shared_request_id,
        source_hash=upload_document_b.source_hash,
        questions=["Who owns this?"],
    ).document_id == upload_document_b.id


def test_operational_database_health_keeps_separate_api_token_protection(
    owned_application,
    monkeypatch,
) -> None:
    client, _factory, (user_a_id, _user_b_id) = owned_application
    monkeypatch.setenv("API_TOKEN", TEST_OPERATIONAL_TOKEN)

    jwt_response = client.get("/health/db", headers=_authorization(user_a_id))
    api_token_response = client.get(
        "/health/db",
        headers={"Authorization": f"Bearer {TEST_OPERATIONAL_TOKEN}"},
    )

    assert jwt_response.status_code == 401
    assert api_token_response.status_code == 200
    assert api_token_response.json() == {"database": "ok"}


def test_database_health_fails_safely_when_operational_token_is_unconfigured(
    owned_application,
    monkeypatch,
) -> None:
    client, _factory, _user_ids = owned_application
    monkeypatch.delenv("API_TOKEN", raising=False)

    response = client.get(
        "/health/db",
        headers={"Authorization": f"Bearer {TEST_OPERATIONAL_TOKEN}"},
    )
    public_health = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Operational database health authentication is not configured."
    }
    assert public_health.status_code == 200
    assert public_health.json()["status"] == "healthy"


def test_application_import_does_not_require_an_operational_api_token() -> None:
    environment = os.environ.copy()
    environment.pop("API_TOKEN", None)
    environment["DATABASE_URL"] = "sqlite://"
    environment.pop("DB_CA_CERT", None)
    environment.pop("DB_CA_CERT_B64", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=main.BASE_DIR,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_production_rag_code_contains_no_legacy_identity_boundary() -> None:
    source = (main.BASE_DIR / "main.py").read_text(encoding="utf-8")

    assert "_get_legacy_api_token_user_id" not in source
    assert "LEGACY_API_TOKEN_USER_ID" not in source
    assert "local-dev-user" not in source
