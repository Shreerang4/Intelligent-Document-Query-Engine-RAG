import os

os.environ.setdefault("API_TOKEN", "test-token-123")

"""
Unit tests for the RAG core logic.
Run with: pytest test_unit.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import faiss
import fitz
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main


def deterministic_vector(text: str, dim: int = 4) -> list[float]:
    data = text.encode("utf-8") or b"\x00"
    vector = []
    for index in range(dim):
        total = sum(data[offset] for offset in range(index, len(data), dim))
        vector.append(float(total + index + 1))
    return vector


def make_mock_embedding_model(
    dim: int = 4,
    query_overrides: dict[str, list[float]] | None = None,
):
    model = MagicMock()
    model.embed_documents.side_effect = lambda texts: [deterministic_vector(text, dim) for text in texts]
    model.embed_query.side_effect = (
        lambda text: query_overrides[text] if query_overrides and text in query_overrides else deterministic_vector(text, dim)
    )
    return model


def make_mock_reranker():
    reranker = MagicMock()
    reranker.predict.side_effect = lambda pairs: [float(len(pair[1])) for pair in pairs]
    return reranker


def make_mock_groq_client(response_text: str = "Mocked answer."):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value.choices = [choice]
    return client


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def make_mock_pdf_response(content: bytes, content_type: str = "application/pdf"):
    response = MagicMock()
    response.content = content
    response.headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(content)),
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture(autouse=True)
def clear_runtime_state():
    main.document_cache.clear()
    main.model_cache.clear()


@pytest.mark.asyncio
async def test_load_and_chunk_pdf_returns_structured_chunks():
    pdf_bytes = make_pdf_bytes(["Hello world. " * 100, "Second page text."])

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = make_mock_pdf_response(pdf_bytes)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        chunks = await main.load_and_chunk_pdf("https://example.com/dummy.pdf")

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(set(chunk) == {"text", "page", "chunk_id"} for chunk in chunks)
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_id"] == 0
    assert all(isinstance(chunk["text"], str) and chunk["text"] for chunk in chunks)


@pytest.mark.asyncio
async def test_load_and_chunk_pdf_raises_on_bad_url():
    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("connection refused")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await main.load_and_chunk_pdf("https://bad-host.invalid/doc.pdf")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_load_and_chunk_pdf_invalid_pdf_returns_422():
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = make_mock_pdf_response(b"not-a-real-pdf")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await main.load_and_chunk_pdf("https://example.com/not-a-pdf.pdf")

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_load_and_chunk_pdf_empty_text_returns_422():
    pdf_bytes = make_pdf_bytes(["", ""])

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = make_mock_pdf_response(pdf_bytes)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await main.load_and_chunk_pdf("https://example.com/blank.pdf")

    assert exc_info.value.status_code == 422


def test_create_vector_store_correct_size():
    chunks = [
        {"text": "chunk one", "page": 1, "chunk_id": 0},
        {"text": "chunk two", "page": 1, "chunk_id": 1},
        {"text": "chunk three", "page": 2, "chunk_id": 2},
    ]
    index = main.create_vector_store(chunks, make_mock_embedding_model())
    assert index.ntotal == len(chunks)


def test_create_vector_store_empty_raises():
    with pytest.raises(HTTPException) as exc_info:
        main.create_vector_store([], make_mock_embedding_model())
    assert exc_info.value.status_code == 400


def test_retrieve_context_filters_faiss_minus_one_and_returns_sources():
    chunks = [{"text": "only chunk", "page": 3, "chunk_id": 9}]
    embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    index = faiss.IndexFlatL2(4)
    index.add(embedding.reshape(1, -1))

    embedding_model = MagicMock()
    embedding_model.embed_query.return_value = embedding.tolist()
    reranker = MagicMock()
    reranker.predict.return_value = [1.0]

    context, sources = main.retrieve_context(
        "test question",
        index,
        chunks,
        embedding_model,
        reranker,
        k_initial=8,
        k_final=1,
    )

    assert context == "only chunk"
    assert len(sources) == 1
    assert sources[0]["page"] == 3
    assert sources[0]["chunk_id"] == 9


def test_retrieve_context_empty_index_returns_empty_values():
    index = faiss.IndexFlatL2(4)
    embedding_model = MagicMock()
    embedding_model.embed_query.return_value = [0.0, 0.0, 0.0, 0.0]

    context, sources = main.retrieve_context("question", index, [], embedding_model, MagicMock())

    assert context == ""
    assert sources == []


def test_generate_answer_returns_llm_response():
    client = make_mock_groq_client("The grace period is 30 days.")
    answer = main.generate_answer("What is the grace period?", "Grace period is 30 days.", client)
    assert answer == "The grace period is 30 days."


def test_generate_answer_llm_failure_raises():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API down")

    with pytest.raises(RuntimeError):
        main.generate_answer("question", "some context", client)


def test_missing_token_returns_401():
    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.post("/hackrx/run", json={"documents": "https://example.com/a.pdf", "questions": ["q"]})
    assert resp.status_code == 401


def test_wrong_token_returns_401():
    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.post(
        "/hackrx/run",
        headers={"Authorization": "Bearer wrong-token"},
        json={"documents": "https://example.com/a.pdf", "questions": ["q"]},
    )
    assert resp.status_code == 401


def test_too_many_questions_returns_400():
    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.post(
        "/hackrx/run",
        headers={"Authorization": "Bearer test-token-123"},
        json={"documents": "https://example.com/a.pdf", "questions": ["q"] * 11},
    )
    assert resp.status_code == 400


def test_run_query_returns_structured_answers():
    question = "What is the grace period?"
    chunks = [
        {
            "text": "The grace period is 30 days from the premium due date.",
            "page": 1,
            "chunk_id": 0,
        }
    ]
    embedding_model = make_mock_embedding_model(
        query_overrides={question: deterministic_vector(chunks[0]["text"])}
    )

    client = TestClient(main.app, raise_server_exceptions=False)
    with (
        patch.object(main, "load_and_chunk_pdf", new=AsyncMock(return_value=chunks)),
        patch.object(main, "get_embedding_model", return_value=embedding_model),
        patch.object(main, "get_reranker_model", return_value=make_mock_reranker()),
        patch.object(main, "get_groq_client", return_value=make_mock_groq_client("The grace period is 30 days.")),
    ):
        resp = client.post(
            "/hackrx/run",
            headers={"Authorization": "Bearer test-token-123"},
            json={"documents": "https://example.com/a.pdf", "questions": [question]},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["answers"], list)
    assert payload["answers"][0]["question"] == question
    assert payload["answers"][0]["answer"] == "The grace period is 30 days."
    assert payload["answers"][0]["status"] == "ok"
    assert isinstance(payload["answers"][0]["sources"], list)
    assert payload["answers"][0]["sources"][0]["page"] == 1
    assert payload["answers"][0]["sources"][0]["chunk_id"] == 0


def test_batch_returns_200_when_one_question_fails():
    chunks = [{"text": "Helpful context", "page": 2, "chunk_id": 7}]
    client = TestClient(main.app, raise_server_exceptions=False)

    def fake_retrieve_context(question, *_args, **_kwargs):
        if question == "bad question":
            raise RuntimeError("boom")
        return "Helpful context", chunks

    with (
        patch.object(main, "load_and_chunk_pdf", new=AsyncMock(return_value=chunks)),
        patch.object(main, "create_vector_store", return_value=MagicMock()),
        patch.object(main, "get_embedding_model", return_value=MagicMock()),
        patch.object(main, "get_reranker_model", return_value=MagicMock()),
        patch.object(main, "get_groq_client", return_value=MagicMock()),
        patch.object(main, "retrieve_context", side_effect=fake_retrieve_context),
        patch.object(main, "generate_answer", return_value="Processed successfully."),
    ):
        resp = client.post(
            "/hackrx/run",
            headers={"Authorization": "Bearer test-token-123"},
            json={
                "documents": "https://example.com/a.pdf",
                "questions": ["good question", "bad question"],
            },
        )

    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert [item["question"] for item in answers] == ["good question", "bad question"]
    assert answers[0]["status"] == "ok"
    assert answers[0]["answer"] == "Processed successfully."
    assert answers[1]["status"] == "error"
    assert answers[1]["answer"] == "Failed to process this question."
    assert answers[1]["sources"] == []


def test_document_cache_headers_show_miss_then_hit():
    chunks = [
        {
            "text": "Cached answer context.",
            "page": 1,
            "chunk_id": 0,
        }
    ]
    embedding_model = make_mock_embedding_model(
        query_overrides={"What is cached?": deterministic_vector(chunks[0]["text"])}
    )
    mock_loader = AsyncMock(return_value=chunks)

    client = TestClient(main.app, raise_server_exceptions=False)
    with (
        patch.object(main, "load_and_chunk_pdf", new=mock_loader),
        patch.object(main, "get_embedding_model", return_value=embedding_model),
        patch.object(main, "get_reranker_model", return_value=make_mock_reranker()),
        patch.object(main, "get_groq_client", return_value=make_mock_groq_client("Cached answer.")),
    ):
        payload = {
            "documents": "https://example.com/cached.pdf",
            "questions": ["What is cached?"],
        }
        first = client.post(
            "/hackrx/run",
            headers={"Authorization": "Bearer test-token-123"},
            json=payload,
        )
        second = client.post(
            "/hackrx/run",
            headers={"Authorization": "Bearer test-token-123"},
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Document-Cache"] == "MISS"
    assert second.headers["X-Document-Cache"] == "HIT"
    assert mock_loader.await_count == 1


def test_health_endpoint():
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert resp.json()["version"] == "2.1.0"
