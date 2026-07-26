from __future__ import annotations

import io
import json
import os
import uuid

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException, Response, UploadFile
from starlette.datastructures import Headers

os.environ.setdefault("API_TOKEN", "test-api-token")

import main
from backend.app.schemas import AnswerItem, QueryRequest
from persistence.document_artifacts import StoredDocumentArtifacts
from persistence.upload_results import RecoveredUploadResult, RequestConflictError


PDF_BYTES = b"%PDF-1.7 test upload bytes"
CHUNKS = [{"text": "stored passage", "page": 1, "chunk_id": 0}]
MATRIX = np.ascontiguousarray(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


class FakeEmbeddingModel:
    def __init__(self, *, fail_document_embedding: bool = False) -> None:
        self.fail_document_embedding = fail_document_embedding
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts):
        if self.fail_document_embedding:
            raise AssertionError("embed_documents must not run on this path")
        self.document_calls.append(list(texts))
        return MATRIX.tolist()


@pytest.fixture(autouse=True)
def clear_process_caches():
    main.document_cache.clear()
    main.model_cache.clear()
    yield
    main.document_cache.clear()
    main.model_cache.clear()


def _upload_file() -> UploadFile:
    return UploadFile(
        file=io.BytesIO(PDF_BYTES),
        filename="report.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )


def _processed_result(question: str = "What happened?") -> main.ProcessedQuestionResult:
    return main.ProcessedQuestionResult(
        answer_item=AnswerItem(
            question=question,
            answer="A grounded answer.",
            status="ok",
            sources=[],
        ),
        source_chunks=[],
        latency_ms=5.0,
    )


def _stub_question_pipeline(monkeypatch) -> None:
    async def fake_run(questions, *_args, **_kwargs):
        return [_processed_result(question) for question in questions]

    monkeypatch.setattr(main, "_run_question_results", fake_run)
    monkeypatch.setattr(main, "get_reranker_model", lambda: object())
    monkeypatch.setattr(main, "get_groq_client", lambda: object())


async def _call_upload(*, request_id: str | None = None) -> tuple[Response, object]:
    response = Response()
    result = await main.upload_query_pipeline(
        response=response,
        background_tasks=BackgroundTasks(),
        file=_upload_file(),
        questions_json=json.dumps(["What happened?"]),
        request_id=request_id,
        authorization=f"Bearer {main.EXPECTED_TOKEN}",
    )
    return response, result


@pytest.mark.asyncio
async def test_ram_hit_does_not_load_database_artifacts(monkeypatch) -> None:
    index = main.build_faiss_index(MATRIX)
    cache_key = main._upload_cache_key(PDF_BYTES)
    main._set_document_cache_entry(
        cache_key,
        CHUNKS,
        index,
        document_id="ram-document",
    )

    def fail_artifact_load(**_kwargs):
        raise AssertionError("RAM hit must not query stored document artifacts")

    captured = {}
    monkeypatch.setattr(main, "load_document_artifacts", fail_artifact_load)
    monkeypatch.setattr(main, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **kwargs: captured.update(kwargs) or "ram-document",
    )
    _stub_question_pipeline(monkeypatch)

    response, result = await _call_upload()

    assert response.headers["X-Document-Cache"] == "RAM_HIT"
    assert result.answers[0].answer == "A grounded answer."
    assert captured["embedding_matrix"] is None


@pytest.mark.asyncio
async def test_mysql_hit_rebuilds_faiss_without_document_embedding(monkeypatch) -> None:
    embedding_model = FakeEmbeddingModel(fail_document_embedding=True)
    artifacts = StoredDocumentArtifacts(
        document_id="mysql-document",
        chunks=CHUNKS,
        embedding_matrix=MATRIX,
        needs_embedding_backfill=False,
    )
    captured = {}
    monkeypatch.setattr(main, "load_document_artifacts", lambda **_kwargs: artifacts)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(
        main,
        "load_or_backfill_document_embeddings",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("valid embeddings must not be backfilled")),
    )
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **kwargs: captured.update(kwargs) or "mysql-document",
    )
    _stub_question_pipeline(monkeypatch)

    response, _ = await _call_upload()

    cache_entry = main.document_cache[main._upload_cache_key(PDF_BYTES)]
    assert response.headers["X-Document-Cache"] == "MYSQL_HIT"
    assert cache_entry.faiss_index.ntotal == 1
    assert embedding_model.document_calls == []
    np.testing.assert_array_equal(captured["embedding_matrix"], MATRIX)


@pytest.mark.asyncio
async def test_full_miss_embeds_once_and_passes_complete_batch_to_persistence(monkeypatch) -> None:
    embedding_model = FakeEmbeddingModel()
    captured = {}
    monkeypatch.setattr(main, "load_document_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", lambda _pdf: CHUNKS)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **kwargs: captured.update(kwargs) or kwargs["proposed_document_id"],
    )
    _stub_question_pipeline(monkeypatch)

    response, result = await _call_upload(request_id=str(uuid.uuid4()))

    assert response.headers["X-Document-Cache"] == "FULL_MISS"
    assert len(embedding_model.document_calls) == 1
    assert embedding_model.document_calls[0] == ["passage: stored passage"]
    assert len(captured["question_results"]) == 1
    assert captured["chunks"] == CHUNKS
    np.testing.assert_array_equal(captured["embedding_matrix"], MATRIX)
    assert result.answers[0].status == "ok"
    assert main._upload_cache_key(PDF_BYTES) in main.document_cache


@pytest.mark.asyncio
async def test_legacy_mysql_hit_reembeds_and_backfills_once(monkeypatch) -> None:
    embedding_model = FakeEmbeddingModel()
    artifacts = StoredDocumentArtifacts(
        document_id="legacy-document",
        chunks=CHUNKS,
        embedding_matrix=None,
        needs_embedding_backfill=True,
    )
    backfills = []

    def fake_backfill(**kwargs):
        matrix = kwargs["embedding_generator"](CHUNKS)
        backfills.append(matrix)
        return StoredDocumentArtifacts(
            document_id="legacy-document",
            chunks=CHUNKS,
            embedding_matrix=matrix,
            needs_embedding_backfill=False,
        )

    monkeypatch.setattr(main, "load_document_artifacts", lambda **_kwargs: artifacts)
    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", lambda _pdf: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(main, "load_or_backfill_document_embeddings", fake_backfill)
    monkeypatch.setattr(main, "persist_upload_result_atomic", lambda **_kwargs: "legacy-document")
    _stub_question_pipeline(monkeypatch)

    response, _ = await _call_upload()

    assert response.headers["X-Document-Cache"] == "MYSQL_HIT"
    assert len(embedding_model.document_calls) == 1
    assert len(backfills) == 1
    np.testing.assert_array_equal(backfills[0], MATRIX)


@pytest.mark.asyncio
async def test_persistence_failure_still_returns_generated_answer(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_document_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", lambda _pdf: CHUNKS)
    monkeypatch.setattr(main, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    _stub_question_pipeline(monkeypatch)

    response, result = await _call_upload()

    assert response.headers["X-Document-Cache"] == "FULL_MISS"
    assert result.answers[0].answer == "A grounded answer."
    assert main._upload_cache_key(PDF_BYTES) not in main.document_cache


@pytest.mark.asyncio
async def test_request_retry_returns_stored_answer_without_loading_models(monkeypatch) -> None:
    request_id = str(uuid.uuid4())
    recovered = RecoveredUploadResult(
        document_id="recovered-document",
        response_payload={
            "answers": [
                {
                    "question": "What happened?",
                    "answer": "Previously committed answer.",
                    "status": "ok",
                    "sources": [],
                    "claim_verifications": [],
                }
            ]
        },
    )
    monkeypatch.setattr(main, "recover_upload_request", lambda **_kwargs: recovered)
    monkeypatch.setattr(
        main,
        "get_embedding_model",
        lambda: (_ for _ in ()).throw(AssertionError("models must not load during recovery")),
    )
    monkeypatch.setattr(
        main,
        "get_groq_client",
        lambda: (_ for _ in ()).throw(AssertionError("Groq must not run during recovery")),
    )

    response, result = await _call_upload(request_id=request_id)

    assert response.headers["X-Document-Cache"] == "RECOVERED"
    assert result.answers[0].answer == "Previously committed answer."


@pytest.mark.asyncio
async def test_request_id_with_different_input_returns_conflict(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "recover_upload_request",
        lambda **_kwargs: (_ for _ in ()).throw(RequestConflictError("request_id conflict")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload(request_id=str(uuid.uuid4()))

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_url_query_path_keeps_existing_response_behavior(monkeypatch) -> None:
    fake_index = main.build_faiss_index(MATRIX)
    embedding_model = FakeEmbeddingModel(fail_document_embedding=True)

    async def fake_cached_document(*_args, **_kwargs):
        return CHUNKS, fake_index

    async def fake_run_questions(*_args, **_kwargs):
        return main.QueryResponse(answers=[_processed_result().answer_item])

    monkeypatch.setattr(main, "_get_cached_document", fake_cached_document)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(main, "get_reranker_model", lambda: object())
    monkeypatch.setattr(main, "get_groq_client", lambda: object())
    monkeypatch.setattr(main, "persist_ingested_document_best_effort", lambda **_kwargs: "url-document")
    monkeypatch.setattr(main, "_run_questions", fake_run_questions)

    result = await main.run_query_pipeline(
        request=QueryRequest(
            documents="https://example.com/report.pdf",
            questions=["What happened?"],
        ),
        response=Response(),
        background_tasks=BackgroundTasks(),
        authorization=f"Bearer {main.EXPECTED_TOKEN}",
    )

    assert result.answers[0].answer == "A grounded answer."
