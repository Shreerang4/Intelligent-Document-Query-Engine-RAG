from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException, Response, UploadFile
from starlette.datastructures import Headers

import main
from backend.app.schemas import AnswerItem, QueryRequest
from persistence.document_artifacts import StoredDocumentArtifacts
from persistence.upload_results import RecoveredUploadResult, RequestConflictError


PDF_BYTES = b"%PDF-1.7 test upload bytes"
CHUNKS = [{"text": "stored passage", "page": 1, "chunk_id": 0}]
MATRIX = np.ascontiguousarray(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
USER_A_ID = "00000000-0000-4000-8000-00000000000a"
USER_B_ID = "00000000-0000-4000-8000-00000000000b"


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


async def _call_upload(
    *,
    request_id: str | None = None,
    user_id: str = USER_A_ID,
) -> tuple[Response, object]:
    response = Response()
    result = await main.upload_query_pipeline(
        response=response,
        background_tasks=BackgroundTasks(),
        file=_upload_file(),
        questions_json=json.dumps(["What happened?"]),
        request_id=request_id,
        user_id=user_id,
    )
    return response, result


@pytest.mark.asyncio
async def test_ram_hit_does_not_load_database_artifacts(monkeypatch) -> None:
    index = main.build_faiss_index(MATRIX)
    cache_key = main._upload_cache_key(USER_A_ID, PDF_BYTES)
    main._set_document_cache_entry(
        USER_A_ID,
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

    cache_entry = main.document_cache[main._upload_cache_key(USER_A_ID, PDF_BYTES)]
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
    assert main._upload_cache_key(USER_A_ID, PDF_BYTES) in main.document_cache


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
    assert main._upload_cache_key(USER_A_ID, PDF_BYTES) not in main.document_cache


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
    captured_cache = {}
    captured_persistence = {}
    captured_questions = {}

    async def fake_cached_document(user_id, cache_key, *_args, **_kwargs):
        captured_cache.update(user_id=user_id, cache_key=cache_key)
        return CHUNKS, fake_index

    async def fake_run_questions(*_args, **kwargs):
        captured_questions.update(kwargs)
        return main.QueryResponse(answers=[_processed_result().answer_item])

    monkeypatch.setattr(main, "_get_cached_document", fake_cached_document)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(main, "get_reranker_model", lambda: object())
    monkeypatch.setattr(main, "get_groq_client", lambda: object())
    monkeypatch.setattr(
        main,
        "persist_ingested_document_best_effort",
        lambda **kwargs: captured_persistence.update(kwargs) or "url-document",
    )
    monkeypatch.setattr(main, "_run_questions", fake_run_questions)

    result = await main.run_query_pipeline(
        request=QueryRequest(
            documents="https://example.com/report.pdf",
            questions=["What happened?"],
        ),
        response=Response(),
        background_tasks=BackgroundTasks(),
        user_id=USER_A_ID,
    )

    assert result.answers[0].answer == "A grounded answer."
    assert captured_cache["user_id"] == USER_A_ID
    assert captured_cache["cache_key"].user_id == USER_A_ID
    assert captured_persistence["cache_key"] == captured_cache["cache_key"].resource_key
    assert captured_persistence["user_id"] == USER_A_ID
    assert captured_questions["user_id"] == USER_A_ID


@pytest.mark.asyncio
async def test_same_pdf_cache_is_isolated_between_users_and_reused_within_user(
    monkeypatch,
) -> None:
    artifact_lookups: list[str] = []
    persisted: list[dict] = []
    extracted_chunks: list[list[dict]] = []
    embedding_model = FakeEmbeddingModel()

    def fresh_chunks(_pdf_bytes):
        chunks = [dict(CHUNKS[0])]
        extracted_chunks.append(chunks)
        return chunks

    def load_artifacts(**kwargs):
        artifact_lookups.append(kwargs["user_id"])
        return None

    monkeypatch.setattr(main, "load_document_artifacts", load_artifacts)
    monkeypatch.setattr(main, "load_and_chunk_pdf_bytes", fresh_chunks)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **kwargs: persisted.append(kwargs) or f"{kwargs['user_id']}-document",
    )
    _stub_question_pipeline(monkeypatch)

    response_a_first, _ = await _call_upload(user_id=USER_A_ID)
    response_b, _ = await _call_upload(user_id=USER_B_ID)
    response_a_second, _ = await _call_upload(user_id=USER_A_ID)

    key_a = main._upload_cache_key(USER_A_ID, PDF_BYTES)
    key_b = main._upload_cache_key(USER_B_ID, PDF_BYTES)
    entry_a = main.document_cache[key_a]
    entry_b = main.document_cache[key_b]

    assert key_a != key_b
    assert key_a.resource_key == key_b.resource_key
    assert response_a_first.headers["X-Document-Cache"] == "FULL_MISS"
    assert response_b.headers["X-Document-Cache"] == "FULL_MISS"
    assert response_a_second.headers["X-Document-Cache"] == "RAM_HIT"
    assert artifact_lookups == [USER_A_ID, USER_B_ID]
    assert len(extracted_chunks) == 2
    assert entry_a.user_id == USER_A_ID
    assert entry_b.user_id == USER_B_ID
    assert entry_a.document_id == f"{USER_A_ID}-document"
    assert entry_b.document_id == f"{USER_B_ID}-document"
    assert entry_a.document_id != entry_b.document_id
    assert entry_a.chunks is not entry_b.chunks
    assert entry_a.faiss_index is not entry_b.faiss_index
    assert [call["user_id"] for call in persisted] == [USER_A_ID, USER_B_ID, USER_A_ID]


@pytest.mark.asyncio
async def test_same_url_uses_separate_user_cache_entries() -> None:
    url = "https://example.com/shared.pdf"
    embedding_model = FakeEmbeddingModel()
    load_counts = {"user-a": 0, "user-b": 0}

    async def load_for(user_id: str):
        load_counts[user_id] += 1
        return [
            {
                "text": f"{user_id} in-memory chunks",
                "page": 1,
                "chunk_id": 0,
            }
        ]

    response_a_first = Response()
    chunks_a, index_a = await main._get_cached_document(
        "user-a",
        main._url_cache_key("user-a", url),
        response_a_first,
        lambda: load_for("user-a"),
        embedding_model,
    )
    response_b = Response()
    chunks_b, index_b = await main._get_cached_document(
        "user-b",
        main._url_cache_key("user-b", url),
        response_b,
        lambda: load_for("user-b"),
        embedding_model,
    )
    response_a_second = Response()
    repeated_chunks_a, repeated_index_a = await main._get_cached_document(
        "user-a",
        main._url_cache_key("user-a", url),
        response_a_second,
        lambda: (_ for _ in ()).throw(AssertionError("A should receive its RAM hit")),
        embedding_model,
    )

    assert response_a_first.headers["X-Document-Cache"] == "MISS"
    assert response_b.headers["X-Document-Cache"] == "MISS"
    assert response_a_second.headers["X-Document-Cache"] == "HIT"
    assert load_counts == {"user-a": 1, "user-b": 1}
    assert chunks_a[0]["text"] == "user-a in-memory chunks"
    assert chunks_b[0]["text"] == "user-b in-memory chunks"
    assert index_a is not index_b
    assert repeated_chunks_a is chunks_a
    assert repeated_index_a is index_a


@pytest.mark.asyncio
async def test_mysql_recovery_installs_only_the_supplied_users_cache_entry(
    monkeypatch,
) -> None:
    artifact_lookups: list[str] = []
    artifacts_by_user = {
        USER_A_ID: StoredDocumentArtifacts(
            document_id="mysql-document-a",
            chunks=[{"text": "A persisted passage", "page": 1, "chunk_id": 0}],
            embedding_matrix=MATRIX,
            needs_embedding_backfill=False,
        ),
        USER_B_ID: StoredDocumentArtifacts(
            document_id="mysql-document-b",
            chunks=[{"text": "B persisted passage", "page": 1, "chunk_id": 0}],
            embedding_matrix=MATRIX,
            needs_embedding_backfill=False,
        ),
    }

    def load_artifacts(**kwargs):
        user_id = kwargs["user_id"]
        artifact_lookups.append(user_id)
        return artifacts_by_user[user_id]

    monkeypatch.setattr(main, "load_document_artifacts", load_artifacts)
    monkeypatch.setattr(
        main,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(fail_document_embedding=True),
    )
    monkeypatch.setattr(
        main,
        "persist_upload_result_atomic",
        lambda **kwargs: artifacts_by_user[kwargs["user_id"]].document_id,
    )
    _stub_question_pipeline(monkeypatch)

    response_a, _ = await _call_upload(user_id=USER_A_ID)
    response_b, _ = await _call_upload(user_id=USER_B_ID)

    entry_a = main.document_cache[main._upload_cache_key(USER_A_ID, PDF_BYTES)]
    entry_b = main.document_cache[main._upload_cache_key(USER_B_ID, PDF_BYTES)]
    assert response_a.headers["X-Document-Cache"] == "MYSQL_HIT"
    assert response_b.headers["X-Document-Cache"] == "MYSQL_HIT"
    assert artifact_lookups == [USER_A_ID, USER_B_ID]
    assert entry_a.user_id == USER_A_ID
    assert entry_b.user_id == USER_B_ID
    assert entry_a.document_id == "mysql-document-a"
    assert entry_b.document_id == "mysql-document-b"
    assert entry_a.chunks[0]["text"] == "A persisted passage"
    assert entry_b.chunks[0]["text"] == "B persisted passage"
    assert entry_a.faiss_index is not entry_b.faiss_index


@pytest.mark.asyncio
async def test_background_query_persistence_keeps_the_captured_user_id(monkeypatch) -> None:
    writes = []
    background_tasks = BackgroundTasks()

    async def fake_process(*_args, **_kwargs):
        return _processed_result("Background question?")

    monkeypatch.setattr(main, "_process_question_result", fake_process)
    monkeypatch.setattr(
        main,
        "persist_query_result_best_effort",
        lambda **kwargs: writes.append(kwargs),
    )

    await main.process_question(
        "Background question?",
        main.build_faiss_index(MATRIX),
        CHUNKS,
        FakeEmbeddingModel(),
        object(),
        object(),
        user_id="user-a",
        document_id="document-a",
        background_tasks=background_tasks,
    )
    await background_tasks()

    assert len(writes) == 1
    assert writes[0]["user_id"] == "user-a"
    assert writes[0]["document_id"] == "document-a"


def test_cache_helpers_reject_cross_user_keys_and_inconsistent_entries() -> None:
    index = main.build_faiss_index(MATRIX)
    key_a = main._upload_cache_key("user-a", PDF_BYTES)
    key_b = main._upload_cache_key("user-b", PDF_BYTES)

    with pytest.raises(ValueError, match="does not belong"):
        main._get_document_cache_entry(user_id="user-b", cache_key=key_a)
    with pytest.raises(ValueError, match="does not belong"):
        main._set_document_cache_entry("user-b", key_a, CHUNKS, index)

    main.document_cache[key_b] = main.DocumentCacheEntry(
        user_id="user-a",
        chunks=CHUNKS,
        faiss_index=index,
        created_at=0.0,
        last_accessed=0.0,
        document_id="document-a",
    )
    with pytest.raises(RuntimeError, match="ownership"):
        main._get_document_cache_entry(
            user_id="user-b",
            cache_key=key_b,
            now=0.0,
        )
