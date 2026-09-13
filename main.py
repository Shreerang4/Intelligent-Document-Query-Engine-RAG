import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

import faiss
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

from backend.app.schemas import (
    AnswerItem,
    ClaimVerificationItem,
    ClaimVerificationSource,
    HistoryCitationItem,
    HistoryCitationsResponse,
    HistoryDocumentItem,
    HistoryDocumentsResponse,
    HistoryQueriesResponse,
    HistoryQueryItem,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from backend.app.auth.router import router as auth_router
from backend.app.auth.config import get_auth_allowed_origins
from backend.app.auth.dependencies import get_authenticated_user_id
from backend.app.auth.http_errors import install_auth_http_error_handlers
from backend.app.security_headers import install_security_headers
from backend.app.rag.config import (
    DEFAULT_EMBEDDING_MODEL_NAME as SHARED_EMBEDDING_MODEL_NAME,
    DEFAULT_MAX_PDF_BYTES as SHARED_MAX_PDF_BYTES,
    get_embedding_model_name as shared_get_embedding_model_name,
    get_max_pdf_bytes as shared_get_max_pdf_bytes,
)
from backend.app.rag.embeddings import (
    create_chunk_embeddings as shared_create_chunk_embeddings,
    format_documents_for_embedding as shared_format_documents_for_embedding,
    format_query_for_embedding as shared_format_query_for_embedding,
    get_embedding_input_format_version as shared_get_embedding_input_format_version,
    get_embedding_model as shared_get_embedding_model,
    uses_e5_embedding_format as shared_uses_e5_embedding_format,
)
from backend.app.rag.ingestion import (
    ChunkRecord,
    InvalidPdfError,
    NoMeaningfulTextError,
    PdfTooLargeError,
    ascii_normalize,
    parse_and_chunk_pdf_bytes,
)
from persistence.document_artifacts import (
    ArtifactValidationError,
    StoredDocumentArtifacts,
    load_document_artifacts,
    load_or_backfill_document_embeddings,
)
from persistence.ingestion import persist_ingested_document_best_effort
from persistence.history import (
    list_user_document_queries,
    list_user_documents,
    list_user_query_citations,
)
from persistence.ownership import OwnedResourceNotFoundError
from persistence.query_history import persist_query_result_best_effort
from persistence.upload_results import (
    RecoveredUploadResult,
    RequestConflictError,
    persist_upload_result_atomic,
    recover_upload_request,
)

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / 'frontend' / 'dist'
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / 'index.html'
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / 'assets'


app = FastAPI(title="Intelligent Document Query Engine", version="2.3.0")
install_auth_http_error_handlers(app)
install_security_headers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(get_auth_allowed_origins()),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

model_cache: Dict[str, object] = {}
HISTORY_MAX_LIMIT = 100
MAX_QUESTIONS_PER_REQUEST = 10
MAX_PDF_BYTES = SHARED_MAX_PDF_BYTES
HTTP_TIMEOUT_SECONDS = 30
RETRIEVAL_K_INITIAL = 20
RETRIEVAL_K_FINAL = 8
MAX_CONCURRENT_QUESTIONS = 4
DOCUMENT_CACHE_MAX_ITEMS = 8
DOCUMENT_CACHE_TTL_SECONDS = 3600
EMBEDDING_MODEL_NAME = SHARED_EMBEDDING_MODEL_NAME
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
RETRIEVAL_MODE = "faiss_reranker"
HYBRID_E5_K_INITIAL = 30
HYBRID_BM25_K_INITIAL = 20
HYBRID_K_FINAL = 5
LLM_MODEL_NAME = "openai/gpt-oss-20b"
MAX_CLAIMS_PER_ANSWER = 5
CLAIM_VERIFICATION_K_FINAL = 3
CLAIM_VERIFICATION_FAILURE_MESSAGE = "Verification failed or evidence was insufficient."
_invalid_int_settings_warnings: set[tuple[str, str]] = set()
SourceModelT = TypeVar("SourceModelT", bound=BaseModel)
CLAIM_VERDICTS = {"supported", "weakly_supported", "unsupported"}


@dataclass(frozen=True)
class DocumentCacheKey:
    user_id: str
    resource_key: str


@dataclass
class DocumentCacheEntry:
    user_id: str
    chunks: List[ChunkRecord]
    faiss_index: faiss.IndexFlatL2
    created_at: float
    last_accessed: float
    document_id: Optional[str] = None


@dataclass
class ProcessedQuestionResult:
    answer_item: AnswerItem
    source_chunks: List[ChunkRecord]
    latency_ms: float

    @property
    def is_abstained(self) -> bool:
        return _is_non_informative_answer(self.answer_item.answer)


@dataclass
class BM25Index:
    tokenized_corpus: List[List[str]]
    document_frequencies: Counter[str]
    average_document_length: float


document_cache: Dict[DocumentCacheKey, DocumentCacheEntry] = {}


def _warn_invalid_int_setting(name: str, raw_value: str, default: int) -> None:
    warning_key = (name, raw_value)
    if warning_key in _invalid_int_settings_warnings:
        return
    _invalid_int_settings_warnings.add(warning_key)
    logger.warning("Invalid %s=%r; using default %s.", name, raw_value, default)


def _get_int_setting(name: str, default: int, *, min_value: Optional[int] = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        _warn_invalid_int_setting(name, raw_value, default)
        return default

    if min_value is not None and parsed_value < min_value:
        _warn_invalid_int_setting(name, raw_value, default)
        return default
    return parsed_value


def _get_str_setting(name: str, default: str) -> str:
    return os.getenv(name) or default


def get_max_pdf_bytes() -> int:
    return shared_get_max_pdf_bytes()


def get_http_timeout_seconds() -> int:
    return _get_int_setting("HTTP_TIMEOUT_SECONDS", HTTP_TIMEOUT_SECONDS, min_value=1)


def get_retrieval_k_initial() -> int:
    return _get_int_setting("RETRIEVAL_K_INITIAL", RETRIEVAL_K_INITIAL, min_value=1)


def get_retrieval_k_final() -> int:
    return _get_int_setting("RETRIEVAL_K_FINAL", RETRIEVAL_K_FINAL, min_value=1)


def get_retrieval_mode() -> str:
    mode = _get_str_setting("RETRIEVAL_MODE", RETRIEVAL_MODE).strip().lower()
    if mode == "e5":
        return "faiss_reranker"
    return mode


def get_hybrid_e5_k_initial() -> int:
    return _get_int_setting("HYBRID_E5_K_INITIAL", HYBRID_E5_K_INITIAL, min_value=1)


def get_hybrid_bm25_k_initial() -> int:
    return _get_int_setting("HYBRID_BM25_K_INITIAL", HYBRID_BM25_K_INITIAL, min_value=1)


def get_hybrid_k_final() -> int:
    return _get_int_setting("HYBRID_K_FINAL", HYBRID_K_FINAL, min_value=1)


def get_max_concurrent_questions() -> int:
    return _get_int_setting("MAX_CONCURRENT_QUESTIONS", MAX_CONCURRENT_QUESTIONS, min_value=1)


def get_document_cache_max_items() -> int:
    return _get_int_setting("DOCUMENT_CACHE_MAX_ITEMS", DOCUMENT_CACHE_MAX_ITEMS, min_value=1)


def get_document_cache_ttl_seconds() -> int:
    return _get_int_setting("DOCUMENT_CACHE_TTL_SECONDS", DOCUMENT_CACHE_TTL_SECONDS, min_value=0)


def get_embedding_model_name() -> str:
    return shared_get_embedding_model_name()


def uses_e5_embedding_format(model_name: Optional[str] = None) -> bool:
    """Return True when the selected embedder expects E5 query/passage prefixes."""
    return shared_uses_e5_embedding_format(model_name)


def get_embedding_input_format_version(model_name: Optional[str] = None) -> str:
    return shared_get_embedding_input_format_version(model_name)


def _format_documents_for_embedding(texts: List[str], model_name: Optional[str] = None) -> List[str]:
    return shared_format_documents_for_embedding(texts, model_name)


def _format_query_for_embedding(question: str, model_name: Optional[str] = None) -> str:
    return shared_format_query_for_embedding(question, model_name)


def _embedding_cache_namespace() -> str:
    model_name = get_embedding_model_name().strip()
    payload = {
        "embedding_model": model_name,
        "input_format": get_embedding_input_format_version(model_name),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"embedding:{digest[:16]}"


def get_reranker_model_name() -> str:
    return _get_str_setting("RERANKER_MODEL_NAME", RERANKER_MODEL_NAME)


def get_llm_model_name() -> str:
    return _get_str_setting("LLM_MODEL_NAME", LLM_MODEL_NAME)


def get_embedding_model() -> Any:
    return shared_get_embedding_model(model_cache)


def get_reranker_model() -> Any:
    model_name = get_reranker_model_name()
    cache_key = f"reranker:{model_name}"
    if cache_key not in model_cache:
        logger.info("Loading reranker model: %s", model_name)
        from sentence_transformers import CrossEncoder

        model_cache[cache_key] = CrossEncoder(model_name)
    return model_cache[cache_key]


def get_groq_client() -> Groq:
    if "groq_client" not in model_cache:
        model_cache["groq_client"] = Groq()
    return model_cache["groq_client"]  # type: ignore[return-value]


def _require_operational_api_token(authorization: Optional[str]) -> None:
    expected_token = os.getenv("API_TOKEN")
    if expected_token is None or not expected_token.strip():
        raise HTTPException(
            status_code=503,
            detail="Operational database health authentication is not configured.",
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")


def _clamp_history_limit(limit: int) -> int:
    return max(1, min(limit, HISTORY_MAX_LIMIT))


def _normalize_questions(questions: List[Any]) -> List[str]:
    if not isinstance(questions, list):
        raise HTTPException(status_code=400, detail="Questions must be provided as a list.")

    if not questions:
        raise HTTPException(status_code=400, detail="At least one question is required.")

    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_QUESTIONS_PER_REQUEST} questions allowed per request.",
        )

    normalized_questions: List[str] = []
    for question in questions:
        if not isinstance(question, str):
            raise HTTPException(status_code=400, detail="Each question must be a string.")

        normalized_question = question.strip()
        if not normalized_question:
            raise HTTPException(status_code=400, detail="Questions must not be empty.")
        normalized_questions.append(normalized_question)

    return normalized_questions


def _parse_upload_questions_json(questions_json: str) -> List[str]:
    try:
        parsed_questions = json.loads(questions_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="questions_json must be valid JSON.") from exc

    if not isinstance(parsed_questions, list):
        raise HTTPException(status_code=400, detail="questions_json must be a JSON array of strings.")

    return _normalize_questions(parsed_questions)


def _normalize_request_id(request_id: Optional[str]) -> Optional[str]:
    normalized = (request_id or "").strip()
    if not normalized:
        return None
    try:
        return str(uuid.UUID(normalized))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="request_id must be a valid UUID.") from exc


def _validate_pdf_response_headers(response: httpx.Response) -> None:
    content_type = response.headers.get("Content-Type")
    if content_type:
        content_type = content_type.lower()
        if "pdf" not in content_type and "octet-stream" not in content_type:
            raise HTTPException(status_code=400, detail="Document URL must point to a PDF file.")

    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            size_bytes = int(content_length)
        except ValueError:
            logger.warning("Ignoring invalid Content-Length header from upstream document response.")
        else:
            if size_bytes > get_max_pdf_bytes():
                raise HTTPException(status_code=400, detail="PDF exceeds maximum allowed size.")


def _normalize_excerpt(text: str, max_length: int = 280) -> str:
    return " ".join(text.split())[:max_length]


def _build_source_models(chunks: List[ChunkRecord], model_cls: type[SourceModelT]) -> List[SourceModelT]:
    return [
        model_cls(
            page=chunk["page"],
            chunk_id=chunk["chunk_id"],
            excerpt=_normalize_excerpt(chunk["text"]),
        )
        for chunk in chunks
    ]


def _build_source_references(chunks: List[ChunkRecord]) -> List[SourceReference]:
    return _build_source_models(chunks, SourceReference)


def _build_claim_verification_sources(chunks: List[ChunkRecord]) -> List[ClaimVerificationSource]:
    return _build_source_models(chunks, ClaimVerificationSource)


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _strip_code_fences(text)
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_claim_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)).strip()


def _deduplicate_strings(items: List[Any], *, limit: int) -> List[str]:
    seen: set[str] = set()
    normalized_items: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        normalized_item = _normalize_claim_text(item)
        if not normalized_item:
            continue
        dedupe_key = normalized_item.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_items.append(normalized_item)
        if len(normalized_items) >= limit:
            break
    return normalized_items


def _is_non_informative_answer(answer: str) -> bool:
    normalized_answer = " ".join(answer.split()).strip().lower().rstrip(".")
    return normalized_answer in {
        "",
        "information not found in the document",
        "failed to process this question",
    }


def _fallback_extract_claims(text: str) -> List[str]:
    sentence_like_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return _deduplicate_strings(sentence_like_parts, limit=MAX_CLAIMS_PER_ANSWER)


def _parse_claim_extraction_output(raw_content: str, answer: str) -> List[str]:
    payload = _extract_json_object(raw_content)
    if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        return _deduplicate_strings(payload["claims"], limit=MAX_CLAIMS_PER_ANSWER)
    return _fallback_extract_claims(answer)


def _default_verification_rationale(verdict: str) -> str:
    if verdict == "supported":
        return "The claim is directly stated in the retrieved evidence."
    if verdict == "weakly_supported":
        return "The evidence only partially or indirectly supports the claim."
    return "The retrieved evidence does not support the claim."


def _build_verification_failure_item(claim: str) -> ClaimVerificationItem:
    return ClaimVerificationItem(
        claim=claim,
        verdict="unsupported",
        rationale=CLAIM_VERIFICATION_FAILURE_MESSAGE,
        sources=[],
    )


def _parse_claim_verification_output(
    claim: str,
    raw_content: str,
    evidence_chunks: List[ChunkRecord],
) -> ClaimVerificationItem:
    payload = _extract_json_object(raw_content)
    if not isinstance(payload, dict):
        return _build_verification_failure_item(claim)

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in CLAIM_VERDICTS:
        return _build_verification_failure_item(claim)

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = _default_verification_rationale(verdict)

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in evidence_chunks}
    source_chunks: List[ChunkRecord] = []
    raw_chunk_ids = payload.get("use_chunk_ids", [])
    if not isinstance(raw_chunk_ids, list):
        raw_chunk_ids = []

    for item in raw_chunk_ids:
        if isinstance(item, bool):
            continue
        try:
            chunk_id = int(item)
        except (TypeError, ValueError):
            continue
        chunk = chunk_lookup.get(chunk_id)
        if chunk is not None and chunk not in source_chunks:
            source_chunks.append(chunk)

    return ClaimVerificationItem(
        claim=claim,
        verdict=verdict,
        rationale=" ".join(rationale.split()),
        sources=_build_claim_verification_sources(source_chunks),
    )


def _owned_document_cache_key(user_id: str, resource_key: str) -> DocumentCacheKey:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Document cache operations require a non-empty user_id.")
    return DocumentCacheKey(user_id=user_id, resource_key=resource_key)


def _url_cache_key(user_id: str, url: str) -> DocumentCacheKey:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _owned_document_cache_key(
        user_id,
        f"url:{_embedding_cache_namespace()}:{url_hash}",
    )


def _upload_cache_key(user_id: str, pdf_bytes: bytes) -> DocumentCacheKey:
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    return _owned_document_cache_key(
        user_id,
        f"upload:{_embedding_cache_namespace()}:{pdf_hash}",
    )


def _is_supported_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return "pdf" in content_type or filename.endswith(".pdf")


def _ascii_normalize(text: str) -> str:
    return ascii_normalize(text)


def _clean_generated_answer_text(text: str) -> str:
    cleaned = _ascii_normalize(text)
    cleaned = re.sub(r"[^\x20-\x7E\n]", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_and_chunk_pdf_bytes(pdf_bytes: bytes) -> List[ChunkRecord]:
    """Split PDF bytes into page-aware text chunks."""
    try:
        return parse_and_chunk_pdf_bytes(pdf_bytes)
    except PdfTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidPdfError as exc:
        raise HTTPException(status_code=422, detail="Failed to parse PDF document.") from exc
    except NoMeaningfulTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def load_and_chunk_pdf(url: str) -> List[ChunkRecord]:
    """Download a PDF from a URL and split it into page-aware text chunks."""
    parsed_url = httpx.URL(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Document URL must use http or https.")

    logger.info("Downloading URL document.")
    try:
        async with httpx.AsyncClient(
            timeout=get_http_timeout_seconds(),
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Document URL returned error {exc.response.status_code}",
        ) from exc

    _validate_pdf_response_headers(response)
    return await asyncio.to_thread(load_and_chunk_pdf_bytes, response.content)


def create_chunk_embeddings(
    chunks: List[ChunkRecord],
    embedding_model: Any,
) -> np.ndarray:
    """Create one contiguous float32 embedding row per chunk."""
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks to process.")
    return shared_create_chunk_embeddings(chunks, embedding_model)


def build_faiss_index(embedding_array: np.ndarray) -> faiss.IndexFlatL2:
    """Build the existing exact L2 index from validated stored or new vectors."""
    matrix = np.ascontiguousarray(embedding_array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid embeddings to index.")

    index = faiss.IndexFlatL2(int(matrix.shape[1]))
    index.add(matrix)
    return index


def create_vector_store(
    chunks: List[ChunkRecord],
    embedding_model: Any,
) -> faiss.IndexFlatL2:
    """Create an in-memory FAISS index from structured chunk records."""
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks to process.")

    logger.info("Creating embeddings and building FAISS index.")
    embedding_array = create_chunk_embeddings(chunks, embedding_model)
    index = build_faiss_index(embedding_array)
    logger.info("FAISS index created with %s vectors.", index.ntotal)
    return index


def _build_context(selected_chunks: List[ChunkRecord]) -> str:
    return "\n\n".join(
        f"[Page {chunk['page']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in selected_chunks
    )


def _bm25_tokenize(text: str) -> List[str]:
    return text.lower().split()


def _bm25_cache_key(chunks: List[ChunkRecord]) -> str:
    return f"bm25:{id(chunks)}:{len(chunks)}"


def _get_bm25_index(chunks: List[ChunkRecord]) -> BM25Index:
    cache_key = _bm25_cache_key(chunks)
    cached_index = model_cache.get(cache_key)
    if isinstance(cached_index, BM25Index):
        return cached_index

    tokenized_corpus = [_bm25_tokenize(chunk["text"]) for chunk in chunks]
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_corpus:
        document_frequencies.update(set(tokens))

    average_document_length = (
        sum(len(tokens) for tokens in tokenized_corpus) / len(tokenized_corpus)
        if tokenized_corpus
        else 0.0
    )
    bm25_index = BM25Index(
        tokenized_corpus=tokenized_corpus,
        document_frequencies=document_frequencies,
        average_document_length=average_document_length,
    )
    model_cache[cache_key] = bm25_index
    return bm25_index


def _bm25_rank_chunks(question: str, chunks: List[ChunkRecord], k: int) -> List[Tuple[int, float]]:
    bm25_index = _get_bm25_index(chunks)
    query_terms = _bm25_tokenize(question)
    if not query_terms or not bm25_index.tokenized_corpus:
        return []

    document_count = len(bm25_index.tokenized_corpus)
    k1 = 1.5
    b = 0.75
    scores: List[Tuple[int, float]] = []
    for index, tokens in enumerate(bm25_index.tokenized_corpus):
        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = bm25_index.document_frequencies.get(term, 0)
            idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / bm25_index.average_document_length
            )
            score += idf * (term_frequency * (k1 + 1)) / denominator
        scores.append((index, score))

    return sorted(scores, key=lambda item: item[1], reverse=True)[:k]


def _copy_chunk_with_retrieval_metadata(
    chunk: ChunkRecord,
    *,
    source: str,
    e5_rank: Optional[int] = None,
    bm25_rank: Optional[int] = None,
) -> ChunkRecord:
    enriched_chunk = dict(chunk)
    enriched_chunk["retrieval_sources"] = [source]
    if e5_rank is not None:
        enriched_chunk["e5_rank"] = e5_rank
    if bm25_rank is not None:
        enriched_chunk["bm25_rank"] = bm25_rank
    return enriched_chunk  # type: ignore[return-value]


def retrieve_hybrid_context(
    question: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
) -> Tuple[str, List[ChunkRecord]]:
    e5_k = get_hybrid_e5_k_initial()
    bm25_k = get_hybrid_bm25_k_initial()
    final_k = get_hybrid_k_final()

    model_name = get_embedding_model_name()
    formatted_question = _format_query_for_embedding(question, model_name)
    question_embedding = np.array([embedding_model.embed_query(formatted_question)], dtype="float32")
    _, indices = faiss_index.search(question_embedding, e5_k)

    merged_by_chunk_id: Dict[int, ChunkRecord] = {}
    for rank, index in enumerate((idx for idx in indices[0] if idx != -1), start=1):
        chunk = chunks[index]
        merged_by_chunk_id[chunk["chunk_id"]] = _copy_chunk_with_retrieval_metadata(
            chunk,
            source="e5",
            e5_rank=rank,
        )

    for rank, (index, _score) in enumerate(_bm25_rank_chunks(question, chunks, bm25_k), start=1):
        chunk = chunks[index]
        existing_chunk = merged_by_chunk_id.get(chunk["chunk_id"])
        if existing_chunk is None:
            merged_by_chunk_id[chunk["chunk_id"]] = _copy_chunk_with_retrieval_metadata(
                chunk,
                source="bm25",
                bm25_rank=rank,
            )
        else:
            sources = existing_chunk.setdefault("retrieval_sources", [])  # type: ignore[attr-defined]
            if "bm25" not in sources:
                sources.append("bm25")
            existing_chunk["bm25_rank"] = rank  # type: ignore[typeddict-unknown-key]

    retrieved_chunks = list(merged_by_chunk_id.values())
    if not retrieved_chunks:
        logger.info("Hybrid retrieval returned no chunks.")
        return "", []

    rerank_pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(rerank_pairs)
    reranked = sorted(zip(retrieved_chunks, rerank_scores), key=lambda item: item[1], reverse=True)
    selected_chunks: List[ChunkRecord] = []
    for chunk, score in reranked[:final_k]:
        chunk["reranker_score"] = float(score)  # type: ignore[typeddict-unknown-key]
        selected_chunks.append(chunk)

    logger.info(
        "Hybrid retrieval selected chunk_ids=%s.",
        [chunk["chunk_id"] for chunk in selected_chunks],
    )
    return _build_context(selected_chunks), selected_chunks


def retrieve_context(
    question: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    k_initial: Optional[int] = None,
    k_final: Optional[int] = None,
    use_reranker: bool = True,
) -> Tuple[str, List[ChunkRecord]]:
    """Retrieve relevant context and grounded source chunks for a question."""
    if k_initial is None:
        k_initial = get_retrieval_k_initial()
    if k_final is None:
        k_final = get_retrieval_k_final()

    if get_retrieval_mode() == "e5_bm25_reranker" and use_reranker:
        return retrieve_hybrid_context(question, faiss_index, chunks, embedding_model, reranker)

    model_name = get_embedding_model_name()
    formatted_question = _format_query_for_embedding(question, model_name)
    question_embedding = np.array([embedding_model.embed_query(formatted_question)], dtype="float32")
    _, indices = faiss_index.search(question_embedding, k_initial)

    valid_indices = [index for index in indices[0] if index != -1]
    retrieved_chunks = [chunks[index] for index in valid_indices]
    if not retrieved_chunks:
        logger.info("Retrieval returned no chunks.")
        return "", []

    if use_reranker:
        rerank_pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
        rerank_scores = reranker.predict(rerank_pairs)
        reranked = sorted(zip(retrieved_chunks, rerank_scores), key=lambda item: item[1], reverse=True)
        selected_chunks = [chunk for chunk, _ in reranked[:k_final]]
    else:
        selected_chunks = retrieved_chunks[:k_final]

    logger.info(
        "Retrieval selected chunk_ids=%s.",
        [chunk["chunk_id"] for chunk in selected_chunks],
    )

    return _build_context(selected_chunks), selected_chunks


def generate_answer(question: str, context: str, client: Groq) -> str:
    """Call the Groq LLM to synthesize a grounded answer."""
    logger.info("Generating grounded answer.")
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document analyst. Answer the user's question "
                        "using ONLY the provided context.\n\n"
                        "Rules:\n"
                        "1. Give the most grounded answer possible from the context.\n"
                        "2. Do NOT require exact wording. If the context gives enough evidence "
                        "to reasonably answer, answer it.\n"
                        "3. If the context partially supports an answer but not directly or completely, "
                        "say that clearly. Example style: "
                        "'The document does not state this directly, but the available evidence suggests ...'\n"
                        "4. For yes/no questions:\n"
                        "   - Answer 'Yes' if the context supports the claim.\n"
                        "   - Answer 'No' if the context clearly contradicts the claim or clearly indicates a different topic.\n"
                        "   - Use 'Information not found in the document.' only if the context is genuinely insufficient.\n"
                        "5. For topic questions such as 'Is the document about X?', if the context clearly indicates "
                        "another topic, answer 'No' and briefly state the actual topic.\n"
                        "6. Only return 'Information not found in the document.' when the evidence is near-zero or genuinely insufficient.\n"
                        "7. Do not invent facts beyond the context.\n"
                        "8. Do not reproduce extraction artifacts, symbols, board coordinates, or formatting junk unless absolutely necessary.\n"
                        "9. Keep the answer concise, natural, and evidence-grounded."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            model=get_llm_model_name(),
            timeout=90,
            temperature=0,
            top_p=1,
        )
    except Exception as exc:
        logger.error("LLM call failed error=%s.", type(exc).__name__)
        raise RuntimeError("LLM call failed.") from exc

    return _clean_generated_answer_text(_extract_response_text(response))


def extract_claims(answer: str, client: Groq) -> List[str]:
    """Extract a short list of atomic factual claims from an answer."""
    if _is_non_informative_answer(answer):
        return []

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract atomic factual claims from answers. "
                        "Return strict JSON only in the form "
                        '{"claims": ["claim 1", "claim 2"]}. '
                        f"Return at most {MAX_CLAIMS_PER_ANSWER} short, standalone, factual claims. "
                        'If the answer contains no factual claims, return {"claims": []}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Answer:\n{answer}",
                },
            ],
            model=get_llm_model_name(),
            timeout=60,
            temperature=0,
            top_p=1,
        )
        raw_content = _extract_response_text(response)
    except Exception as exc:
        logger.warning("Claim extraction failed error=%s.", type(exc).__name__)
        return _fallback_extract_claims(answer)

    return _parse_claim_extraction_output(raw_content, answer)


def verify_claim(
    claim: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    client: Groq,
) -> ClaimVerificationItem:
    """Verify a single claim against retrieved document evidence."""
    try:
        _, evidence_chunks = retrieve_context(
            claim,
            faiss_index,
            chunks,
            embedding_model,
            reranker,
            k_final=min(get_retrieval_k_final(), CLAIM_VERIFICATION_K_FINAL),
        )
    except Exception:
        logger.exception("Evidence retrieval failed during claim verification.")
        return _build_verification_failure_item(claim)

    if not evidence_chunks:
        return ClaimVerificationItem(
            claim=claim,
            verdict="unsupported",
            rationale="No relevant evidence was retrieved for this claim.",
            sources=[],
        )

    evidence_context = "\n\n".join(
        f"[Page {chunk['page']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in evidence_chunks
    )
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You verify a claim using only the provided document evidence. "
                        "Return strict JSON only with keys verdict, rationale, and use_chunk_ids. "
                        'Allowed verdict values are "supported", "weakly_supported", and "unsupported". '
                        "Mark a claim supported only when the evidence clearly states it, "
                        "weakly_supported when support is partial or indirect, and unsupported otherwise. "
                        "Use only chunk IDs that appear in the evidence context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{claim}\n\n"
                        f"Evidence:\n{evidence_context}\n\n"
                        'Return JSON like {"verdict":"supported","rationale":"Short explanation","use_chunk_ids":[1]}.'
                    ),
                },
            ],
            model=get_llm_model_name(),
            timeout=60,
            temperature=0,
            top_p=1,
        )
    except Exception:
        logger.exception("LLM verification failed for claim.")
        return _build_verification_failure_item(claim)

    return _parse_claim_verification_output(claim, _extract_response_text(response), evidence_chunks)


def verify_answer_claims(
    answer: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
) -> List[ClaimVerificationItem]:
    """Extract and verify atomic claims for a generated answer."""
    try:
        claims = extract_claims(answer, groq_client)
    except Exception:
        logger.exception("Claim extraction raised unexpectedly.")
        return []

    claim_verifications: List[ClaimVerificationItem] = []
    for claim in claims[:MAX_CLAIMS_PER_ANSWER]:
        try:
            claim_verifications.append(
                verify_claim(
                    claim,
                    faiss_index,
                    chunks,
                    embedding_model,
                    reranker,
                    groq_client,
                )
            )
        except Exception:
            logger.exception("Claim verification raised unexpectedly.")
            claim_verifications.append(_build_verification_failure_item(claim))

    return claim_verifications


async def _process_question_result(
    question: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
) -> ProcessedQuestionResult:
    loop = asyncio.get_running_loop()
    started_at = time.perf_counter()

    try:
        context, source_chunks = await loop.run_in_executor(
            None,
            retrieve_context,
            question,
            faiss_index,
            chunks,
            embedding_model,
            reranker,
        )
        if not context:
            answer_item = AnswerItem(
                question=question,
                answer="Information not found in the document.",
                status="no_context",
                sources=[],
            )
            return ProcessedQuestionResult(
                answer_item=answer_item,
                source_chunks=[],
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )

        answer = await loop.run_in_executor(None, generate_answer, question, context, groq_client)
        claim_verifications: List[ClaimVerificationItem] = []
        try:
            claim_verifications = await loop.run_in_executor(
                None,
                verify_answer_claims,
                answer,
                faiss_index,
                chunks,
                embedding_model,
                reranker,
                groq_client,
            )
        except Exception:
            logger.exception("Claim verification failed for a question.")

        answer_item = AnswerItem(
            question=question,
            answer=answer,
            status="ok",
            sources=_build_source_references(source_chunks),
            claim_verifications=claim_verifications,
        )
        return ProcessedQuestionResult(
            answer_item=answer_item,
            source_chunks=source_chunks,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
    except Exception:
        logger.exception("Failed to process a question.")
        answer_item = AnswerItem(
            question=question,
            answer="Failed to process this question.",
            status="error",
            sources=[],
        )
        return ProcessedQuestionResult(
            answer_item=answer_item,
            source_chunks=[],
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )


async def process_question(
    question: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
    user_id: str,
    document_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AnswerItem:
    """Run one question and retain the URL path's best-effort history behavior."""
    result = await _process_question_result(
        question,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        groq_client,
    )
    if background_tasks is not None and document_id is not None:
        background_tasks.add_task(
            persist_query_result_best_effort,
            user_id=user_id,
            document_id=document_id,
            question=result.answer_item.question,
            answer=result.answer_item.answer,
            status=result.answer_item.status,
            is_abstained=result.is_abstained,
            claim_verifications=result.answer_item.claim_verifications,
            sources=result.answer_item.sources,
            source_chunks=result.source_chunks,
            embedding_model=get_embedding_model_name(),
            retrieval_mode=get_retrieval_mode(),
            reranker_model=get_reranker_model_name(),
            k_initial=get_retrieval_k_initial(),
            k_final=get_retrieval_k_final(),
            latency_ms=result.latency_ms,
        )
    return result.answer_item


def _evict_expired_document_cache_entries(now: Optional[float] = None) -> None:
    current_time = time.time() if now is None else now
    ttl_seconds = get_document_cache_ttl_seconds()
    expired_keys = [
        cache_key
        for cache_key, entry in document_cache.items()
        if current_time - entry.created_at > ttl_seconds
    ]
    for cache_key in expired_keys:
        document_cache.pop(cache_key, None)


def _evict_lru_document_cache_entry() -> None:
    if not document_cache:
        return
    lru_cache_key = min(document_cache.items(), key=lambda item: item[1].last_accessed)[0]
    document_cache.pop(lru_cache_key, None)


def _require_cache_key_owner(user_id: str, cache_key: DocumentCacheKey) -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Document cache operations require a non-empty user_id.")
    if not isinstance(cache_key, DocumentCacheKey) or cache_key.user_id != user_id:
        raise ValueError("Document cache key does not belong to the supplied user.")


def _get_document_cache_entry(
    *,
    user_id: str,
    cache_key: DocumentCacheKey,
    now: Optional[float] = None,
) -> Optional[DocumentCacheEntry]:
    _require_cache_key_owner(user_id, cache_key)
    current_time = time.time() if now is None else now
    _evict_expired_document_cache_entries(current_time)
    cache_entry = document_cache.get(cache_key)
    if cache_entry is None:
        return None
    if cache_entry.user_id != user_id:
        raise RuntimeError("Document cache entry ownership does not match its cache key.")
    cache_entry.last_accessed = current_time
    return cache_entry


def _set_document_cache_entry(
    user_id: str,
    cache_key: DocumentCacheKey,
    chunks: List[ChunkRecord],
    faiss_index: faiss.IndexFlatL2,
    now: Optional[float] = None,
    document_id: Optional[str] = None,
) -> None:
    _require_cache_key_owner(user_id, cache_key)
    current_time = time.time() if now is None else now
    _evict_expired_document_cache_entries(current_time)
    max_items = get_document_cache_max_items()
    if cache_key not in document_cache:
        while len(document_cache) >= max_items:
            _evict_lru_document_cache_entry()

    document_cache[cache_key] = DocumentCacheEntry(
        user_id=user_id,
        chunks=chunks,
        faiss_index=faiss_index,
        created_at=current_time,
        last_accessed=current_time,
        document_id=document_id,
    )


def _set_cache_headers(response: Response, cache_status: str) -> None:
    response.headers["X-Document-Cache"] = cache_status
    response.headers["X-Cache-Entries"] = str(len(document_cache))


def _resolve_frontend_asset(relative_path: str) -> Optional[Path]:
    if not FRONTEND_DIST_DIR.is_dir():
        return None

    normalized_path = relative_path.strip("/")
    if not normalized_path:
        return FRONTEND_INDEX_FILE if FRONTEND_INDEX_FILE.is_file() else None

    candidate = (FRONTEND_DIST_DIR / normalized_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


async def _get_cached_document(
    user_id: str,
    cache_key: DocumentCacheKey,
    response: Response,
    chunk_loader: Callable[[], Awaitable[List[ChunkRecord]]],
    embedding_model: Any,
) -> Tuple[List[ChunkRecord], faiss.IndexFlatL2]:
    current_time = time.time()
    cache_entry = _get_document_cache_entry(
        user_id=user_id,
        cache_key=cache_key,
        now=current_time,
    )
    if cache_entry is not None:
        logger.info("Document cache hit for %s input.", cache_key.resource_key.split(":", 1)[0])
        _set_cache_headers(response, "HIT")
        return cache_entry.chunks, cache_entry.faiss_index

    chunks = await chunk_loader()
    faiss_index = await asyncio.to_thread(create_vector_store, chunks, embedding_model)
    _set_document_cache_entry(
        user_id,
        cache_key,
        chunks,
        faiss_index,
        now=current_time,
    )
    _set_cache_headers(response, "MISS")
    logger.info("Document cached for %s input.", cache_key.resource_key.split(":", 1)[0])
    return chunks, faiss_index


async def _run_question_results(
    questions: List[str],
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
) -> List[ProcessedQuestionResult]:
    concurrency_limit = get_max_concurrent_questions()
    logger.info("Processing %s questions with concurrency limit %s.", len(questions), concurrency_limit)
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_with_limit(question: str) -> ProcessedQuestionResult:
        async with semaphore:
            return await _process_question_result(
                question,
                faiss_index,
                chunks,
                embedding_model,
                reranker,
                groq_client,
            )

    return list(await asyncio.gather(*(run_with_limit(question) for question in questions)))


async def _run_questions(
    questions: List[str],
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
    user_id: str,
    document_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> QueryResponse:
    concurrency_limit = get_max_concurrent_questions()
    logger.info("Processing %s questions with concurrency limit %s.", len(questions), concurrency_limit)
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_with_limit(question: str) -> AnswerItem:
        async with semaphore:
            return await process_question(
                question,
                faiss_index,
                chunks,
                embedding_model,
                reranker,
                groq_client,
                user_id=user_id,
                document_id=document_id,
                background_tasks=background_tasks,
            )

    answers = await asyncio.gather(*(run_with_limit(question) for question in questions))
    return QueryResponse(answers=list(answers))


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_pipeline(
    request: QueryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_authenticated_user_id),
) -> QueryResponse:
    questions = _normalize_questions(request.questions)

    url = str(request.documents)
    embedding_model = await asyncio.to_thread(get_embedding_model)
    reranker = await asyncio.to_thread(get_reranker_model)
    groq_client = await asyncio.to_thread(get_groq_client)

    cache_key = _url_cache_key(user_id, url)
    chunks, faiss_index = await _get_cached_document(
        user_id,
        cache_key,
        response,
        lambda: load_and_chunk_pdf(url),
        embedding_model,
    )
    document_id = await asyncio.to_thread(
        persist_ingested_document_best_effort,
        user_id=user_id,
        source_type="url",
        source_url=url,
        cache_key=cache_key.resource_key,
        chunks=chunks,
        embedding_model=get_embedding_model_name(),
        embedding_format=get_embedding_input_format_version(),
        retrieval_mode=get_retrieval_mode(),
        reranker_model=get_reranker_model_name(),
        k_initial=get_retrieval_k_initial(),
        k_final=get_retrieval_k_final(),
    )

    return await _run_questions(
        questions,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        groq_client,
        user_id=user_id,
        document_id=document_id,
        background_tasks=background_tasks,
    )


@app.post("/hackrx/upload-run", response_model=QueryResponse)
async def upload_query_pipeline(
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    questions_json: str = Form(...),
    request_id: Optional[str] = Form(None),
    user_id: str = Depends(get_authenticated_user_id),
) -> QueryResponse:
    questions = _parse_upload_questions_json(questions_json)
    normalized_request_id = _normalize_request_id(request_id)
    _ = background_tasks

    if not _is_supported_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    filename = file.filename
    try:
        pdf_bytes = await file.read()
    finally:
        await file.close()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > get_max_pdf_bytes():
        raise HTTPException(status_code=400, detail="PDF exceeds maximum allowed size.")

    source_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if normalized_request_id is not None:
        try:
            recovered = await asyncio.to_thread(
                recover_upload_request,
                user_id=user_id,
                request_id=normalized_request_id,
                source_hash=source_hash,
                questions=questions,
            )
        except RequestConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning(
                "Upload recovery lookup failed request_id=%s error=%s.",
                normalized_request_id,
                type(exc).__name__,
            )
            recovered = None
        if recovered is not None:
            _set_cache_headers(response, "RECOVERED")
            logger.info(
                "Upload request recovered request_id=%s document_id=%s status=RECOVERED.",
                normalized_request_id,
                recovered.document_id,
            )
            return QueryResponse.model_validate(recovered.response_payload)

    cache_key = _upload_cache_key(user_id, pdf_bytes)
    current_time = time.time()
    cache_entry = _get_document_cache_entry(
        user_id=user_id,
        cache_key=cache_key,
        now=current_time,
    )
    embedding_matrix: Optional[np.ndarray] = None
    cache_status: str
    should_cache_after_persistence = False
    if cache_entry is not None:
        cache_entry.last_accessed = current_time
        cache_status = "RAM_HIT"
        chunks = cache_entry.chunks
        faiss_index = cache_entry.faiss_index
        proposed_document_id = cache_entry.document_id or str(uuid.uuid4())
    else:
        artifacts: Optional[StoredDocumentArtifacts]
        try:
            artifacts = await asyncio.to_thread(
                load_document_artifacts,
                user_id=user_id,
                source_hash=source_hash,
                embedding_model=get_embedding_model_name(),
                embedding_format=get_embedding_input_format_version(),
            )
        except ArtifactValidationError as exc:
            logger.warning("Stored document artifacts are unusable error=%s.", type(exc).__name__)
            artifacts = None
        except Exception as exc:
            logger.warning("Stored document lookup failed safely error=%s.", type(exc).__name__)
            artifacts = None

        if artifacts is not None and artifacts.chunks:
            cache_status = "MYSQL_HIT"
            chunks = artifacts.chunks  # type: ignore[assignment]
            proposed_document_id = artifacts.document_id
            embedding_matrix = artifacts.embedding_matrix
            embedding_model = await asyncio.to_thread(get_embedding_model)
            if embedding_matrix is None:
                generated_backfill_chunks: Optional[List[ChunkRecord]] = None
                generated_backfill_matrix: Optional[np.ndarray] = None

                def generate_backfill_embeddings(
                    stored_chunks: List[ChunkRecord],
                ) -> np.ndarray:
                    nonlocal generated_backfill_chunks, generated_backfill_matrix
                    generated_backfill_chunks = stored_chunks
                    generated_backfill_matrix = create_chunk_embeddings(
                        stored_chunks,
                        embedding_model,
                    )
                    return generated_backfill_matrix

                try:
                    artifacts = await asyncio.to_thread(
                        load_or_backfill_document_embeddings,
                        user_id=user_id,
                        document_id=artifacts.document_id,
                        source_hash=source_hash,
                        embedding_model=get_embedding_model_name(),
                        embedding_format=get_embedding_input_format_version(),
                        embedding_generator=generate_backfill_embeddings,
                    )
                    chunks = artifacts.chunks  # type: ignore[assignment]
                    embedding_matrix = artifacts.embedding_matrix
                except Exception as exc:
                    logger.warning(
                        "Embedding backfill failed document_id=%s error=%s.",
                        artifacts.document_id,
                        type(exc).__name__,
                    )
                    if generated_backfill_matrix is not None and generated_backfill_chunks is not None:
                        chunks = generated_backfill_chunks
                        embedding_matrix = generated_backfill_matrix
                    else:
                        embedding_matrix = await asyncio.to_thread(
                            create_chunk_embeddings,
                            chunks,
                            embedding_model,
                        )
            faiss_index = await asyncio.to_thread(build_faiss_index, embedding_matrix)
            _set_document_cache_entry(
                user_id,
                cache_key,
                chunks,
                faiss_index,
                now=current_time,
                document_id=proposed_document_id,
            )
        else:
            cache_status = "FULL_MISS"
            chunks = await asyncio.to_thread(load_and_chunk_pdf_bytes, pdf_bytes)
            embedding_model = await asyncio.to_thread(get_embedding_model)
            embedding_matrix = await asyncio.to_thread(
                create_chunk_embeddings,
                chunks,
                embedding_model,
            )
            faiss_index = await asyncio.to_thread(build_faiss_index, embedding_matrix)
            proposed_document_id = str(uuid.uuid4())
            should_cache_after_persistence = True

    _set_cache_headers(response, cache_status)
    logger.info(
        "Upload cache flow request_id=%s document_id=%s status=%s.",
        normalized_request_id,
        proposed_document_id,
        cache_status,
    )

    if cache_entry is not None:
        embedding_model = await asyncio.to_thread(get_embedding_model)
    reranker = await asyncio.to_thread(get_reranker_model)
    groq_client = await asyncio.to_thread(get_groq_client)
    processed_results = await _run_question_results(
        questions,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        groq_client,
    )
    generated_response = QueryResponse(answers=[result.answer_item for result in processed_results])

    persistence_succeeded = False
    actual_document_id = proposed_document_id
    try:
        actual_document_id = await asyncio.to_thread(
            persist_upload_result_atomic,
            user_id=user_id,
            proposed_document_id=proposed_document_id,
            source_hash=source_hash,
            filename=filename,
            cache_key=cache_key.resource_key,
            chunks=chunks,
            embedding_matrix=embedding_matrix,
            question_results=processed_results,
            request_id=normalized_request_id,
            embedding_model=get_embedding_model_name(),
            embedding_format=get_embedding_input_format_version(),
            retrieval_mode=get_retrieval_mode(),
            reranker_model=get_reranker_model_name(),
            k_initial=get_retrieval_k_initial(),
            k_final=get_retrieval_k_final(),
        )
        persistence_succeeded = True
        logger.info(
            "Upload persistence request_id=%s document_id=%s status=SUCCESS.",
            normalized_request_id,
            actual_document_id,
        )
    except Exception as exc:
        if normalized_request_id is not None:
            try:
                recovered_after_race = await asyncio.to_thread(
                    recover_upload_request,
                    user_id=user_id,
                    request_id=normalized_request_id,
                    source_hash=source_hash,
                    questions=questions,
                )
            except RequestConflictError as conflict_exc:
                raise HTTPException(status_code=409, detail=str(conflict_exc)) from conflict_exc
            except Exception:
                recovered_after_race = None
            if recovered_after_race is not None:
                _set_document_cache_entry(
                    user_id,
                    cache_key,
                    chunks,
                    faiss_index,
                    document_id=recovered_after_race.document_id,
                )
                _set_cache_headers(response, "RECOVERED")
                return QueryResponse.model_validate(recovered_after_race.response_payload)

        logger.warning(
            "Upload persistence request_id=%s document_id=%s status=FAILED error=%s.",
            normalized_request_id,
            proposed_document_id,
            type(exc).__name__,
        )

    if persistence_succeeded:
        if should_cache_after_persistence:
            _set_document_cache_entry(
                user_id,
                cache_key,
                chunks,
                faiss_index,
                document_id=actual_document_id,
            )
        elif cache_entry is not None:
            cache_entry.document_id = actual_document_id
        _set_cache_headers(response, cache_status)

    return generated_response


@app.get("/history/documents", response_model=HistoryDocumentsResponse)
async def list_history_documents(
    limit: int = HISTORY_MAX_LIMIT,
    user_id: str = Depends(get_authenticated_user_id),
) -> HistoryDocumentsResponse:
    try:
        rows = await asyncio.to_thread(
            list_user_documents,
            user_id=user_id,
            limit=_clamp_history_limit(limit),
        )

        return HistoryDocumentsResponse(
            documents=[
                HistoryDocumentItem(
                    id=document.id,
                    filename=document.filename,
                    source_type=document.source_type,
                    source_url=document.source_url,
                    status=document.status,
                    created_at=document.created_at,
                    chunk_count=document.chunk_count,
                    query_count=document.query_count,
                )
                for document in rows
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("History document list failed safely: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="History is unavailable.") from exc


@app.get("/history/documents/{document_id}/queries", response_model=HistoryQueriesResponse)
async def list_history_document_queries(
    document_id: str,
    limit: int = HISTORY_MAX_LIMIT,
    user_id: str = Depends(get_authenticated_user_id),
) -> HistoryQueriesResponse:
    try:
        rows = await asyncio.to_thread(
            list_user_document_queries,
            user_id=user_id,
            document_id=document_id,
            limit=_clamp_history_limit(limit),
        )

        return HistoryQueriesResponse(
            queries=[
                HistoryQueryItem(
                    id=query.id,
                    question=query.question,
                    answer=query.answer,
                    is_abstained=query.is_abstained,
                    status=query.status,
                    latency_ms=query.latency_ms,
                    created_at=query.created_at,
                )
                for query in rows
            ]
        )
    except OwnedResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except Exception as exc:
        logger.warning("History query list failed safely: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="History is unavailable.") from exc


@app.get("/history/queries/{query_id}/citations", response_model=HistoryCitationsResponse)
async def list_history_query_citations(
    query_id: str,
    limit: int = HISTORY_MAX_LIMIT,
    user_id: str = Depends(get_authenticated_user_id),
) -> HistoryCitationsResponse:
    try:
        rows = await asyncio.to_thread(
            list_user_query_citations,
            user_id=user_id,
            query_id=query_id,
            limit=_clamp_history_limit(limit),
        )

        return HistoryCitationsResponse(
            citations=[
                HistoryCitationItem(
                    rank=citation.rank,
                    page_number=citation.page_number,
                    excerpt=citation.excerpt,
                    chunk_id=citation.chunk_id,
                )
                for citation in rows
            ]
        )
    except OwnedResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Query not found.") from exc
    except Exception as exc:
        logger.warning("History citation list failed safely: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="History is unavailable.") from exc


@app.get("/health/db")
async def database_health_check(authorization: Optional[str] = Header(None)) -> Dict[str, object]:
    _require_operational_api_token(authorization)

    try:
        from sqlalchemy import text

        from persistence.db import SessionLocal

        def check_database() -> None:
            with SessionLocal() as session:
                session.execute(text("SELECT 1"))

        await asyncio.to_thread(check_database)

        return {"database": "ok"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Database health check failed safely: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Database health check failed.") from exc


@app.get("/health")
async def health_check() -> Dict[str, object]:
    _evict_expired_document_cache_entries()
    return {
        "status": "healthy",
        "version": "2.3.0",
        "cache_entries": len(document_cache),
        "embedding_model_loaded": any(key.startswith("embedding_model:") for key in model_cache),
        "reranker_loaded": any(key.startswith("reranker:") for key in model_cache),
        "groq_client_loaded": "groq_client" in model_cache,
    }
if FRONTEND_ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="frontend-assets")


@app.get("/", include_in_schema=False)
async def serve_frontend_root() -> FileResponse:
    index_file = _resolve_frontend_asset("")
    if index_file is None:
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(index_file)


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_spa(full_path: str) -> FileResponse:
    if (
        full_path.startswith("auth/")
        or full_path.startswith("hackrx/")
        or full_path.startswith("history/")
        or full_path in {"auth", "hackrx", "history", "health", "docs", "redoc", "openapi.json"}
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    asset_file = _resolve_frontend_asset(full_path)
    if asset_file is not None:
        return FileResponse(asset_file)

    if Path(full_path).suffix:
        raise HTTPException(status_code=404, detail="Not Found")

    index_file = _resolve_frontend_asset("")
    if index_file is None:
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(index_file)
