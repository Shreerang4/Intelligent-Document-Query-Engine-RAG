import asyncio
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, TypeVar

import faiss
import fitz
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, HttpUrl

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


class ChunkRecord(TypedDict):
    text: str
    page: int
    chunk_id: int


class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[Any]


class SourceReference(BaseModel):
    page: int
    chunk_id: int
    excerpt: str


class ClaimVerificationSource(BaseModel):
    page: int
    chunk_id: int
    excerpt: str


class ClaimVerificationItem(BaseModel):
    claim: str
    verdict: Literal["supported", "weakly_supported", "unsupported"]
    rationale: str
    sources: List[ClaimVerificationSource]


class AnswerItem(BaseModel):
    question: str
    answer: str
    status: str
    sources: List[SourceReference]
    claim_verifications: List[ClaimVerificationItem] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answers: List[AnswerItem]


app = FastAPI(title="Intelligent Document Query Engine", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPECTED_TOKEN = os.getenv("API_TOKEN")
if not EXPECTED_TOKEN:
    raise RuntimeError("API_TOKEN environment variable is not set.")

model_cache: Dict[str, object] = {}
MAX_QUESTIONS_PER_REQUEST = 10
MAX_PDF_BYTES = 15728640
HTTP_TIMEOUT_SECONDS = 30
RETRIEVAL_K_INITIAL = 8
RETRIEVAL_K_FINAL = 3
MAX_CONCURRENT_QUESTIONS = 4
DOCUMENT_CACHE_MAX_ITEMS = 8
DOCUMENT_CACHE_TTL_SECONDS = 3600
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
MAX_CLAIMS_PER_ANSWER = 5
CLAIM_VERIFICATION_K_FINAL = 3
CLAIM_VERIFICATION_FAILURE_MESSAGE = "Verification failed or evidence was insufficient."
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
_invalid_int_settings_warnings: set[tuple[str, str]] = set()
SourceModelT = TypeVar("SourceModelT", bound=BaseModel)
CLAIM_VERDICTS = {"supported", "weakly_supported", "unsupported"}


@dataclass
class DocumentCacheEntry:
    chunks: List[ChunkRecord]
    faiss_index: faiss.IndexFlatL2
    created_at: float
    last_accessed: float


document_cache: Dict[str, DocumentCacheEntry] = {}


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
    return _get_int_setting("MAX_PDF_BYTES", MAX_PDF_BYTES, min_value=1)


def get_http_timeout_seconds() -> int:
    return _get_int_setting("HTTP_TIMEOUT_SECONDS", HTTP_TIMEOUT_SECONDS, min_value=1)


def get_retrieval_k_initial() -> int:
    return _get_int_setting("RETRIEVAL_K_INITIAL", RETRIEVAL_K_INITIAL, min_value=1)


def get_retrieval_k_final() -> int:
    return _get_int_setting("RETRIEVAL_K_FINAL", RETRIEVAL_K_FINAL, min_value=1)


def get_max_concurrent_questions() -> int:
    return _get_int_setting("MAX_CONCURRENT_QUESTIONS", MAX_CONCURRENT_QUESTIONS, min_value=1)


def get_document_cache_max_items() -> int:
    return _get_int_setting("DOCUMENT_CACHE_MAX_ITEMS", DOCUMENT_CACHE_MAX_ITEMS, min_value=1)


def get_document_cache_ttl_seconds() -> int:
    return _get_int_setting("DOCUMENT_CACHE_TTL_SECONDS", DOCUMENT_CACHE_TTL_SECONDS, min_value=0)


def get_embedding_model_name() -> str:
    return _get_str_setting("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)


def get_reranker_model_name() -> str:
    return _get_str_setting("RERANKER_MODEL_NAME", RERANKER_MODEL_NAME)


def get_llm_model_name() -> str:
    return _get_str_setting("LLM_MODEL_NAME", LLM_MODEL_NAME)


def get_embedding_model() -> Any:
    model_name = get_embedding_model_name()
    cache_key = f"embedding_model:{model_name}"
    if cache_key not in model_cache:
        logger.info("Loading embedding model: %s", model_name)
        from langchain_huggingface import HuggingFaceEmbeddings

        model_cache[cache_key] = HuggingFaceEmbeddings(model_name=model_name)
    return model_cache[cache_key]


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


def _require_bearer_token(authorization: Optional[str]) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme != "Bearer" or token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")


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


def _url_cache_key(url: str) -> str:
    return f"url:{hashlib.sha256(url.encode('utf-8')).hexdigest()}"


def _upload_cache_key(pdf_bytes: bytes) -> str:
    return f"upload:{hashlib.sha256(pdf_bytes).hexdigest()}"


def _is_supported_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return "pdf" in content_type or filename.endswith(".pdf")


def _ascii_normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _is_board_coordinate_line(line: str) -> bool:
    compact = " ".join(line.lower().split())
    if re.fullmatch(r"(?:[a-h](?:\s+[a-h]){3,7})", compact):
        return True
    if re.fullmatch(r"(?:[1-8](?:\s+[1-8]){0,7})", compact):
        return True
    return False


def _line_has_language_content(line: str) -> bool:
    non_space = sum(1 for char in line if not char.isspace())
    alpha_chars = sum(1 for char in line if char.isalpha())

    if non_space == 0:
        return False
    if alpha_chars == 0:
        return False
    if alpha_chars < 2 and non_space < 12:
        return False
    if alpha_chars / max(non_space, 1) < 0.18 and alpha_chars < 12:
        return False
    return True


def _clean_extracted_line(line: str) -> str:
    cleaned = _ascii_normalize(line)
    cleaned = re.sub(r"[^\x20-\x7E]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""
    if _is_board_coordinate_line(cleaned):
        return ""
    if re.fullmatch(r"[1-8]", cleaned):
        return ""
    if not _line_has_language_content(cleaned):
        return ""

    return cleaned


def _clean_extracted_page_text(page_text: str) -> str:
    cleaned_lines: List[str] = []
    previous_line = ""

    for raw_line in page_text.splitlines():
        cleaned_line = _clean_extracted_line(raw_line)
        if not cleaned_line:
            continue
        if cleaned_line == previous_line:
            continue
        cleaned_lines.append(cleaned_line)
        previous_line = cleaned_line

    return "\n".join(cleaned_lines).strip()


def _is_low_quality_chunk(text: str) -> bool:
    non_space = sum(1 for char in text if not char.isspace())
    alpha_chars = sum(1 for char in text if char.isalpha())

    if non_space == 0:
        return True
    if alpha_chars == 0:
        return True
    if len(text) < 30 and alpha_chars < 12:
        return True
    if alpha_chars / max(non_space, 1) < 0.22 and alpha_chars < 80:
        return True
    return False


def _clean_generated_answer_text(text: str) -> str:
    cleaned = _ascii_normalize(text)
    cleaned = re.sub(r"[^\x20-\x7E\n]", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_and_chunk_pdf_bytes(pdf_bytes: bytes) -> List[ChunkRecord]:
    """Split PDF bytes into page-aware text chunks."""
    if len(pdf_bytes) > get_max_pdf_bytes():
        raise HTTPException(status_code=400, detail="PDF exceeds maximum allowed size.")

    chunk_records: List[ChunkRecord] = []
    chunk_id = 0

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                raw_page_text = page.get_text("text")
                page_text = _clean_extracted_page_text(raw_page_text)
                if not page_text or not page_text.strip():
                    continue

                for chunk_text in TEXT_SPLITTER.split_text(page_text):
                    normalized_text = chunk_text.strip()
                    if not normalized_text:
                        continue
                    if _is_low_quality_chunk(normalized_text):
                        continue
                    chunk_records.append(
                        {
                            "text": normalized_text,
                            "page": page_number,
                            "chunk_id": chunk_id,
                        }
                    )
                    chunk_id += 1
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Failed to parse PDF document.") from exc

    if not chunk_records:
        raise HTTPException(status_code=422, detail="No meaningful text found in the PDF.")

    logger.info("Document parsed into %s chunks.", len(chunk_records))
    return chunk_records


async def load_and_chunk_pdf(url: str) -> List[ChunkRecord]:
    """Download a PDF from a URL and split it into page-aware text chunks."""
    parsed_url = httpx.URL(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Document URL must use http or https.")

    logger.info("Downloading document from %s", url)
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
    return load_and_chunk_pdf_bytes(response.content)


def create_vector_store(
    chunks: List[ChunkRecord],
    embedding_model: Any,
) -> faiss.IndexFlatL2:
    """Create an in-memory FAISS index from structured chunk records."""
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks to process.")

    logger.info("Creating embeddings and building FAISS index...")
    chunk_embeddings = embedding_model.embed_documents([chunk["text"] for chunk in chunks])
    embedding_array = np.array(chunk_embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    logger.info("FAISS index created with %s vectors.", index.ntotal)
    return index


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

    question_embedding = np.array([embedding_model.embed_query(question)], dtype="float32")
    _, indices = faiss_index.search(question_embedding, k_initial)

    valid_indices = [index for index in indices[0] if index != -1]
    retrieved_chunks = [chunks[index] for index in valid_indices]
    if not retrieved_chunks:
        logger.info("No retrieved chunks for question: %r", question)
        return "", []

    if use_reranker:
        rerank_pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
        rerank_scores = reranker.predict(rerank_pairs)
        reranked = sorted(zip(retrieved_chunks, rerank_scores), key=lambda item: item[1], reverse=True)
        selected_chunks = [chunk for chunk, _ in reranked[:k_final]]
    else:
        selected_chunks = retrieved_chunks[:k_final]

    logger.info(
        "Retrieved chunk_ids for question %r: %s",
        question,
        [chunk["chunk_id"] for chunk in selected_chunks],
    )

    context = "\n\n".join(
        f"[Page {chunk['page']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in selected_chunks
    )
    return context, selected_chunks


def generate_answer(question: str, context: str, client: Groq) -> str:
    """Call the Groq LLM to synthesize a grounded answer."""
    logger.info("Generating answer for: '%s...'", question[:60])
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
        logger.error("LLM call failed: %s", exc)
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
        logger.warning("Claim extraction failed: %s", exc)
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


async def process_question(
    question: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
) -> AnswerItem:
    """Run retrieval and generation for a single question without failing the batch."""
    loop = asyncio.get_running_loop()
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
            return AnswerItem(
                question=question,
                answer="Information not found in the document.",
                status="no_context",
                sources=[],
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
            logger.exception("Claim verification failed for question: %s", question)

        return AnswerItem(
            question=question,
            answer=answer,
            status="ok",
            sources=_build_source_references(source_chunks),
            claim_verifications=claim_verifications,
        )
    except Exception:
        logger.exception("Failed to process question: %s", question)
        return AnswerItem(
            question=question,
            answer="Failed to process this question.",
            status="error",
            sources=[],
        )


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


def _set_document_cache_entry(
    cache_key: str,
    chunks: List[ChunkRecord],
    faiss_index: faiss.IndexFlatL2,
    now: Optional[float] = None,
) -> None:
    current_time = time.time() if now is None else now
    _evict_expired_document_cache_entries(current_time)
    max_items = get_document_cache_max_items()
    if cache_key not in document_cache:
        while len(document_cache) >= max_items:
            _evict_lru_document_cache_entry()

    document_cache[cache_key] = DocumentCacheEntry(
        chunks=chunks,
        faiss_index=faiss_index,
        created_at=current_time,
        last_accessed=current_time,
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
    cache_key: str,
    response: Response,
    chunk_loader: Callable[[], Awaitable[List[ChunkRecord]]],
    embedding_model: Any,
) -> Tuple[List[ChunkRecord], faiss.IndexFlatL2]:
    current_time = time.time()
    _evict_expired_document_cache_entries(current_time)

    cache_entry = document_cache.get(cache_key)
    if cache_entry is not None:
        cache_entry.last_accessed = current_time
        logger.info("Document cache hit for %s input.", cache_key.split(":", 1)[0])
        _set_cache_headers(response, "HIT")
        return cache_entry.chunks, cache_entry.faiss_index

    chunks = await chunk_loader()
    faiss_index = create_vector_store(chunks, embedding_model)
    _set_document_cache_entry(cache_key, chunks, faiss_index, now=current_time)
    _set_cache_headers(response, "MISS")
    logger.info("Document cached for %s input.", cache_key.split(":", 1)[0])
    return chunks, faiss_index


async def _run_questions(
    questions: List[str],
    faiss_index: faiss.IndexFlatL2,
    chunks: List[ChunkRecord],
    embedding_model: Any,
    reranker: Any,
    groq_client: Groq,
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
            )

    answers = await asyncio.gather(*(run_with_limit(question) for question in questions))
    return QueryResponse(answers=list(answers))


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_pipeline(
    request: QueryRequest,
    response: Response,
    authorization: Optional[str] = Header(None),
) -> QueryResponse:
    _require_bearer_token(authorization)
    questions = _normalize_questions(request.questions)

    url = str(request.documents)
    embedding_model = get_embedding_model()
    reranker = get_reranker_model()
    groq_client = get_groq_client()

    chunks, faiss_index = await _get_cached_document(
        _url_cache_key(url),
        response,
        lambda: load_and_chunk_pdf(url),
        embedding_model,
    )

    return await _run_questions(
        questions,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        groq_client,
    )


@app.post("/hackrx/upload-run", response_model=QueryResponse)
async def upload_query_pipeline(
    response: Response,
    file: UploadFile = File(...),
    questions_json: str = Form(...),
    authorization: Optional[str] = Header(None),
) -> QueryResponse:
    _require_bearer_token(authorization)
    questions = _parse_upload_questions_json(questions_json)

    if not _is_supported_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    try:
        pdf_bytes = await file.read()
    finally:
        await file.close()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > get_max_pdf_bytes():
        raise HTTPException(status_code=400, detail="PDF exceeds maximum allowed size.")

    cache_key = _upload_cache_key(pdf_bytes)
    current_time = time.time()
    _evict_expired_document_cache_entries(current_time)

    cache_entry = document_cache.get(cache_key)
    if cache_entry is not None:
        cache_entry.last_accessed = current_time
        _set_cache_headers(response, "HIT")
        logger.info("Document cache hit for upload input.")
        chunks = cache_entry.chunks
        faiss_index = cache_entry.faiss_index
        embedding_model = get_embedding_model()
    else:
        chunks = load_and_chunk_pdf_bytes(pdf_bytes)
        embedding_model = get_embedding_model()
        faiss_index = create_vector_store(chunks, embedding_model)
        _set_document_cache_entry(cache_key, chunks, faiss_index, now=current_time)
        _set_cache_headers(response, "MISS")
        logger.info("Document cached for upload input.")

    reranker = get_reranker_model()
    groq_client = get_groq_client()

    return await _run_questions(
        questions,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        groq_client,
    )


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
    if full_path.startswith("hackrx/") or full_path in {"hackrx", "health", "docs", "redoc", "openapi.json"}:
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