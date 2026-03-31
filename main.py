import asyncio
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict

import faiss
import fitz
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Response
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, HttpUrl

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    text: str
    page: int
    chunk_id: int


class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class SourceReference(BaseModel):
    page: int
    chunk_id: int
    excerpt: str


class AnswerItem(BaseModel):
    question: str
    answer: str
    status: str
    sources: List[SourceReference]


class QueryResponse(BaseModel):
    answers: List[AnswerItem]


app = FastAPI(title="Intelligent Document Query Engine", version="2.1.0")

EXPECTED_TOKEN = os.getenv("API_TOKEN")
if not EXPECTED_TOKEN:
    raise RuntimeError("API_TOKEN environment variable is not set.")

model_cache: Dict[str, object] = {}


def get_embedding_model() -> Any:
    if "embedding_model" not in model_cache:
        logger.info("Loading embedding model for the first time...")
        from langchain_huggingface import HuggingFaceEmbeddings

        model_cache["embedding_model"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model_cache["embedding_model"]


def get_reranker_model() -> Any:
    if "reranker" not in model_cache:
        logger.info("Loading reranker model for the first time...")
        from sentence_transformers import CrossEncoder

        model_cache["reranker"] = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    return model_cache["reranker"]


def get_groq_client() -> Groq:
    if "groq_client" not in model_cache:
        model_cache["groq_client"] = Groq()
    return model_cache["groq_client"]  # type: ignore[return-value]


def get_max_pdf_bytes() -> int:
    return int(os.getenv("MAX_PDF_BYTES", "15728640"))


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


def _build_source_references(chunks: List[ChunkRecord]) -> List[SourceReference]:
    return [
        SourceReference(
            page=chunk["page"],
            chunk_id=chunk["chunk_id"],
            excerpt=_normalize_excerpt(chunk["text"]),
        )
        for chunk in chunks
    ]


def _url_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


async def load_and_chunk_pdf(url: str) -> List[ChunkRecord]:
    """Download a PDF from a URL and split it into page-aware text chunks."""
    parsed_url = httpx.URL(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Document URL must use http or https.")

    logger.info("Downloading document from %s", url)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunk_records: List[ChunkRecord] = []
    chunk_id = 0

    try:
        with fitz.open(stream=response.content, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                if not page_text or not page_text.strip():
                    continue

                for chunk_text in text_splitter.split_text(page_text):
                    normalized = chunk_text.strip()
                    if not normalized:
                        continue
                    chunk_records.append(
                        {
                            "text": normalized,
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
    k_initial: int = 8,
    k_final: int = 3,
) -> Tuple[str, List[ChunkRecord]]:
    """Retrieve relevant context and grounded source chunks for a question."""
    question_embedding = np.array([embedding_model.embed_query(question)], dtype="float32")
    _, indices = faiss_index.search(question_embedding, k_initial)

    valid_indices = [index for index in indices[0] if index != -1]
    retrieved_chunks = [chunks[index] for index in valid_indices]
    if not retrieved_chunks:
        return "", []

    rerank_pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(rerank_pairs)
    reranked = sorted(zip(retrieved_chunks, rerank_scores), key=lambda item: item[1], reverse=True)
    selected_chunks = [chunk for chunk, _ in reranked[:k_final]]
    context = "\n\n".join(chunk["text"] for chunk in selected_chunks)
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
                        "based ONLY on the provided context. Be accurate and specific. "
                        "If the answer is not clearly stated in the context, respond with "
                        "'Information not found in the document.'"
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            model="llama-3.1-8b-instant",
            timeout=90,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise RuntimeError("LLM call failed.") from exc

    return response.choices[0].message.content


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
        return AnswerItem(
            question=question,
            answer=answer,
            status="ok",
            sources=_build_source_references(source_chunks),
        )
    except Exception:
        logger.exception("Failed to process question: %s", question)
        return AnswerItem(
            question=question,
            answer="Failed to process this question.",
            status="error",
            sources=[],
        )


document_cache: Dict[str, Tuple[List[ChunkRecord], faiss.IndexFlatL2]] = {}


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_pipeline(
    request: QueryRequest,
    response: Response,
    authorization: Optional[str] = Header(None),
) -> QueryResponse:
    if (
        not authorization
        or not authorization.startswith("Bearer ")
        or authorization.split("Bearer ")[1] != EXPECTED_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")

    if len(request.questions) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 questions allowed per request.")

    url = str(request.documents)
    cache_key = _url_cache_key(url)

    embedding_model = get_embedding_model()
    reranker = get_reranker_model()
    groq_client = get_groq_client()

    if cache_key in document_cache:
        logger.info("Document found in cache. Skipping download and embedding.")
        chunks, faiss_index = document_cache[cache_key]
        response.headers["X-Document-Cache"] = "HIT"
    else:
        chunks = await load_and_chunk_pdf(url)
        faiss_index = create_vector_store(chunks, embedding_model)
        document_cache[cache_key] = (chunks, faiss_index)
        response.headers["X-Document-Cache"] = "MISS"
        logger.info("Document processed and cached.")

    logger.info("Processing %s questions in parallel...", len(request.questions))
    answers = await asyncio.gather(
        *[
            process_question(q, faiss_index, chunks, embedding_model, reranker, groq_client)
            for q in request.questions
        ]
    )
    return QueryResponse(answers=list(answers))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "version": "2.1.0"}