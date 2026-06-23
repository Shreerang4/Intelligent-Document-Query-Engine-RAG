"""Thin seam between the eval harness and the production pipeline (main.py).

Call ingest_document() once per PDF, then query() per question.
No auth, no HTTP, no claim-verification LLM calls.
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root must precede eval/ on sys.path so `import main` resolves correctly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# main.py checks API_TOKEN at import time for the HTTP endpoints.
# The eval never calls those endpoints, so a placeholder satisfies the check.
import os as _os
if not _os.environ.get("API_TOKEN"):
    _os.environ["API_TOKEN"] = "eval-placeholder"

import main as pipeline  # the production backend


def ingest_document(pdf_path: str) -> Tuple[List[Dict], Any]:
    """Read a PDF from disk, chunk it, and build a FAISS index.

    Returns (chunks, faiss_index).
    Raises ValueError on bad input — wraps any FastAPI HTTPException so callers
    don't need to know about the HTTP layer.
    """
    from fastapi import HTTPException  # lazy: keeps module-level surface minimal

    pdf_bytes = Path(pdf_path).read_bytes()
    try:
        chunks = pipeline.load_and_chunk_pdf_bytes(pdf_bytes)
        embedding_model = pipeline.get_embedding_model()
        faiss_index = pipeline.create_vector_store(chunks, embedding_model)
    except HTTPException as exc:
        raise ValueError(f"PDF ingestion failed: {exc.detail}") from exc
    return chunks, faiss_index


def query(
    question: str,
    chunks: List[Dict],
    faiss_index: Any,
    use_reranker: bool = True,
    run_llm: bool = True,
    k_initial: int = 20,
    k_final: int = 10,
) -> Dict:
    """Run retrieval (and optionally LLM generation) for one question.

    k_initial=20, k_final=10 gives enough candidates to compute Recall@3/5
    and rank-based MRR in the eval harness.

    Returns a dict with:
        answer        - str if run_llm=True, else None
        abstained     - True when context was empty or answer is non-informative
        source_chunks - list of {text, page, chunk_id} in rank order
        latency       - {retrieval_s, llm_s, total_s} in seconds

    run_llm=False works without a Groq API key (retrieval-only path).
    """
    embedding_model = pipeline.get_embedding_model()
    # Skip loading the reranker model entirely when it won't be used.
    reranker = pipeline.get_reranker_model() if use_reranker else None

    t_start = time.perf_counter()
    context, raw_chunks = pipeline.retrieve_context(
        question,
        faiss_index,
        chunks,
        embedding_model,
        reranker,
        k_initial=k_initial,
        k_final=k_final,
        use_reranker=use_reranker,
    )
    retrieval_s = time.perf_counter() - t_start

    answer: Optional[str] = None
    llm_s = 0.0

    if run_llm:
        if not context:
            answer = "Information not found in the document."
        else:
            groq_client = pipeline.get_groq_client()
            t_llm = time.perf_counter()
            answer = pipeline.generate_answer(question, context, groq_client)
            llm_s = time.perf_counter() - t_llm

    abstained = (not context) or (
        answer is not None and pipeline._is_non_informative_answer(answer)
    )

    return {
        "answer": answer,
        "abstained": abstained,
        "source_chunks": [
            {"text": c["text"], "page": c["page"], "chunk_id": c["chunk_id"]}
            for c in raw_chunks
        ],
        "latency": {
            "retrieval_s": retrieval_s,
            "llm_s": llm_s,
            "total_s": retrieval_s + llm_s,
        },
    }
