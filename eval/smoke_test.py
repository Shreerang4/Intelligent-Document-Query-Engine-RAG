"""Smoke test: exercise the three retrieval modes and print ordered results.

Usage (from repo root):
    python eval/smoke_test.py <path/to/document.pdf> "<question>"

Does NOT call the Groq LLM (run_llm=False), so no API key is required.
Proves only that the plumbing works — no scoring or metrics.
"""
import sys
import time
from pathlib import Path
from typing import List

# Repo root on sys.path so `eval` is importable as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document, query  # noqa: E402
from eval.bm25_retriever import bm25_search               # noqa: E402


def _preview(text: str, max_chars: int = 80) -> str:
    return text.replace("\n", " ")[:max_chars]


def _print_results(label: str, chunks: List, latency_s: float) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}  |  latency: {latency_s:.3f}s")
    print(f"{'─' * 60}")
    if not chunks:
        print("  (no chunks returned)")
        return
    for rank, c in enumerate(chunks, start=1):
        print(f"  [{rank:>2}] page={c['page']}  chunk_id={c['chunk_id']}")
        print(f"        {_preview(c['text'])!r}")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    print(f"PDF      : {pdf_path}")
    print(f"Question : {question}")
    print("\nIngesting document (embedding model loads on first run)...")

    t0 = time.perf_counter()
    chunks, faiss_index = ingest_document(pdf_path)
    ingest_s = time.perf_counter() - t0
    print(f"Ingested {len(chunks)} chunks in {ingest_s:.2f}s.")

    # ── Mode 1: FAISS + reranker ────────────────────────────────────────────
    r1 = query(question, chunks, faiss_index, use_reranker=True, run_llm=False)
    _print_results("Mode 1 — FAISS + Reranker", r1["source_chunks"], r1["latency"]["retrieval_s"])

    # ── Mode 2: FAISS only (no reranker) ───────────────────────────────────
    r2 = query(question, chunks, faiss_index, use_reranker=False, run_llm=False)
    _print_results("Mode 2 — FAISS only (no reranker)", r2["source_chunks"], r2["latency"]["retrieval_s"])

    # ── Mode 3: BM25 ───────────────────────────────────────────────────────
    t_bm25 = time.perf_counter()
    bm25_chunks = bm25_search(question, chunks, k=10)
    bm25_s = time.perf_counter() - t_bm25
    _print_results("Mode 3 — BM25 (lexical)", bm25_chunks, bm25_s)

    print(f"\n{'═' * 60}")
    print("Smoke test complete. No scoring — plumbing check only.")


if __name__ == "__main__":
    main()
