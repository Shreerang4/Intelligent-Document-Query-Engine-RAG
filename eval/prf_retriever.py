"""Pseudo-relevance-feedback (PRF) query expansion retriever — eval only.

Composes the existing pipeline_adapter and production CrossEncoder reranker.
No new retrieval or reranking math; main.py is NOT edited.

Algorithm (implemented exactly as specified):
  Round 1  Retrieve on the raw question via FAISS+CrossEncoder (k_initial=20,
           k_final=10). Take the top-5 returned chunks as document evidence.
  Expand   One Groq LLM call: given the question and the 5 evidence chunk texts,
           generate n_variants alternative phrasings that mirror the document's
           own terminology. Fallback to [] if the call errors or returns nothing
           usable — PRF never fails hard.
  Round 2  Retrieve for the original question + each variant separately (same
           production path, use_reranker=True, k_initial=20, k_final=10 each).
           Merge the candidate pools by chunk_id, re-rank the merged pool with
           the CrossEncoder against the ORIGINAL question, keep top k_final.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import os as _os
_os.environ.setdefault("API_TOKEN", "eval-placeholder")

# main is imported by pipeline_adapter before this module is loaded in the runner;
# Python's module cache makes this a zero-cost lookup in the common case.
import main as _pipeline

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = """\
You are a query-reformulation assistant for a document search system.

ORIGINAL QUESTION:
{question}

REAL EXCERPTS FROM THE DOCUMENT BEING SEARCHED (verbatim text, not summaries):
{excerpts}

Task: rewrite the original question into exactly {n_variants} alternative phrasings.
Rules:
  - Use ONLY terminology and phrasing that appears in the excerpts above, or is a
    natural near-synonym of language found there. Mirror the document's own wording.
  - Do NOT invent formal jargon if the excerpts use plain language, and do NOT use
    plain language if the excerpts use formal jargon.
  - Do NOT add information not implied by the original question.
  - Output ONLY the {n_variants} alternatives, one per line, no numbering, no
    commentary, no blank lines before or after.\
"""


def _call_expansion_llm(
    question: str,
    evidence_chunks: List[Dict],
    n_variants: int = 3,
    model: str = "llama-3.1-8b-instant",
) -> Tuple[List[str], bool]:
    """Return (variants, fallback_fired).

    fallback_fired=True when the call failed or returned nothing usable;
    variants will be [] in that case and Round 2 uses the original query only.
    """
    excerpts = "\n\n---\n\n".join(c["text"][:600] for c in evidence_chunks[:5])
    prompt = _EXPANSION_PROMPT.format(
        question=question,
        excerpts=excerpts,
        n_variants=n_variants,
    )
    try:
        client = _pipeline.get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        raw = (response.choices[0].message.content or "").strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        if not variants:
            logger.warning(
                "PRF expansion returned empty response for %r — fallback fired.",
                question[:60],
            )
            return [], True
        return variants[:n_variants], False
    except Exception as exc:
        logger.warning(
            "PRF expansion LLM call failed (%s) — fallback to original query.", exc
        )
        return [], True


def prf_query(
    question: str,
    chunks: List[Dict],
    faiss_index: Any,
    k_initial: int = 20,
    k_final: int = 10,
    n_top_for_expansion: int = 5,
    n_variants: int = 3,
) -> Dict:
    """Run PRF query expansion retrieval and return the final ranked list.

    Returns:
        ranked_chunks   top k_final chunks in rank order (text/page/chunk_id)
        variants        expansion variants generated ([] if fallback fired)
        fallback_fired  True when the expansion LLM call failed or returned nothing
        latency         {round1_s, expansion_s, round2_s, total_s}
    """
    from eval.pipeline_adapter import query as _adapter_query

    t_start = time.perf_counter()

    # ── Round 1: retrieve on raw question ─────────────────────────────────────
    t0 = time.perf_counter()
    r1 = _adapter_query(
        question, chunks, faiss_index,
        use_reranker=True, run_llm=False,
        k_initial=k_initial, k_final=k_final,
    )
    round1_s = time.perf_counter() - t0
    evidence_chunks = r1["source_chunks"][:n_top_for_expansion]

    # ── Expansion: document-grounded query variants ────────────────────────────
    t0 = time.perf_counter()
    variants, fallback_fired = _call_expansion_llm(question, evidence_chunks, n_variants)
    expansion_s = time.perf_counter() - t0

    # ── Round 2: retrieve for original + variants, merge, rerank ──────────────
    t0 = time.perf_counter()
    queries = [question] + variants  # original always present

    # Merge retrieved chunks across all queries; dedupe by chunk_id.
    chunk_pool: Dict[int, Dict] = {}
    for q in queries:
        r = _adapter_query(
            q, chunks, faiss_index,
            use_reranker=True, run_llm=False,
            k_initial=k_initial, k_final=k_final,
        )
        for c in r["source_chunks"]:
            chunk_pool.setdefault(c["chunk_id"], c)

    merged = list(chunk_pool.values())

    # Rerank the merged pool with CrossEncoder against the ORIGINAL question.
    reranker = _pipeline.get_reranker_model()
    rerank_pairs = [[question, c["text"]] for c in merged]
    scores = reranker.predict(rerank_pairs)
    ranked = sorted(zip(merged, scores), key=lambda pair: pair[1], reverse=True)
    ranked_chunks = [c for c, _ in ranked[:k_final]]

    round2_s = time.perf_counter() - t0
    total_s = time.perf_counter() - t_start

    return {
        "ranked_chunks": ranked_chunks,
        "variants": variants,
        "fallback_fired": fallback_fired,
        "latency": {
            "round1_s": round1_s,
            "expansion_s": expansion_s,
            "round2_s": round2_s,
            "total_s": total_s,
        },
    }
