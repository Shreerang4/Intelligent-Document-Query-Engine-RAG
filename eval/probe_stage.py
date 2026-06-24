"""Stage-attribution probe: FAISS-MISS vs RERANKER-DROP for the 7 IN-CORPUS gaps.

For each needs_review question, determines whether the correct chunk is absent
from the top-k_initial FAISS pool (FAISS-MISS) or present in FAISS but dropped
by the CrossEncoder below top-10 (RERANKER-DROP).

Correct chunk identification uses the hint text path only (no page fallback),
matching diagnose_misses.py exactly.

Usage (from repo root):
    .venv\\Scripts\\python.exe eval/probe_stage.py
    [--gold PATH] [--docs DIR] [--report PATH] [--k-initial N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document, query as _adapter_query
from eval.metrics import normalize_text
from eval.runner import _resolve_pdf

_K_FINAL = 10


def _text_hit_ids(hint: str, chunks: List[Dict]) -> Set[int]:
    """chunk_ids where the hint appears verbatim (text path only, no page fallback)."""
    if not hint:
        return set()
    norm = normalize_text(hint)
    return {c["chunk_id"] for c in chunks if norm in normalize_text(c["text"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="eval/benchmark/questions.json")
    parser.add_argument("--docs", default="eval/benchmark/docs")
    parser.add_argument("--report", default="eval/results/baseline_report.json")
    parser.add_argument("--k-initial", type=int, default=20,
                        help="FAISS candidate pool size to probe (default: 20)")
    args = parser.parse_args()

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    gold_path = _abs(args.gold)
    docs_dir  = _abs(args.docs)
    report_path = _abs(args.report)
    k_initial = args.k_initial

    questions: List[Dict] = json.loads(gold_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_ids = [item["id"] for item in report.get("needs_review", [])]

    if not review_ids:
        print("needs_review is empty -- nothing to probe.")
        return

    print(f"Stage-attribution probe: {len(review_ids)} needs_review question(s)  "
          f"k_initial={k_initial}  k_final={_K_FINAL}\n")

    # Group by document so each PDF is ingested once.
    by_doc: Dict[str, List[str]] = {}
    for qid in review_ids:
        by_doc.setdefault(by_id[qid]["document_id"], []).append(qid)

    rows: List[Dict[str, Any]] = []

    for doc_id, qids in by_doc.items():
        doc_file = by_id[qids[0]]["document_file"]
        pdf_path = _resolve_pdf(doc_file, docs_dir)
        print(f"Ingesting {pdf_path.name} ({len(qids)} question(s))...", flush=True)
        chunks, faiss_index = ingest_document(str(pdf_path))
        print(f"  {len(chunks)} chunks\n", flush=True)

        for qid in qids:
            q = by_id[qid]
            question = q["question"]
            hint = q.get("supporting_text_hint", "")

            correct_ids = _text_hit_ids(hint, chunks)
            if not correct_ids:
                print(f"  [{qid}] SKIP -- hint text not found in corpus (hint: {hint!r})")
                rows.append({
                    "id": qid, "type": q["question_type"],
                    "correct_ids": [], "faiss_rank": None,
                    "rerank_rank": None, "classification": "ARTIFACT",
                })
                continue

            # ── 1. Raw FAISS top-k_initial ─────────────────────────────────────
            # k_final=k_initial ensures retrieve_context doesn't truncate before
            # we can see all FAISS candidates (it slices to k_final when no reranker).
            faiss_result = _adapter_query(
                question, chunks, faiss_index,
                use_reranker=False, run_llm=False,
                k_initial=k_initial, k_final=k_initial,
            )
            faiss_ids = [c["chunk_id"] for c in faiss_result["source_chunks"]]

            faiss_rank: Optional[int] = None
            for rank, cid in enumerate(faiss_ids, start=1):
                if cid in correct_ids:
                    faiss_rank = rank
                    break

            # ── 2. Post-reranker top-k_final ───────────────────────────────────
            rerank_result = _adapter_query(
                question, chunks, faiss_index,
                use_reranker=True, run_llm=False,
                k_initial=k_initial, k_final=_K_FINAL,
            )
            rerank_ids = [c["chunk_id"] for c in rerank_result["source_chunks"]]

            rerank_rank: Optional[int] = None
            for rank, cid in enumerate(rerank_ids, start=1):
                if cid in correct_ids:
                    rerank_rank = rank
                    break

            # ── Classification ─────────────────────────────────────────────────
            if faiss_rank is None:
                classification = "FAISS-MISS"
            elif rerank_rank is not None:
                classification = "RECOVERED"
            else:
                classification = "RERANKER-DROP"

            faiss_str  = str(faiss_rank)  if faiss_rank  is not None else f"none/top-{k_initial}"
            rerank_str = str(rerank_rank) if rerank_rank is not None else f"below-{_K_FINAL}"

            print(f"  [{qid}]  {classification}")
            print(f"    hint             : {hint!r}")
            print(f"    correct chunk_ids: {sorted(correct_ids)}")
            print(f"    FAISS rank       : {faiss_str}  ({len(faiss_ids)} candidates)")
            print(f"    rerank rank      : {rerank_str}  (top-{_K_FINAL})")
            print()

            rows.append({
                "id": qid, "type": q["question_type"],
                "correct_ids": sorted(correct_ids),
                "faiss_rank": faiss_rank, "rerank_rank": rerank_rank,
                "classification": classification,
            })

    # ── Compact summary table ───────────────────────────────────────────────────
    W_ID, W_TY, W_FR, W_RR = 34, 12, 16, 14
    sep = "=" * (W_ID + W_TY + W_FR + W_RR + 14)
    print(sep)
    print(f"{'id':<{W_ID}} {'type':<{W_TY}} {'faiss_rank':<{W_FR}} "
          f"{'rerank_rank':<{W_RR}} classification")
    print("-" * len(sep))
    for row in rows:
        fs = str(row["faiss_rank"]) if row["faiss_rank"] is not None else f"none/top-{k_initial}"
        rs = str(row["rerank_rank"]) if row["rerank_rank"] is not None else f"below-{_K_FINAL}"
        print(f"{row['id']:<{W_ID}} {row['type']:<{W_TY}} {fs:<{W_FR}} "
              f"{rs:<{W_RR}} {row['classification']}")
    print(sep)

    n_fm  = sum(1 for r in rows if r["classification"] == "FAISS-MISS")
    n_rd  = sum(1 for r in rows if r["classification"] == "RERANKER-DROP")
    n_rec = sum(1 for r in rows if r["classification"] == "RECOVERED")
    n_art = sum(1 for r in rows if r["classification"] == "ARTIFACT")

    print(f"\nTALLY  (k_initial={k_initial}, k_final={_K_FINAL}):")
    print(f"  FAISS-MISS    (not in top-{k_initial} FAISS pool)         : {n_fm}")
    print(f"  RERANKER-DROP (in FAISS pool, dropped by CrossEncoder)    : {n_rd}")
    print(f"  RECOVERED     (actually in top-{_K_FINAL} post-rerank)  : {n_rec}")
    if n_art:
        print(f"  ARTIFACT      (hint text absent from corpus, skipped)    : {n_art}")

    print()
    if n_fm > n_rd:
        print("=> Hypothesis A dominant: FAISS pool too narrow. "
              "Increasing k_initial may help.")
    elif n_rd > n_fm:
        print("=> Hypothesis B dominant: CrossEncoder drops correct chunks. "
              "Reranker scoring is the bottleneck.")
    elif n_fm == n_rd and n_fm > 0:
        print("=> Split evenly between FAISS-MISS and RERANKER-DROP.")
    else:
        print("=> No FAISS-MISS or RERANKER-DROP (unexpected for needs_review set).")


if __name__ == "__main__":
    main()
