"""k_initial sweep probe: recall vs candidate-pool-size curve.

Runs retrieval (CrossEncoder reranker ON, k_final=10) over all answerable
questions for k_initial in [20, 35, 50] and reports:
  - Overall recall@3/@5/MRR and latency at each k_initial.
  - Per-question-type breakdown (conceptual / paraphrase / distractor / lexical).
  - For the 7 baseline needs_review IDs: how many become hit@3 / hit@5.

No Groq key needed (run_llm=False throughout).

Usage (from repo root):
    .venv\\Scripts\\python.exe eval/probe_k_sweep.py
    [--gold PATH] [--docs DIR] [--report PATH] [--k-values 20,35,50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document, query as _adapter_query
from eval.metrics import is_retrieval_hit, recall_at_k, mrr, latency_percentiles
from eval.runner import _resolve_pdf

_K_FINAL = 10


def _make_record(q: Dict, ranked_chunks: List[Dict], latency_ms: float) -> Dict:
    return {
        "id": q["id"],
        "answer_type": q["answer_type"],
        "question_type": q["question_type"],
        "supporting_text_hint": q.get("supporting_text_hint", ""),
        "supporting_pages": q.get("supporting_pages", []),
        "ranked_chunks": ranked_chunks,
        "retrieval_latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="eval/benchmark/questions.json")
    parser.add_argument("--docs", default="eval/benchmark/docs")
    parser.add_argument("--report", default="eval/results/baseline_report.json")
    parser.add_argument("--k-values", default="20,35,50",
                        help="Comma-separated k_initial values (default: 20,35,50)")
    args = parser.parse_args()

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    gold_path   = _abs(args.gold)
    docs_dir    = _abs(args.docs)
    report_path = _abs(args.report)
    k_values    = [int(v.strip()) for v in args.k_values.split(",")]

    questions: List[Dict] = json.loads(gold_path.read_text(encoding="utf-8"))
    answerable: List[Dict] = [q for q in questions if q["answer_type"] == "answerable"]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_ids: Set[str] = {item["id"] for item in report.get("needs_review", [])}

    print(f"k_initial sweep  k_values={k_values}  k_final={_K_FINAL}")
    print(f"  {len(answerable)} answerable questions  |  "
          f"{len(review_ids)} in baseline needs_review\n")

    # Group by document and ingest each PDF once.
    by_doc: Dict[str, List[Dict]] = {}
    for q in answerable:
        by_doc.setdefault(q["document_id"], []).append(q)

    doc_ingested: Dict[str, Any] = {}
    for doc_id, doc_qs in by_doc.items():
        pdf_path = _resolve_pdf(doc_qs[0]["document_file"], docs_dir)
        print(f"Ingesting {pdf_path.name}...", flush=True)
        chunks, faiss_index = ingest_document(str(pdf_path))
        doc_ingested[doc_id] = (chunks, faiss_index)
        print(f"  {len(chunks)} chunks\n", flush=True)

    qtypes = sorted({q["question_type"] for q in answerable})
    results: Dict[int, Dict] = {}

    for k in k_values:
        print(f"Running k_initial={k}...", flush=True)
        records: List[Dict] = []

        for doc_id, doc_qs in by_doc.items():
            chunks, faiss_index = doc_ingested[doc_id]
            for q in doc_qs:
                t0 = time.perf_counter()
                r = _adapter_query(
                    q["question"], chunks, faiss_index,
                    use_reranker=True, run_llm=False,
                    k_initial=k, k_final=_K_FINAL,
                )
                lat_ms = (time.perf_counter() - t0) * 1000
                records.append(_make_record(q, r["source_chunks"], lat_ms))

        by_type: Dict[str, Any] = {}
        for qt in qtypes:
            qt_recs = [r for r in records if r["question_type"] == qt]
            by_type[qt] = {
                "n": len(qt_recs),
                "recall_at_3": recall_at_k(qt_recs, 3),
                "recall_at_5": recall_at_k(qt_recs, 5),
                "mrr": mrr(qt_recs),
            }

        lat_stats = latency_percentiles([r["retrieval_latency_ms"] for r in records])

        rescued_3 = sum(
            1 for r in records
            if r["id"] in review_ids
            and any(is_retrieval_hit(c, r) for c in r["ranked_chunks"][:3])
        )
        rescued_5 = sum(
            1 for r in records
            if r["id"] in review_ids
            and any(is_retrieval_hit(c, r) for c in r["ranked_chunks"][:5])
        )

        results[k] = {
            "overall": {
                "recall_at_3": recall_at_k(records, 3),
                "recall_at_5": recall_at_k(records, 5),
                "mrr": mrr(records),
            },
            "by_type": by_type,
            "latency": lat_stats,
            "rescued_3": rescued_3,
            "rescued_5": rescued_5,
        }

        ov = results[k]["overall"]
        print(f"  R@3={ov['recall_at_3']:.1%}  R@5={ov['recall_at_5']:.1%}  "
              f"MRR={ov['mrr']:.3f}  "
              f"p50={lat_stats['p50']:.0f}ms  p95={lat_stats['p95']:.0f}ms  "
              f"rescued@3={rescued_3}/{len(review_ids)}  rescued@5={rescued_5}/{len(review_ids)}\n")

    # ── Final report tables ─────────────────────────────────────────────────────
    SEP = "=" * 82
    print(SEP)
    print("k_initial SWEEP  (reranker ON, k_final=10, retrieval-only)")
    print(SEP)

    # Overall table
    print(f"\n{'Overall':} ({len(answerable)} answerable questions):")
    hdr = f"  {'k_initial':<10} {'R@3':<7} {'R@5':<7} {'MRR':<7} {'p50':<8} {'p95':<8} {'rescued@3':<11} rescued@5"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for k in k_values:
        rv = results[k]
        ov = rv["overall"]
        lt = rv["latency"]
        print(f"  {k:<10} {ov['recall_at_3']:.1%}  {ov['recall_at_5']:.1%}  "
              f"{ov['mrr']:.3f}  {lt['p50']:.0f}ms    {lt['p95']:.0f}ms    "
              f"{rv['rescued_3']}/{len(review_ids)}          {rv['rescued_5']}/{len(review_ids)}")

    # Per question-type tables (conceptual + paraphrase are the focus)
    for qt in qtypes:
        print(f"\n  [{qt}]")
        print(f"  {'k_initial':<10} {'n':<4} {'R@3':<8} {'R@5':<8} MRR")
        print(f"  {'-' * 38}")
        for k in k_values:
            bt = results[k]["by_type"].get(qt, {})
            n  = bt.get("n", 0)
            print(f"  {k:<10} {n:<4} {bt.get('recall_at_3', 0):.1%}    "
                  f"{bt.get('recall_at_5', 0):.1%}    {bt.get('mrr', 0):.3f}")

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
