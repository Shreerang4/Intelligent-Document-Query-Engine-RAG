"""Evaluation runner — produces baseline_report.json and baseline_report.md.

Usage (from repo root):
    python eval/runner.py [options]

Options:
    --gold PATH    Questions JSON  (default: eval/benchmark/questions.json)
    --docs DIR     PDF docs dir    (default: eval/benchmark/docs)
    --out  DIR     Output dir      (default: eval/results)
    --no-llm       Skip LLM — retrieval metrics only
    --limit N      Run only the first N questions (quick sanity check)

Limitations:
    - Cache bypass: ingestion calls load_and_chunk_pdf_bytes + create_vector_store
      directly, bypassing document_cache. Cold/warm latencies therefore measure
      "first vs subsequent FAISS query after a fresh ingest," not HTTP cold-start.
    - LLM abstention metrics (abstention_accuracy, false_abstention_rate) are
      only reported when --no-llm is NOT set and GROQ_API_KEY is present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── path bootstrap ─────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import os
_GROQ_KEY_PRESENT = bool(os.environ.get("GROQ_API_KEY"))

from eval.pipeline_adapter import ingest_document, query as _adapter_query
from eval.bm25_retriever import bm25_search
from eval.metrics import (
    is_retrieval_hit,
    recall_at_k,
    mrr,
    abstention_accuracy,
    false_abstention_rate,
    key_fact_match_rate,
    latency_percentiles,
)

_MODES = ("faiss_reranker", "faiss_only", "bm25")


# ── PDF resolution ─────────────────────────────────────────────────────────────
def _resolve_pdf(doc_file: str, docs_dir: Path) -> Path:
    """Return the actual PDF path, tolerating double-extension upload artifacts.

    questions.json stores "infosys_ar_2024_25.pdf" but uploaded files may be
    named "infosys_ar_2024_25.pdf.pdf". We try the exact name, then the doubled
    extension, then scan by stem so either format works.
    """
    expected_name = Path(doc_file).name         # "infosys_ar_2024_25.pdf"
    expected_stem = Path(expected_name).stem    # "infosys_ar_2024_25"

    exact = docs_dir / expected_name
    if exact.exists():
        return exact

    doubled = docs_dir / (expected_name + ".pdf")
    if doubled.exists():
        return doubled

    for p in sorted(docs_dir.glob("*.pdf")):
        if p.stem == expected_stem or p.name == expected_name:
            return p

    return exact  # intentionally missing — yields a clear FileNotFoundError message


# ── EvalRecord builder ─────────────────────────────────────────────────────────
def _make_record(
    q: Dict,
    mode: str,
    ranked_chunks: List[Dict],
    abstained: bool,
    answer: Optional[str],
    retrieval_latency_ms: float,
    llm_latency_ms: Optional[float],
) -> Dict:
    return {
        # consumed by metrics.py
        "id": q["id"],
        "answer_type": q["answer_type"],
        "question_type": q["question_type"],
        "mode": mode,
        "supporting_text_hint": q.get("supporting_text_hint", ""),
        "supporting_pages": q.get("supporting_pages", []),
        "key_facts": q.get("key_facts", []),
        "ranked_chunks": ranked_chunks,
        "abstained": abstained,
        "answer": answer,
        # runner-only fields
        "document_id": q["document_id"],
        "question_text": q["question"],
        "retrieval_latency_ms": retrieval_latency_ms,
        "llm_latency_ms": llm_latency_ms,
    }


# ── per-document evaluation ────────────────────────────────────────────────────
def _run_doc(
    doc_id: str,
    questions: List[Dict],
    pdf_path: Path,
    run_llm: bool,
    llm_delay: float = 0.0,
) -> Tuple[List[Dict], Dict[str, List[float]]]:
    """Ingest one PDF, run all three modes for every question.

    Returns (records, cold_warm) where cold_warm["cold_ms"] holds the retrieval
    latency of the first question per document and cold_warm["warm_ms"] holds the
    rest — a proxy for model-warm vs fully-warm FAISS query time.
    """
    print(f"\n  Ingesting {pdf_path.name} ({len(questions)} questions)...", flush=True)
    t0 = time.perf_counter()
    chunks, faiss_index = ingest_document(str(pdf_path))
    print(f"  {len(chunks)} chunks, ingest {time.perf_counter() - t0:.1f}s", flush=True)

    records: List[Dict] = []
    cold_ms: List[float] = []
    warm_ms: List[float] = []

    for q_idx, q in enumerate(questions):
        question = q["question"]
        is_first = q_idx == 0
        print(f"  [{q_idx + 1:>2}/{len(questions)}] {q['id']}", flush=True)

        # ── faiss_reranker ────────────────────────────────────────────────────
        r1 = _adapter_query(question, chunks, faiss_index,
                            use_reranker=True, run_llm=run_llm)
        r1_retr_ms = r1["latency"]["retrieval_s"] * 1000
        r1_llm_ms = r1["latency"]["llm_s"] * 1000 if run_llm else None
        records.append(_make_record(
            q, "faiss_reranker", r1["source_chunks"],
            r1["abstained"], r1["answer"],
            r1_retr_ms, r1_llm_ms,
        ))
        (cold_ms if is_first else warm_ms).append(r1_retr_ms)
        if run_llm and llm_delay > 0:
            time.sleep(llm_delay)  # stay within Groq free-tier TPM limit

        # ── faiss_only ────────────────────────────────────────────────────────
        r2 = _adapter_query(question, chunks, faiss_index,
                            use_reranker=False, run_llm=False)
        r2_retr_ms = r2["latency"]["retrieval_s"] * 1000
        records.append(_make_record(
            q, "faiss_only", r2["source_chunks"],
            r2["abstained"], None,
            r2_retr_ms, None,
        ))

        # ── bm25 ──────────────────────────────────────────────────────────────
        t_bm25 = time.perf_counter()
        bm25_chunks = bm25_search(question, chunks, k=10)
        bm25_retr_ms = (time.perf_counter() - t_bm25) * 1000
        records.append(_make_record(
            q, "bm25", bm25_chunks,
            len(bm25_chunks) == 0, None,
            bm25_retr_ms, None,
        ))

    return records, {"cold_ms": cold_ms, "warm_ms": warm_ms}


# ── metric aggregation ─────────────────────────────────────────────────────────
def _compute_metrics(records: List[Dict], llm_active: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for mode in _MODES:
        mode_recs = [r for r in records if r["mode"] == mode]
        answerable = [r for r in mode_recs if r["answer_type"] == "answerable"]

        # retrieval metrics — only over answerable (unanswerable have no gold)
        qtypes = sorted({r["question_type"] for r in answerable})
        by_type: Dict[str, Any] = {}
        for qt in qtypes:
            qt_recs = [r for r in answerable if r["question_type"] == qt]
            by_type[qt] = {
                "n": len(qt_recs),
                "recall_at_3": recall_at_k(qt_recs, 3),
                "recall_at_5": recall_at_k(qt_recs, 5),
                "mrr": mrr(qt_recs),
            }

        retr_latencies = [r["retrieval_latency_ms"] for r in mode_recs]
        mode_entry: Dict[str, Any] = {
            "n_questions": len(mode_recs),
            "retrieval": {
                "recall_at_3": recall_at_k(answerable, 3),
                "recall_at_5": recall_at_k(answerable, 5),
                "mrr": mrr(answerable),
                "by_question_type": by_type,
            },
            "latency": latency_percentiles(retr_latencies),
        }

        # LLM quality — faiss_reranker only, and only when LLM actually ran
        if mode == "faiss_reranker" and llm_active:
            llm_lats = [r["llm_latency_ms"] for r in mode_recs
                        if r["llm_latency_ms"] is not None]
            mode_entry["llm"] = {
                "key_fact_match_rate": key_fact_match_rate(mode_recs),
                "abstention_accuracy": abstention_accuracy(mode_recs),
                "false_abstention_rate": false_abstention_rate(mode_recs),
                "latency": latency_percentiles(llm_lats) if llm_lats else None,
            }

        out[mode] = mode_entry

    return out


# ── needs-review list ──────────────────────────────────────────────────────────
def _needs_review(records: List[Dict]) -> List[Dict]:
    """Answerable questions with zero retrieval hits across ALL three modes."""
    answerable_ids = sorted({r["id"] for r in records
                             if r["answer_type"] == "answerable"})
    result: List[Dict] = []
    for qid in answerable_ids:
        q_recs = [r for r in records if r["id"] == qid]
        all_miss = all(
            not any(is_retrieval_hit(c, r) for c in r["ranked_chunks"])
            for r in q_recs
        )
        if all_miss:
            ref = q_recs[0]
            result.append({
                "id": qid,
                "document_id": ref["document_id"],
                "question": ref["question_text"],
                "question_type": ref["question_type"],
                "supporting_pages": ref["supporting_pages"],
                "supporting_text_hint": ref["supporting_text_hint"],
            })
    return result


# ── Markdown report ────────────────────────────────────────────────────────────
def _pct(v: float) -> str:
    return f"{v:.1%}"

def _ms(v: float) -> str:
    return f"{v:.0f} ms"


def _md_report(
    metrics: Dict[str, Any],
    needs_review: List[Dict],
    meta: Dict[str, Any],
    cold_warm: Dict[str, List[float]],
) -> str:
    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}")

    h(1, "Baseline Evaluation Report")
    lines.append("")
    lines.append(f"**Run date:** {meta['run_date']}")
    lines.append(f"**Questions:** {meta['n_questions']} across {meta['n_documents']} document(s)")
    lines.append(f"**LLM (Groq):** {'ON' if meta['llm_on'] else 'OFF — retrieval metrics only'}")

    # ── Retrieval metrics ──────────────────────────────────────────────────────
    h(2, "Retrieval Metrics (answerable questions only)")
    lines.append("")
    lines.append("| Mode | R@3 | R@5 | MRR | Retr p50 | Retr p95 |")
    lines.append("|------|-----|-----|-----|----------|----------|")
    for mode in _MODES:
        m = metrics[mode]
        r = m["retrieval"]
        lat = m["latency"]
        lines.append(
            f"| {mode} | {_pct(r['recall_at_3'])} | {_pct(r['recall_at_5'])}"
            f" | {r['mrr']:.3f} | {_ms(lat['p50'])} | {_ms(lat['p95'])} |"
        )

    # ── By question type ───────────────────────────────────────────────────────
    h(2, "Retrieval by Question Type (faiss_reranker, answerable only)")
    lines.append("")
    lines.append("| Question Type | n | R@3 | R@5 | MRR |")
    lines.append("|---------------|---|-----|-----|-----|")
    for qt, vals in sorted(metrics["faiss_reranker"]["retrieval"]["by_question_type"].items()):
        lines.append(
            f"| {qt} | {vals['n']} | {_pct(vals['recall_at_3'])}"
            f" | {_pct(vals['recall_at_5'])} | {vals['mrr']:.3f} |"
        )

    # ── LLM quality ────────────────────────────────────────────────────────────
    if "llm" in metrics["faiss_reranker"]:
        llm = metrics["faiss_reranker"]["llm"]
        h(2, "LLM Answer Quality (faiss_reranker mode)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Key-fact match rate | {_pct(llm['key_fact_match_rate'])} |")
        lines.append(f"| Abstention accuracy (unanswerable Qs) | {_pct(llm['abstention_accuracy'])} |")
        lines.append(f"| False abstention rate (answerable Qs) | {_pct(llm['false_abstention_rate'])} |")
        if llm.get("latency"):
            lat = llm["latency"]
            lines.append(f"| LLM latency p50 | {_ms(lat['p50'])} |")
            lines.append(f"| LLM latency p95 | {_ms(lat['p95'])} |")

    # ── Cold vs warm ───────────────────────────────────────────────────────────
    h(2, "Cold vs Warm Retrieval Latency (faiss_reranker)")
    lines.append("")
    lines.append(
        "> **Cold** = first faiss_reranker query per document (may include model-load overhead).  "
    )
    lines.append("> **Warm** = all subsequent queries on the same document's FAISS index.")
    lines.append("")
    if cold_warm["cold_ms"] and cold_warm["warm_ms"]:
        cold_lat = latency_percentiles(cold_warm["cold_ms"])
        warm_lat = latency_percentiles(cold_warm["warm_ms"])
        lines.append("| Bucket | n | p50 | p95 |")
        lines.append("|--------|---|-----|-----|")
        lines.append(f"| cold (1st per doc) | {cold_lat['n']} | {_ms(cold_lat['p50'])} | {_ms(cold_lat['p95'])} |")
        lines.append(f"| warm | {warm_lat['n']} | {_ms(warm_lat['p50'])} | {_ms(warm_lat['p95'])} |")
    else:
        lines.append("_Insufficient data — need more than one question per document._")

    # ── Needs review ───────────────────────────────────────────────────────────
    h(2, "Questions Needing Review")
    lines.append("")
    if not needs_review:
        lines.append(
            "_All answerable questions had at least one retrieval hit "
            "in at least one mode._"
        )
    else:
        lines.append(
            f"**{len(needs_review)} answerable question(s)** returned zero "
            "retrieval hits across all three modes:"
        )
        lines.append("")
        for item in needs_review:
            lines.append(
                f"- **{item['id']}** (`{item['question_type']}`): {item['question']}"
            )
            if item.get("supporting_text_hint"):
                lines.append(f"  - hint: `{item['supporting_text_hint']}`")
            if item.get("supporting_pages"):
                lines.append(f"  - pages: {item['supporting_pages']}")

    lines.append("")
    return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gold", default="eval/benchmark/questions.json",
                        help="Path to questions JSON (default: eval/benchmark/questions.json)")
    parser.add_argument("--docs", default="eval/benchmark/docs",
                        help="Directory containing benchmark PDFs (default: eval/benchmark/docs)")
    parser.add_argument("--out", default="eval/results",
                        help="Output directory (default: eval/results)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation — retrieval metrics only")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Run only the first N questions (quick sanity check)")
    parser.add_argument("--llm-delay", type=float, default=7.0, metavar="SEC",
                        help="Seconds to sleep between LLM calls (avoids Groq TPM limit; default: 7)")
    args = parser.parse_args()

    # Resolve all paths relative to repo root when not absolute
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    gold_path = _abs(args.gold)
    docs_dir = _abs(args.docs)
    out_dir = _abs(args.out)
    run_llm = _GROQ_KEY_PRESENT and not args.no_llm

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load questions ─────────────────────────────────────────────────────────
    questions: List[Dict] = json.loads(gold_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    print(f"Gold : {gold_path}  ({len(questions)} questions)")
    print(f"Docs : {docs_dir}")
    print(f"Out  : {out_dir}")
    print(f"LLM  : {'ON (Groq)' if run_llm else 'OFF (retrieval only)'}")
    if not _GROQ_KEY_PRESENT and not args.no_llm:
        print("  [GROQ_API_KEY not set — falling back to retrieval-only mode]")

    # ── validate PDFs upfront ──────────────────────────────────────────────────
    doc_pdf_map: Dict[str, Path] = {}
    errors: List[str] = []
    for q in questions:
        doc_id = q["document_id"]
        if doc_id in doc_pdf_map:
            continue
        pdf = _resolve_pdf(q["document_file"], docs_dir)
        if not pdf.exists():
            errors.append(f"PDF not found for {doc_id!r}: tried {pdf}")
        else:
            doc_pdf_map[doc_id] = pdf
            print(f"  PDF: {doc_id} -> {pdf.name}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── group questions by document ────────────────────────────────────────────
    doc_groups: Dict[str, List[Dict]] = {}
    for q in questions:
        doc_groups.setdefault(q["document_id"], []).append(q)

    # ── run evaluation ─────────────────────────────────────────────────────────
    all_records: List[Dict] = []
    all_cold_ms: List[float] = []
    all_warm_ms: List[float] = []

    print(f"\nRunning {len(doc_groups)} document(s)...")
    for doc_id, doc_qs in doc_groups.items():
        records, cw = _run_doc(doc_id, doc_qs, doc_pdf_map[doc_id], run_llm,
                                llm_delay=args.llm_delay)
        all_records.extend(records)
        all_cold_ms.extend(cw["cold_ms"])
        all_warm_ms.extend(cw["warm_ms"])

    # ── aggregate ──────────────────────────────────────────────────────────────
    print("\nAggregating metrics...", flush=True)
    metrics = _compute_metrics(all_records, run_llm)
    review = _needs_review(all_records)

    meta: Dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_questions": len(questions),
        "n_documents": len(doc_groups),
        "llm_on": run_llm,
        "gold_file": str(gold_path.relative_to(_REPO_ROOT)),
    }

    # ── write outputs ──────────────────────────────────────────────────────────
    json_report = {
        "meta": meta,
        "metrics": metrics,
        "needs_review": review,
        "records": all_records,
    }
    json_path = out_dir / "baseline_report.json"
    json_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"JSON -> {json_path}")

    md_path = out_dir / "baseline_report.md"
    md_path.write_text(
        _md_report(metrics, review, meta, {"cold_ms": all_cold_ms, "warm_ms": all_warm_ms}),
        encoding="utf-8",
    )
    print(f"MD   -> {md_path}")

    # ── console summary ────────────────────────────────────────────────────────
    print("\n-- Retrieval summary -----------------------------------------")
    for mode in _MODES:
        r = metrics[mode]["retrieval"]
        lat = metrics[mode]["latency"]
        print(
            f"  {mode:20s}  R@3={_pct(r['recall_at_3'])}  "
            f"R@5={_pct(r['recall_at_5'])}  MRR={r['mrr']:.3f}  "
            f"p50={_ms(lat['p50'])}"
        )
    if "llm" in metrics["faiss_reranker"]:
        llm = metrics["faiss_reranker"]["llm"]
        print("-- LLM quality -----------------------------------------------")
        print(f"  key_fact_match_rate    {_pct(llm['key_fact_match_rate'])}")
        print(f"  abstention_accuracy    {_pct(llm['abstention_accuracy'])}")
        print(f"  false_abstention_rate  {_pct(llm['false_abstention_rate'])}")
    if review:
        print(f"\n  WARNING: {len(review)} question(s) zero hits all modes -- check needs_review")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    main()
