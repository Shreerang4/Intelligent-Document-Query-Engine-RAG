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
from eval.prf_retriever import prf_query as _prf_query
from eval.metrics import (
    is_retrieval_hit,
    recall_at_k,
    mrr,
    abstention_accuracy,
    false_abstention_rate,
    key_fact_match_rate,
    latency_percentiles,
)

_PRIMARY_MODE = os.environ.get("RETRIEVAL_MODE", "faiss_reranker").strip().lower()
if _PRIMARY_MODE == "e5":
    _PRIMARY_MODE = "faiss_reranker"
_MODES = (_PRIMARY_MODE, "faiss_only", "bm25")


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
    run_prf: bool = False,
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
            q, _PRIMARY_MODE, r1["source_chunks"],
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

        # ── faiss_reranker_prf (answerable only) ──────────────────────────────
        if run_prf and q["answer_type"] == "answerable":
            prf_result = _prf_query(question, chunks, faiss_index)
            prf_total_ms = prf_result["latency"]["total_s"] * 1000
            prf_rec = _make_record(
                q, "faiss_reranker_prf", prf_result["ranked_chunks"],
                False, None, prf_total_ms, None,
            )
            prf_rec["prf_variants"] = prf_result["variants"]
            prf_rec["prf_fallback_fired"] = prf_result["fallback_fired"]
            prf_rec["prf_latency"] = prf_result["latency"]
            records.append(prf_rec)
            if llm_delay > 0:
                time.sleep(llm_delay)  # pace the PRF expansion LLM call

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
        if mode == _PRIMARY_MODE and llm_active:
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


# ── PRF metric aggregation ────────────────────────────────────────────────────
def _compute_prf_metrics(records: List[Dict]) -> Dict[str, Any]:
    prf_recs = [r for r in records if r["mode"] == "faiss_reranker_prf"]
    answerable = [r for r in prf_recs if r["answer_type"] == "answerable"]

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

    latencies = [r["retrieval_latency_ms"] for r in prf_recs]
    fallback_count = sum(1 for r in prf_recs if r.get("prf_fallback_fired", False))

    return {
        "n_questions": len(prf_recs),
        "retrieval": {
            "recall_at_3": recall_at_k(answerable, 3),
            "recall_at_5": recall_at_k(answerable, 5),
            "mrr": mrr(answerable),
            "by_question_type": by_type,
        },
        "latency": latency_percentiles(latencies),
        "fallback_count": fallback_count,
    }


def _prf_detail_for_ids(records: List[Dict], qids: List[str]) -> List[Dict]:
    """Per-question PRF result for a list of question IDs."""
    result: List[Dict] = []
    for qid in qids:
        prf_rec = next(
            (r for r in records if r["id"] == qid and r["mode"] == "faiss_reranker_prf"),
            None,
        )
        if prf_rec is None:
            continue
        hit3 = any(is_retrieval_hit(c, prf_rec) for c in prf_rec["ranked_chunks"][:3])
        hit5 = any(is_retrieval_hit(c, prf_rec) for c in prf_rec["ranked_chunks"][:5])
        result.append({
            "id": qid,
            "question": prf_rec["question_text"],
            "question_type": prf_rec["question_type"],
            "hit_at_3": hit3,
            "hit_at_5": hit5,
            "variants": prf_rec.get("prf_variants", []),
            "fallback_fired": prf_rec.get("prf_fallback_fired", False),
        })
    return result


# ── PRF Markdown report ────────────────────────────────────────────────────────
def _md_prf_report(
    prf_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    prf_detail: List[Dict],
    meta: Dict[str, Any],
) -> str:
    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}")

    def delta(prf_v: float, base_v: float) -> str:
        d = prf_v - base_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}"

    h(1, "PRF Evaluation Report (Pseudo-Relevance-Feedback)")
    lines.append("")
    lines.append(f"**Run date:** {meta['run_date']}")
    lines.append(f"**Questions:** {meta['n_questions']} total; PRF evaluated on "
                 f"{prf_metrics['n_questions']} answerable question(s)")
    lines.append(f"**LLM (Groq):** {'ON' if meta['llm_on'] else 'OFF — retrieval metrics only'}")
    lines.append(f"**PRF expansion model:** llama-3.1-8b-instant (3 variants per question)")
    lines.append(f"**PRF fallback fires:** {prf_metrics['fallback_count']} / "
                 f"{prf_metrics['n_questions']} questions used original query only")
    lines.append("")
    lines.append("> PRF latency is **sequential**: Round-1 FAISS+rerank + expansion LLM call "
                 "+ Round-2 multi-query FAISS+rerank + merged-pool CrossEncoder rerank.")

    # ── Side-by-side per question type ─────────────────────────────────────────
    h(2, "Side-by-Side Retrieval: faiss_reranker (baseline) vs faiss_reranker_prf")
    lines.append("")
    lines.append("| Question Type | n | Base R@3 | PRF R@3 | Delta | "
                 "Base R@5 | PRF R@5 | Delta | Base MRR | PRF MRR | Delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    base_by_type = baseline_metrics["retrieval"]["by_question_type"]
    prf_by_type = prf_metrics["retrieval"]["by_question_type"]
    for qt in sorted(set(base_by_type) | set(prf_by_type)):
        bv = base_by_type.get(qt, {"n": 0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0})
        pv = prf_by_type.get(qt, {"n": 0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0})
        n = bv["n"] or pv["n"]
        lines.append(
            f"| {qt} | {n} "
            f"| {_pct(bv['recall_at_3'])} | {_pct(pv['recall_at_3'])} | {delta(pv['recall_at_3'], bv['recall_at_3'])} "
            f"| {_pct(bv['recall_at_5'])} | {_pct(pv['recall_at_5'])} | {delta(pv['recall_at_5'], bv['recall_at_5'])} "
            f"| {bv['mrr']:.3f} | {pv['mrr']:.3f} | {delta(pv['mrr'], bv['mrr'])} |"
        )
    # Overall row
    br = baseline_metrics["retrieval"]
    pr = prf_metrics["retrieval"]
    n_total = prf_metrics["n_questions"]
    lines.append(
        f"| **OVERALL** | **{n_total}** "
        f"| **{_pct(br['recall_at_3'])}** | **{_pct(pr['recall_at_3'])}** | **{delta(pr['recall_at_3'], br['recall_at_3'])}** "
        f"| **{_pct(br['recall_at_5'])}** | **{_pct(pr['recall_at_5'])}** | **{delta(pr['recall_at_5'], br['recall_at_5'])}** "
        f"| **{br['mrr']:.3f}** | **{pr['mrr']:.3f}** | **{delta(pr['mrr'], br['mrr'])}** |"
    )

    # ── Latency ────────────────────────────────────────────────────────────────
    h(2, "Latency")
    lines.append("")
    prf_lat = prf_metrics["latency"]
    base_lat = baseline_metrics["latency"]
    lines.append("| Mode | p50 | p95 | n |")
    lines.append("|------|-----|-----|---|")
    lines.append(f"| faiss_reranker (baseline retrieval only) "
                 f"| {_ms(base_lat['p50'])} | {_ms(base_lat['p95'])} | {base_lat['n']} |")
    lines.append(f"| faiss_reranker_prf (total: R1 + expand + R2) "
                 f"| {_ms(prf_lat['p50'])} | {_ms(prf_lat['p95'])} | {prf_lat['n']} |")

    # ── Per-question PRF results for previously-failing questions ───────────────
    h(2, "PRF Results for Previously-Failing Questions (baseline needs_review)")
    lines.append("")
    if not prf_detail:
        lines.append("_No PRF records for needs_review questions (run --prf with answerable questions)._")
    else:
        lines.append("| id | type | PRF hit@3 | PRF hit@5 | Fallback |")
        lines.append("|----|----|-----------|-----------|----------|")
        for item in prf_detail:
            hit3_str = "YES" if item["hit_at_3"] else "no"
            hit5_str = "YES" if item["hit_at_5"] else "no"
            fb_str = "yes" if item["fallback_fired"] else "-"
            lines.append(
                f"| {item['id']} | {item['question_type']} "
                f"| {hit3_str} | {hit5_str} | {fb_str} |"
            )

        h(3, "Variants generated per question")
        for item in prf_detail:
            lines.append("")
            lines.append(f"**{item['id']}** (`{item['question_type']}`): "
                         f"{item['question']}")
            if item["fallback_fired"]:
                lines.append("  - *(fallback fired — expansion LLM call failed)*")
            elif not item["variants"]:
                lines.append("  - *(no variants returned)*")
            else:
                for v in item["variants"]:
                    lines.append(f"  - {v}")

    lines.append("")
    return "\n".join(lines)


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
    h(2, f"Retrieval by Question Type ({_PRIMARY_MODE}, answerable only)")
    lines.append("")
    lines.append("| Question Type | n | R@3 | R@5 | MRR |")
    lines.append("|---------------|---|-----|-----|-----|")
    for qt, vals in sorted(metrics[_PRIMARY_MODE]["retrieval"]["by_question_type"].items()):
        lines.append(
            f"| {qt} | {vals['n']} | {_pct(vals['recall_at_3'])}"
            f" | {_pct(vals['recall_at_5'])} | {vals['mrr']:.3f} |"
        )

    # ── LLM quality ────────────────────────────────────────────────────────────
    if "llm" in metrics[_PRIMARY_MODE]:
        llm = metrics[_PRIMARY_MODE]["llm"]
        h(2, f"LLM Answer Quality ({_PRIMARY_MODE} mode)")
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
    h(2, f"Cold vs Warm Retrieval Latency ({_PRIMARY_MODE})")
    lines.append("")
    lines.append(
        f"> **Cold** = first {_PRIMARY_MODE} query per document (may include model-load overhead).  "
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
    parser.add_argument("--prf", action="store_true",
                        help="Add pseudo-relevance-feedback mode (faiss_reranker_prf) and write prf_report.*")
    args = parser.parse_args()

    # Resolve all paths relative to repo root when not absolute
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    gold_path = _abs(args.gold)
    docs_dir = _abs(args.docs)
    out_dir = _abs(args.out)
    run_llm = _GROQ_KEY_PRESENT and not args.no_llm
    run_prf = args.prf and _GROQ_KEY_PRESENT

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load questions ─────────────────────────────────────────────────────────
    questions: List[Dict] = json.loads(gold_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    print(f"Gold : {gold_path}  ({len(questions)} questions)")
    print(f"Docs : {docs_dir}")
    print(f"Out  : {out_dir}")
    print(f"LLM  : {'ON (Groq)' if run_llm else 'OFF (retrieval only)'}")
    print(f"PRF  : {'ON (faiss_reranker_prf mode)' if run_prf else 'OFF'}")
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
                                llm_delay=args.llm_delay, run_prf=run_prf)
        all_records.extend(records)
        all_cold_ms.extend(cw["cold_ms"])
        all_warm_ms.extend(cw["warm_ms"])

    # ── aggregate ──────────────────────────────────────────────────────────────
    print("\nAggregating metrics...", flush=True)

    # needs_review uses only the three baseline modes so a PRF hit can't mask a baseline miss
    baseline_records = [r for r in all_records if r["mode"] in _MODES]
    metrics = _compute_metrics(baseline_records, run_llm)
    review = _needs_review(baseline_records)

    meta: Dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_questions": len(questions),
        "n_documents": len(doc_groups),
        "llm_on": run_llm,
        "gold_file": str(gold_path.relative_to(_REPO_ROOT)),
    }

    if run_prf:
        # ── PRF path: write prf_report.* only, do NOT overwrite baseline_report.* ──
        prf_metrics = _compute_prf_metrics(all_records)
        review_ids = [item["id"] for item in review]
        prf_detail = _prf_detail_for_ids(all_records, review_ids)

        # baseline_metrics for the side-by-side table uses the primary retrieval mode only
        baseline_reranker = {
            "retrieval": metrics[_PRIMARY_MODE]["retrieval"],
            "latency": metrics[_PRIMARY_MODE]["latency"],
        }

        prf_json = {
            "meta": meta,
            "prf_metrics": prf_metrics,
            "baseline_faiss_reranker": baseline_reranker,
            "needs_review_prf_detail": prf_detail,
            "records": all_records,
        }
        json_path = out_dir / "prf_report.json"
        json_path.write_text(
            json.dumps(prf_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON -> {json_path}")

        md_path = out_dir / "prf_report.md"
        md_path.write_text(
            _md_prf_report(prf_metrics, baseline_reranker, prf_detail, meta),
            encoding="utf-8",
        )
        print(f"MD   -> {md_path}")

        # ── console summary ────────────────────────────────────────────────────
        pr = prf_metrics["retrieval"]
        br = baseline_reranker["retrieval"]
        prf_lat = prf_metrics["latency"]
        print("\n-- PRF vs baseline (faiss_reranker) --------------------------")
        print(f"  {'mode':<24s}  R@3       R@5       MRR      p50")
        print(
            f"  {_PRIMARY_MODE:<24s}  "
            f"{_pct(br['recall_at_3']):<9s} {_pct(br['recall_at_5']):<9s} "
            f"{br['mrr']:.3f}    {_ms(metrics[_PRIMARY_MODE]['latency']['p50'])}"
        )
        print(
            f"  {'faiss_reranker_prf':<24s}  "
            f"{_pct(pr['recall_at_3']):<9s} {_pct(pr['recall_at_5']):<9s} "
            f"{pr['mrr']:.3f}    {_ms(prf_lat['p50'])}"
        )
        print(f"  PRF fallback fires: {prf_metrics['fallback_count']} / {prf_metrics['n_questions']}")
        if review:
            print(f"\n  Baseline needs_review: {len(review)} question(s)")
            hits3 = sum(1 for d in prf_detail if d["hit_at_3"])
            hits5 = sum(1 for d in prf_detail if d["hit_at_5"])
            print(f"  PRF rescued (hit@3): {hits3}/{len(prf_detail)}   (hit@5): {hits5}/{len(prf_detail)}")
        print("--------------------------------------------------------------")

    else:
        # ── Baseline path: write baseline_report.* ─────────────────────────────
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

        # ── console summary ────────────────────────────────────────────────────
        print("\n-- Retrieval summary -----------------------------------------")
        for mode in _MODES:
            r = metrics[mode]["retrieval"]
            lat = metrics[mode]["latency"]
            print(
                f"  {mode:20s}  R@3={_pct(r['recall_at_3'])}  "
                f"R@5={_pct(r['recall_at_5'])}  MRR={r['mrr']:.3f}  "
                f"p50={_ms(lat['p50'])}"
            )
        if "llm" in metrics[_PRIMARY_MODE]:
            llm = metrics[_PRIMARY_MODE]["llm"]
            print("-- LLM quality -----------------------------------------------")
            print(f"  key_fact_match_rate    {_pct(llm['key_fact_match_rate'])}")
            print(f"  abstention_accuracy    {_pct(llm['abstention_accuracy'])}")
            print(f"  false_abstention_rate  {_pct(llm['false_abstention_rate'])}")
        if review:
            print(f"\n  WARNING: {len(review)} question(s) zero hits all modes -- check needs_review")
        print("--------------------------------------------------------------")


if __name__ == "__main__":
    main()
