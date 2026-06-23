"""Diagnose needs_review questions: hint/extraction ARTIFACT vs genuine retrieval MISS.

Read-only. No production changes, no metric-logic changes, no edits to questions.json.

For each answerable question that scored zero retrieval hits across all modes
(the needs_review list in baseline_report.json), this:
  1. Ingests its document once (via pipeline_adapter).
  2. Scans EVERY chunk in the document with is_retrieval_hit().
  3. Reports whether the hint exists anywhere in the extracted corpus, where, and
     whether that chunk ever appeared in the faiss_reranker results.
  4. Classifies:
       hint_in_corpus == False -> ARTIFACT  (hint text not in extracted chunks)
       hint_in_corpus == True  -> IN-CORPUS (chunk exists; retrieval missed it)

Usage (from repo root):
    .venv\\Scripts\\python.exe eval/diagnose_misses.py
    [--gold PATH] [--docs DIR] [--report PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document, query as _adapter_query
from eval.metrics import is_retrieval_hit, normalize_text
from eval.runner import _resolve_pdf  # reuse the double-extension-tolerant resolver

_K_INITIAL = 20
_K_FINAL = 10


def _preview(text: str, n: int = 120) -> str:
    return text.replace("\n", " ")[:n]


def _matching_chunks(record: Dict, chunks: List[Dict]) -> List[Dict]:
    """Every chunk in the corpus that is_retrieval_hit() considers relevant.

    Either path counts (hint substring OR page fallback) — same definition
    is_retrieval_hit() uses, so hint_in_corpus == bool(this list).
    """
    return [c for c in chunks if is_retrieval_hit(c, record)]


def _text_hit_chunks(hint: str, chunks: List[Dict]) -> List[Dict]:
    """Chunks where the hint text itself was extracted (primary path only).

    Bypasses the page fallback so we can tell a genuine retrieval gap (hint text
    is in the corpus but didn't surface) from a likely extraction/hint artifact
    (only the page number matched; the hint string is nowhere in the chunks).
    """
    if not hint:
        return []
    norm_hint = normalize_text(hint)
    return [c for c in chunks if norm_hint in normalize_text(c["text"])]


def _retrieved_rank(target_ids: set, ranked_chunks: List[Dict]) -> Optional[int]:
    """1-indexed rank of the first ranked chunk whose chunk_id is in target_ids."""
    for rank, c in enumerate(ranked_chunks, start=1):
        if c["chunk_id"] in target_ids:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="eval/benchmark/questions.json")
    parser.add_argument("--docs", default="eval/benchmark/docs")
    parser.add_argument("--report", default="eval/results/baseline_report.json")
    args = parser.parse_args()

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    gold_path = _abs(args.gold)
    docs_dir = _abs(args.docs)
    report_path = _abs(args.report)

    questions: List[Dict] = json.loads(gold_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_ids = [item["id"] for item in report.get("needs_review", [])]

    if not review_ids:
        print("needs_review is empty — nothing to diagnose.")
        return

    print(f"Diagnosing {len(review_ids)} needs_review question(s) from {report_path.name}\n")

    # Group review questions by document so we ingest each PDF only once.
    by_doc: Dict[str, List[str]] = {}
    for qid in review_ids:
        by_doc.setdefault(by_id[qid]["document_id"], []).append(qid)

    rows: List[Dict[str, Any]] = []

    for doc_id, qids in by_doc.items():
        doc_file = by_id[qids[0]]["document_file"]
        pdf_path = _resolve_pdf(doc_file, docs_dir)
        print(f"Ingesting {pdf_path.name} for {len(qids)} question(s)...", flush=True)
        chunks, faiss_index = ingest_document(str(pdf_path))
        print(f"  {len(chunks)} chunks.\n", flush=True)

        for qid in qids:
            q = by_id[qid]
            hint = q.get("supporting_text_hint", "")
            record = {
                "supporting_text_hint": hint,
                "supporting_pages": q.get("supporting_pages", []),
            }

            matches = _matching_chunks(record, chunks)          # hint OR page
            hint_in_corpus = bool(matches)
            text_hits = _text_hit_chunks(hint, chunks)          # hint text only

            # Three-way classification, sharper than hint_in_corpus alone:
            #   ARTIFACT   - nothing matched at all (no page, no hint text)
            #   PAGE-ONLY  - page present but the hint string was never extracted
            #                -> likely a hint/extraction artifact; fix the hint
            #   IN-CORPUS  - the hint text itself is in the corpus
            #                -> genuine retrieval miss
            if not hint_in_corpus:
                classification = "ARTIFACT"
            elif not text_hits:
                classification = "PAGE-ONLY"
            else:
                classification = "IN-CORPUS"

            # Rank check: prefer the precise text-hit chunks; else the page set.
            ref_chunks = text_hits if text_hits else matches
            ref_ids = {c["chunk_id"] for c in ref_chunks}
            rank_str = "n/a"
            if ref_ids:
                r = _adapter_query(
                    q["question"], chunks, faiss_index,
                    use_reranker=True, run_llm=False,
                    k_initial=_K_INITIAL, k_final=_K_FINAL,
                )
                rank = _retrieved_rank(ref_ids, r["source_chunks"])
                rank_str = str(rank) if rank is not None else "not in top-k"

            print(f"[{qid}]  {classification}")
            print(f"  hint               : {hint!r}")
            print(f"  hint_in_corpus     : {hint_in_corpus}  "
                  f"(page-or-text; {len(matches)} chunk(s))")
            print(f"  hint TEXT extracted: {bool(text_hits)}  ({len(text_hits)} chunk(s))")
            for c in ref_chunks[:3]:
                print(f"    - chunk_id={c['chunk_id']} page={c['page']}: {_preview(c['text'])!r}")
            print(f"  faiss_reranker rank of a matching chunk: {rank_str}")
            if classification == "PAGE-ONLY":
                print("  -> hint string never extracted; only the page matched. Fix the hint.")
            elif classification == "IN-CORPUS":
                print("  -> hint text is in the corpus but retrieval missed it. Retrieval gap.")
            print()

            rows.append({
                "id": qid,
                "type": q["question_type"],
                "page": (",".join(str(c["page"]) for c in ref_chunks[:3])
                         or ",".join(str(p) for p in q.get("supporting_pages", []))),
                "classification": classification,
                "evidence": (
                    f"text_hit={bool(text_hits)}; "
                    f"chunk(s) {sorted(ref_ids)[:3]}; rerank rank={rank_str}"
                ),
            })

    # ── compact summary table ──────────────────────────────────────────────────
    print("=" * 96)
    print(f"{'id':<32} {'type':<11} {'page':<8} {'class':<10} evidence")
    print("-" * 96)
    for row in rows:
        print(f"{row['id']:<32} {row['type']:<11} {row['page']:<8} "
              f"{row['classification']:<10} {row['evidence']}")
    print("=" * 96)

    n_artifact = sum(1 for r in rows if r["classification"] == "ARTIFACT")
    n_pageonly = sum(1 for r in rows if r["classification"] == "PAGE-ONLY")
    n_corpus = sum(1 for r in rows if r["classification"] == "IN-CORPUS")
    print(f"\nARTIFACT (no match): {n_artifact}   "
          f"PAGE-ONLY (fix hint): {n_pageonly}   "
          f"IN-CORPUS (retrieval gap): {n_corpus}")


if __name__ == "__main__":
    main()
