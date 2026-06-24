"""Rank probe: where does the correct chunk land under different embedding models?

Compares all-MiniLM-L6-v2 (production baseline), BAAI/bge-small-en-v1.5, and
intfloat/e5-small-v2 on the 7 baseline needs_review IN-CORPUS gaps.

Prefix conventions applied (getting these wrong silently cripples the model):
  MiniLM   : no prefix on either side  (symmetric model)
  BGE      : query gets "Represent this sentence for searching relevant passages: ";
             passages have no prefix  (per BAAI bge-small-en-v1.5 model card)
  E5       : query gets "query: "; passages get "passage: "
             (per intfloat e5-small-v2 model card)

No Groq, no reranker, no production changes.

Usage (from repo root):
    .venv\\Scripts\\python.exe eval/probe_embedders.py
    [--gold PATH] [--docs DIR] [--report PATH]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document
from eval.metrics import normalize_text
from eval.runner import _resolve_pdf

# ── Embedder registry ─────────────────────────────────────────────────────────
EMBEDDERS = [
    {
        "key": "MiniLM",
        "model_name": "all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "prefix_note": "no prefix on either side (symmetric model; matches production)",
    },
    {
        "key": "BGE",
        "model_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "prefix_note": (
            'query: "Represent this sentence for searching relevant passages: {q}"; '
            "passage: no prefix  (BAAI bge-small-en-v1.5 model card)"
        ),
    },
    {
        "key": "E5",
        "model_name": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "prefix_note": (
            'query: "query: {q}"; passage: "passage: {p}"  '
            "(intfloat e5-small-v2 model card)"
        ),
    },
]

TOP_THRESHOLDS = (20, 50)


# ── helpers ───────────────────────────────────────────────────────────────────
def _text_hit_ids(hint: str, chunks: List[Dict]) -> Set[int]:
    """chunk_ids whose extracted text contains hint verbatim (text path only)."""
    if not hint:
        return set()
    norm = normalize_text(hint)
    return {c["chunk_id"] for c in chunks if norm in normalize_text(c["text"])}


def _encode(model, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """Encode a list of texts; return (n, dim) float32 L2-normalised matrix."""
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype("float32")


def _best_rank(
    q_emb: np.ndarray,       # (dim,) unit vector
    chunk_embs: np.ndarray,  # (n, dim) unit matrix
    correct_ids: Set[int],
    chunks: List[Dict],
) -> Tuple[Optional[int], int]:
    """Return (rank_of_best_correct_chunk, total_chunks).  Rank is 1-indexed."""
    sims = chunk_embs @ q_emb          # cosine sim, shape (n,)
    order = np.argsort(-sims)          # descending
    n = len(chunks)
    for rank, idx in enumerate(order, start=1):
        if chunks[int(idx)]["chunk_id"] in correct_ids:
            return rank, n
    return None, n


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold",   default="eval/benchmark/questions.json")
    parser.add_argument("--docs",   default="eval/benchmark/docs")
    parser.add_argument("--report", default="eval/results/baseline_report.json")
    args = parser.parse_args()

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    questions: List[Dict] = json.loads(
        _abs(args.gold).read_text(encoding="utf-8")
    )
    by_id = {q["id"]: q for q in questions}

    report = json.loads(_abs(args.report).read_text(encoding="utf-8"))
    review_ids: List[str] = [item["id"] for item in report.get("needs_review", [])]

    if not review_ids:
        print("needs_review is empty -- nothing to probe.")
        return

    # Group review questions by document
    by_doc: Dict[str, List[str]] = {}
    for qid in review_ids:
        by_doc.setdefault(by_id[qid]["document_id"], []).append(qid)

    # ── Step 1: Ingest all PDFs, collect chunks + correct chunk info ──────────
    print("=" * 72)
    print(f"Rank probe -- {len(review_ids)} needs_review questions, "
          f"{len(EMBEDDERS)} embedders")
    print("=" * 72)
    print()
    print("Ingesting documents...", flush=True)

    doc_chunks: Dict[str, List[Dict]] = {}
    # correct_chunk_info: qid -> first correct chunk for the fairness block
    correct_chunk_info: Dict[str, Dict] = {}
    # correct_ids_map: qid -> set of chunk_ids that are correct
    correct_ids_map: Dict[str, Set[int]] = {}

    for doc_id, qids in by_doc.items():
        doc_file = by_id[qids[0]]["document_file"]
        pdf_path = _resolve_pdf(doc_file, _abs(args.docs))
        print(f"  {pdf_path.name}...", flush=True)
        chunks, _ = ingest_document(str(pdf_path))
        doc_chunks[doc_id] = chunks
        print(f"    {len(chunks)} chunks", flush=True)

        for qid in qids:
            hint = by_id[qid].get("supporting_text_hint", "")
            cids = _text_hit_ids(hint, chunks)
            correct_ids_map[qid] = cids
            # capture the first correct chunk for the fairness block
            for c in chunks:
                if c["chunk_id"] in cids:
                    correct_chunk_info[qid] = {
                        "chunk_id": c["chunk_id"],
                        "page":     c["page"],
                        "text":     c["text"],
                    }
                    break
            print(f"    [{qid}]  hint={hint[:60]!r}  "
                  f"correct_ids={sorted(cids)}", flush=True)

    # ── Step 2: For each embedder, encode docs + probe questions ──────────────
    # results[qid][key] = {"rank": int|None, "total": int}
    results: Dict[str, Dict[str, Dict]] = {qid: {} for qid in review_ids}

    from sentence_transformers import SentenceTransformer

    for cfg in EMBEDDERS:
        key   = cfg["key"]
        mname = cfg["model_name"]
        qpfx  = cfg["query_prefix"]
        ppfx  = cfg["passage_prefix"]

        print()
        print(f"--- {key}  ({mname}) ---")
        print(f"    Prefix: {cfg['prefix_note']}", flush=True)

        model = SentenceTransformer(mname)

        # Encode every document's chunks once
        doc_embs: Dict[str, np.ndarray] = {}
        for doc_id, chunks in doc_chunks.items():
            print(f"    Encoding {len(chunks)} chunks [{doc_id}]...", flush=True)
            texts = [ppfx + c["text"] for c in chunks]
            doc_embs[doc_id] = _encode(model, texts)

        # Rank each question
        for doc_id, qids in by_doc.items():
            chunks = doc_chunks[doc_id]
            chunk_embs = doc_embs[doc_id]
            for qid in qids:
                cids = correct_ids_map[qid]
                if not cids:
                    results[qid][key] = {"rank": None, "total": len(chunks)}
                    print(f"    [{qid}]  SKIP (no correct chunk in corpus)", flush=True)
                    continue

                question = by_id[qid]["question"]
                q_emb = _encode(model, [qpfx + question])[0]

                rank, total = _best_rank(q_emb, chunk_embs, cids, chunks)
                results[qid][key] = {"rank": rank, "total": total}

                flags = "  ".join(
                    f"top-{t}={'Y' if rank is not None and rank <= t else 'N'}"
                    for t in TOP_THRESHOLDS
                )
                print(f"    [{qid}]  rank={rank}/{total}  {flags}", flush=True)

        del model
        gc.collect()

    # ── Output section ────────────────────────────────────────────────────────
    SEP72  = "=" * 72
    SEP88  = "=" * 88
    keys = [cfg["key"] for cfg in EMBEDDERS]

    # -- Per-question rank table -----------------------------------------------
    print()
    print(SEP88)
    print("PER-QUESTION RANK TABLE  (rank = cosine nearest-neighbor position, 1-best)")
    print(SEP88)
    print()

    # Build header
    hdr  = f"{'id':<34} {'type':<12}"
    for k in keys:
        hdr += f"  {k+'-rank':<10} {'t20':<4} {'t50':<4}"
    print(hdr)
    print("-" * len(hdr))

    for qid in review_ids:
        q   = by_id[qid]
        row = f"{qid:<34} {q['question_type']:<12}"
        for k in keys:
            r     = results[qid].get(k, {})
            rank  = r.get("rank")
            total = r.get("total", "?")
            rank_str = f"{rank}/{total}" if rank is not None else f">{total}"
            t20 = "Y" if rank is not None and rank <= 20 else "N"
            t50 = "Y" if rank is not None and rank <= 50 else "N"
            row += f"  {rank_str:<10} {t20:<4} {t50:<4}"
        print(row)

    # -- Fairness blocks -------------------------------------------------------
    print()
    print(SEP88)
    print("FAIRNESS BLOCKS  (question / correct-chunk pairs for human review)")
    print(SEP88)

    _NUMBER_WORDS = (
        "how much", "how many", "what percentage", "what amount",
        "what figure", "which figure", "what total", "how large",
        "how big", "what size", "what number",
    )

    for qid in review_ids:
        q    = by_id[qid]
        info = correct_chunk_info.get(qid)
        print(f"\n[{qid}]  ({q['question_type']})")
        print(f"  Q : {q['question']}")
        print(f"  hint: {q.get('supporting_text_hint', '')!r}")
        if info:
            preview = info["text"][:300].replace("\n", " ")
            print(f"  Chunk  chunk_id={info['chunk_id']}  page={info['page']}:")
            print(f"    {preview!r}")
            q_lower = q["question"].lower()
            asks_number = any(w in q_lower for w in _NUMBER_WORDS)
            has_digit   = any(ch.isdigit() for ch in info["text"])
            if asks_number and not has_digit:
                flag = ("Kind mismatch: question requests a quantitative value; "
                        "correct chunk is narrative prose with no visible digits.")
            elif asks_number and has_digit:
                flag = ("Kind match: question requests a number; "
                        "correct chunk contains numeric content.")
            else:
                flag = ("Both qualitative/conceptual -- "
                        "no obvious kind mismatch; pure semantic gap.")
        else:
            flag = "No correct chunk captured."
        print(f"  Flag: {flag}")

    # -- Summary table ---------------------------------------------------------
    print()
    print(SEP88)
    print("SUMMARY TABLE")
    print(SEP88)
    print()
    print(f"  {'Embedder':<38}  {'top-20 / 7':<12}  {'top-50 / 7'}")
    print(f"  {'-'*62}")
    for cfg in EMBEDDERS:
        k = cfg["key"]
        t20 = sum(
            1 for qid in review_ids
            if (r := results[qid].get(k, {})) and r.get("rank") and r["rank"] <= 20
        )
        t50 = sum(
            1 for qid in review_ids
            if (r := results[qid].get(k, {})) and r.get("rank") and r["rank"] <= 50
        )
        label = f"{k} ({cfg['model_name']})"
        print(f"  {label:<38}  {str(t20)+'/7':<12}  {t50}/7")

    print()
    print("  Verdicts per question:")
    print(f"  {'-'*80}")
    for qid in review_ids:
        q = by_id[qid]

        def _in(k: str, thresh: int) -> bool:
            r = results[qid].get(k, {}).get("rank")
            return r is not None and r <= thresh

        miniLM_t20 = _in("MiniLM", 20)
        other_t20  = any(_in(k, 20) for k in ("BGE", "E5"))
        any_t50    = any(_in(k, 50) for k in keys)
        any_t20    = any(_in(k, 20) for k in keys)

        if not miniLM_t20 and other_t20:
            verdict = "RESCUED by stronger embedder (MiniLM misses top-20; BGE/E5 hits top-20)"
        elif not any_t50:
            verdict = "STILL MISSED by all embedders -- candidate unfair/ambiguous question"
        elif not any_t20:
            verdict = "MARGINAL (best rank is top-50 but no embedder reaches top-20)"
        else:
            verdict = "TOP-20 by at least one model (including MiniLM)"

        ranks = "  ".join(
            f"{k}={results[qid].get(k, {}).get('rank', 'N/A')}"
            for k in keys
        )
        print(f"\n  {qid}  ({q['question_type']})")
        print(f"    ranks: {ranks}")
        print(f"    => {verdict}")

    print()
    print(SEP72)


if __name__ == "__main__":
    main()
