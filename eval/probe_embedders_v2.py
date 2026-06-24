"""Rank probe v2: quality vs cost comparison of three candidate embedders.

Candidates:
  E5    intfloat/e5-small-v2          (~33M  — prior probe baseline)
  GTE   thenlper/gte-base-en-v1.5    (~110M — base-size middle ground)
  QWEN3 Qwen/Qwen3-Embedding-0.6B    (~600M — highest quality, heaviest)

Prefix conventions applied per model card (stated in output):
  E5    : query gets "query: "; passages get "passage: "
  GTE   : no prefix on either side — sentence-transformers handles mean pooling;
          thenlper/gte-base-en-v1.5 model card confirms no retrieval instruction
          is needed for English retrieval tasks.
  QWEN3 : query gets full instruction prefix (QWEN3_INSTRUCTION below);
          passages have no prefix; sentence-transformers >=2.0 handles last-token
          pooling automatically when model_kwargs include is_causal=True.
          If transformers < 4.51.0, Qwen3 is skipped and the error is reported
          verbatim — no silent fallback.

Cost is measured on this CPU (no GPU):
  - Model load time and process working-set delta (Windows ctypes, no psutil).
  - Per-document full-corpus encode time and throughput (chunks/sec).
  - Per-query encode time (median of 5 samples).

Usage (from repo root):
    .venv\\Scripts\\python.exe eval/probe_embedders_v2.py
    [--gold PATH] [--docs DIR] [--report PATH]
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as _cwt
import gc
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.pipeline_adapter import ingest_document
from eval.metrics import normalize_text
from eval.runner import _resolve_pdf

# ── Windows RSS measurement (no psutil) ──────────────────────────────────────
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb",                      _cwt.DWORD),
        ("PageFaultCount",          _cwt.DWORD),
        ("PeakWorkingSetSize",      ctypes.c_size_t),
        ("WorkingSetSize",          ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage",     ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage",  ctypes.c_size_t),
        ("PeakNonPagedPoolUsage",   ctypes.c_size_t),
        ("PagefileUsage",           ctypes.c_size_t),
        ("PeakPagefileUsage",       ctypes.c_size_t),
    ]


def _rss_mb() -> Optional[float]:
    """Current process working-set size in MB (Windows kernel32/psapi)."""
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc), pmc.cb,
        )
        return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None


# ── Qwen3 instruction ─────────────────────────────────────────────────────────
QWEN3_INSTRUCTION = (
    "Instruct: Given a financial question, retrieve the passage that answers it\n"
    "Query: "
)

# ── Embedder registry ─────────────────────────────────────────────────────────
EMBEDDERS: List[Dict[str, Any]] = [
    {
        "key":          "E5",
        "model_name":   "intfloat/e5-small-v2",
        "query_fn":     lambda q: "query: " + q,
        "passage_pfx":  "passage: ",
        "batch_size":   64,
        "prefix_note":  'query: "query: {q}"; passage: "passage: {p}" (e5-small-v2 model card)',
    },
    {
        "key":          "GTE",
        "model_name":   "Alibaba-NLP/gte-base-en-v1.5",
        "load_kwargs":  {"trust_remote_code": True},
        "query_fn":     lambda q: q,
        "passage_pfx":  "",
        "batch_size":   32,
        "prefix_note":  (
            "No prefix on either side. Alibaba-NLP/gte-base-en-v1.5 (relocated from "
            "thenlper/gte-base-en-v1.5; requires trust_remote_code=True). "
            "Model card confirms no retrieval instruction needed for English retrieval."
        ),
    },
    {
        "key":          "QWEN3",
        "model_name":   "Qwen/Qwen3-Embedding-0.6B",
        "query_fn":     lambda q: QWEN3_INSTRUCTION + q,
        "passage_pfx":  "",
        "batch_size":   4,
        "prefix_note":  (
            f'query: "{QWEN3_INSTRUCTION}{{q}}" (full instruction prefix per Qwen3 '
            "model card); passage: no prefix. sentence-transformers handles last-token "
            "pooling when the model is loaded through SentenceTransformer()."
        ),
    },
]

TOP_K = (20, 50)


# ── helpers ───────────────────────────────────────────────────────────────────
def _text_hit_ids(hint: str, chunks: List[Dict]) -> Set[int]:
    if not hint:
        return set()
    norm = normalize_text(hint)
    return {c["chunk_id"] for c in chunks if norm in normalize_text(c["text"])}


def _encode_batch(model: Any, texts: List[str], batch_size: int) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype("float32")


def _best_rank(
    q_emb: np.ndarray,
    chunk_embs: np.ndarray,
    correct_ids: Set[int],
    chunks: List[Dict],
) -> Tuple[Optional[int], int]:
    sims  = chunk_embs @ q_emb
    order = np.argsort(-sims)
    for rank, idx in enumerate(order, start=1):
        if chunks[int(idx)]["chunk_id"] in correct_ids:
            return rank, len(chunks)
    return None, len(chunks)


def _check_qwen3() -> Tuple[bool, str]:
    """Return (ok, message). Checks transformers >= 4.51.0."""
    try:
        ver = importlib.metadata.version("transformers")
        parts = [int(x) for x in ver.split(".")[:3]]
        if parts < [4, 51, 0]:
            return False, f"transformers=={ver} (need >=4.51.0)"
        st_ver = importlib.metadata.version("sentence-transformers")
        return True, f"transformers=={ver}  sentence-transformers=={st_ver} -- OK"
    except Exception as exc:
        return False, str(exc)


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

    questions: List[Dict] = json.loads(_abs(args.gold).read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}

    report  = json.loads(_abs(args.report).read_text(encoding="utf-8"))
    review_ids: List[str] = [item["id"] for item in report.get("needs_review", [])]

    if not review_ids:
        print("needs_review is empty -- nothing to probe.")
        return

    by_doc: Dict[str, List[str]] = {}
    for qid in review_ids:
        by_doc.setdefault(by_id[qid]["document_id"], []).append(qid)

    # ── Ingest all PDFs once ──────────────────────────────────────────────────
    print("=" * 72, flush=True)
    print(f"Rank probe v2 -- {len(review_ids)} questions, {len(EMBEDDERS)} embedders")
    print("=" * 72, flush=True)
    print()
    print("Ingesting documents...", flush=True)

    doc_chunks: Dict[str, List[Dict]] = {}
    correct_ids_map: Dict[str, Set[int]] = {}

    for doc_id, qids in by_doc.items():
        pdf = _resolve_pdf(by_id[qids[0]]["document_file"], _abs(args.docs))
        print(f"  {pdf.name}...", flush=True)
        chunks, _ = ingest_document(str(pdf))
        doc_chunks[doc_id] = chunks
        print(f"    {len(chunks)} chunks", flush=True)
        for qid in qids:
            hint = by_id[qid].get("supporting_text_hint", "")
            cids = _text_hit_ids(hint, chunks)
            correct_ids_map[qid] = cids
            print(f"    [{qid}]  hint={hint[:55]!r}  "
                  f"correct_ids={sorted(cids)}", flush=True)

    from sentence_transformers import SentenceTransformer
    print(f"\nLibrary versions: "
          f"transformers=={importlib.metadata.version('transformers')}  "
          f"sentence-transformers=={importlib.metadata.version('sentence-transformers')}\n",
          flush=True)

    # results[qid][key] = {"rank": int|None, "total": int, "skipped": bool}
    results: Dict[str, Dict[str, Dict]] = {qid: {} for qid in review_ids}
    cost: Dict[str, Dict]    = {}
    keys = [cfg["key"] for cfg in EMBEDDERS]
    doc_ids = list(doc_chunks.keys())

    # ── Loop over embedders ───────────────────────────────────────────────────
    for cfg in EMBEDDERS:
        key      = cfg["key"]
        mname    = cfg["model_name"]
        query_fn = cfg["query_fn"]
        ppfx     = cfg["passage_pfx"]
        bs       = cfg["batch_size"]

        print(f"\n{'=' * 60}", flush=True)
        print(f"  {key}  ({mname})", flush=True)
        print(f"  Prefix: {cfg['prefix_note']}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # ── Qwen3 version gate ────────────────────────────────────────────────
        if key == "QWEN3":
            ver_ok, ver_msg = _check_qwen3()
            print(f"  Version check: {ver_msg}", flush=True)
            if not ver_ok:
                print(f"  ERROR: Cannot load Qwen3 -- skipping (no silent fallback).")
                for qid in review_ids:
                    results[qid][key] = {"rank": None, "total": 0, "skipped": True, "error": ver_msg}
                cost[key] = {"error": ver_msg}
                continue

        # ── Load model ────────────────────────────────────────────────────────
        gc.collect()
        rss_before = _rss_mb()
        t_load = time.perf_counter()
        try:
            model = SentenceTransformer(mname, **cfg.get("load_kwargs", {}))
        except Exception as exc:
            err = str(exc)
            print(f"  ERROR loading model: {err}", flush=True)
            for qid in review_ids:
                results[qid][key] = {"rank": None, "total": 0, "skipped": True, "error": err}
            cost[key] = {"error": err}
            continue
        load_s    = time.perf_counter() - t_load
        rss_after = _rss_mb()
        rss_delta = (rss_after - rss_before) if (rss_before and rss_after) else None

        print(f"  Loaded in {load_s:.1f}s  "
              f"RAM delta: {f'+{rss_delta:.0f} MB' if rss_delta else 'N/A'}  "
              f"(RSS now: {f'{rss_after:.0f} MB' if rss_after else 'N/A'})",
              flush=True)

        # ── Encode all document corpora ───────────────────────────────────────
        doc_embs: Dict[str, np.ndarray] = {}
        doc_times: Dict[str, float]     = {}
        total_chunks = 0

        for doc_id, chunks in doc_chunks.items():
            texts = [ppfx + c["text"] for c in chunks]
            t0    = time.perf_counter()
            embs  = _encode_batch(model, texts, bs)
            elapsed = time.perf_counter() - t0
            doc_embs[doc_id]  = embs
            doc_times[doc_id] = elapsed
            cps = len(chunks) / elapsed if elapsed > 0 else 0
            total_chunks += len(chunks)
            print(f"    Encoded {doc_id[:20]}: "
                  f"{len(chunks)} chunks in {elapsed:.1f}s ({cps:.0f} ch/s)",
                  flush=True)

        total_embed_s = sum(doc_times.values())
        overall_cps   = total_chunks / total_embed_s if total_embed_s > 0 else 0

        # ── Per-query latency (5 samples, median) ─────────────────────────────
        sample_text = query_fn(by_id[review_ids[0]]["question"])
        q_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _encode_batch(model, [sample_text], bs)
            q_times.append(time.perf_counter() - t0)
        per_q_ms = statistics.median(q_times) * 1000

        cost[key] = {
            "load_s":       load_s,
            "rss_delta_mb": rss_delta,
            "rss_total_mb": rss_after,
            "doc_times":    doc_times,
            "total_embed_s": total_embed_s,
            "overall_cps":  overall_cps,
            "per_q_ms":     per_q_ms,
        }
        print(f"  Per-query median: {per_q_ms:.1f} ms", flush=True)

        # ── Rank each question ────────────────────────────────────────────────
        for doc_id, qids in by_doc.items():
            chunks     = doc_chunks[doc_id]
            chunk_embs = doc_embs[doc_id]
            for qid in qids:
                cids = correct_ids_map[qid]
                if not cids:
                    results[qid][key] = {"rank": None, "total": len(chunks), "skipped": False}
                    continue
                q_emb = _encode_batch(model, [query_fn(by_id[qid]["question"])], bs)[0]
                rank, total = _best_rank(q_emb, chunk_embs, cids, chunks)
                results[qid][key] = {"rank": rank, "total": total, "skipped": False}
                flags = "  ".join(
                    f"top-{t}={'Y' if rank and rank <= t else 'N'}"
                    for t in TOP_K
                )
                print(f"    [{qid}]  rank={rank}/{total}  {flags}", flush=True)

        del model
        gc.collect()

    # ── OUTPUT SECTION ────────────────────────────────────────────────────────
    SEP = "=" * 84

    # -- Per-question rank table -----------------------------------------------
    print()
    print(SEP)
    print("PER-QUESTION RANK TABLE  (cosine nearest-neighbor, 1 = most similar)")
    print(SEP)
    print()
    hdr = f"{'id':<34} {'type':<12}"
    for k in keys:
        hdr += f"  {k+'-rank':<10} {'t20':<4} {'t50':<4}"
    print(hdr)
    print("-" * len(hdr))
    for qid in review_ids:
        q   = by_id[qid]
        row = f"{qid:<34} {q['question_type']:<12}"
        for k in keys:
            r    = results[qid].get(k, {})
            rank = r.get("rank")
            tot  = r.get("total", "?")
            skip = r.get("skipped", False)
            if skip:
                row += f"  {'N/A':<10} {'N/A':<4} {'N/A':<4}"
            else:
                rs  = f"{rank}/{tot}" if rank else f">{tot}"
                t20 = "Y" if rank and rank <= 20 else "N"
                t50 = "Y" if rank and rank <= 50 else "N"
                row += f"  {rs:<10} {t20:<4} {t50:<4}"
        print(row)

    # -- Cost table ------------------------------------------------------------
    print()
    print(SEP)
    print("COST TABLE  (CPU-only; WorkingSet RAM via Windows ctypes)")
    print(SEP)
    print()

    # Header: doc columns abbreviated
    doc_labels = [f"{did.split('_')[0]}({len(doc_chunks[did])}ch)" for did in doc_ids]
    print(f"  {'Embedder':<28} {'load_s':<8} {'RAM_delta':<10} {'RAM_total':<10}"
          + "".join(f"  {lbl:<17}" for lbl in doc_labels)
          + f"  {'cps':<7} {'q_ms':<8}")
    print(f"  {'-' * (28+8+10+10 + len(doc_ids)*19 + 15)}")

    for cfg in EMBEDDERS:
        k = cfg["key"]
        c = cost.get(k, {})
        label = f"{k}({cfg['model_name'].split('/')[-1]})"
        if "error" in c:
            print(f"  {label:<28} ERROR: {c['error'][:55]}")
            continue
        load_s    = f"{c['load_s']:.1f}s"
        rd        = f"+{c['rss_delta_mb']:.0f}MB" if c.get("rss_delta_mb") else "N/A"
        rt        = f"{c['rss_total_mb']:.0f}MB"  if c.get("rss_total_mb") else "N/A"
        doc_cols  = "".join(
            f"  {c['doc_times'].get(did, 0):.1f}s ({len(doc_chunks[did])/max(c['doc_times'].get(did,1),0.001):.0f}/s)  "
            for did in doc_ids
        )
        cps       = f"{c['overall_cps']:.0f}/s"
        qms       = f"{c['per_q_ms']:.1f}ms"
        print(f"  {label:<28} {load_s:<8} {rd:<10} {rt:<10}{doc_cols}{cps:<7} {qms:<8}")

    # -- Rescue summary --------------------------------------------------------
    print()
    print(SEP)
    print("RESCUE SUMMARY  (of 7 baseline needs_review questions)")
    print(SEP)
    print()
    print(f"  {'Embedder':<38}  {'top-20 / 7':<12}  {'top-50 / 7'}")
    print(f"  {'-' * 60}")
    for cfg in EMBEDDERS:
        k  = cfg["key"]
        if cost.get(k, {}).get("error"):
            label = f"{k} ({cfg['model_name']}) -- FAILED"
            print(f"  {label:<38}  {'N/A':<12}  N/A")
            continue
        t20 = sum(
            1 for qid in review_ids
            if not results[qid].get(k, {}).get("skipped")
            and isinstance(results[qid].get(k, {}).get("rank"), int)
            and results[qid][k]["rank"] <= 20
        )
        t50 = sum(
            1 for qid in review_ids
            if not results[qid].get(k, {}).get("skipped")
            and isinstance(results[qid].get(k, {}).get("rank"), int)
            and results[qid][k]["rank"] <= 50
        )
        label = f"{k} ({cfg['model_name']})"
        print(f"  {label:<38}  {str(t20)+'/7':<12}  {t50}/7")

    # -- Quality vs cost verdict -----------------------------------------------
    print()
    print(SEP)
    print("QUALITY vs COST VERDICT  (target: CPU-only Space, 16 GB RAM budget)")
    print(SEP)
    print()

    for cfg in EMBEDDERS:
        k = cfg["key"]
        c = cost.get(k, {})
        if "error" in c:
            print(f"  {k}: FAILED TO LOAD")
            print(f"    Error: {c['error']}")
            print()
            continue
        t20 = sum(
            1 for qid in review_ids
            if not results[qid].get(k, {}).get("skipped")
            and isinstance(results[qid].get(k, {}).get("rank"), int)
            and results[qid][k]["rank"] <= 20
        )
        t50 = sum(
            1 for qid in review_ids
            if not results[qid].get(k, {}).get("skipped")
            and isinstance(results[qid].get(k, {}).get("rank"), int)
            and results[qid][k]["rank"] <= 50
        )
        max_doc_s  = max(c.get("doc_times", {1: 0}).values())
        rss_delta  = c.get("rss_delta_mb", 0) or 0
        per_q_ms   = c.get("per_q_ms", 0)

        rss_d_str = f"+{rss_delta:.0f} MB" if rss_delta else "N/A"
        rss_t_str = f"{c['rss_total_mb']:.0f} MB" if c.get("rss_total_mb") else "N/A"
        print(f"  {k} ({cfg['model_name']}):")
        print(f"    Quality : top-20={t20}/7  top-50={t50}/7")
        print(f"    Load    : {c['load_s']:.1f}s  RAM delta: {rss_d_str}  RSS total: {rss_t_str}")
        print(f"    Encode  : max {max_doc_s:.1f}s/doc  {c['overall_cps']:.0f} chunks/s overall")
        print(f"    Query   : {per_q_ms:.1f} ms/question")
        if max_doc_s > 60:
            print(f"    WARNING : {max_doc_s:.0f}s per-doc embedding is too slow for interactive upload")
        elif max_doc_s > 30:
            print(f"    NOTE    : {max_doc_s:.0f}s per-doc embedding is noticeable but tolerable at upload time")
        print()

    # Compute combined score for recommendation
    viable: List[Tuple[str, int, float, float]] = []
    for cfg in EMBEDDERS:
        k = cfg["key"]
        c = cost.get(k, {})
        if "error" in c:
            continue
        t20 = sum(
            1 for qid in review_ids
            if not results[qid].get(k, {}).get("skipped")
            and isinstance(results[qid].get(k, {}).get("rank"), int)
            and results[qid][k]["rank"] <= 20
        )
        max_doc_s = max(c.get("doc_times", {1: 0}).values())
        rss_delta = c.get("rss_delta_mb", 0) or 0
        viable.append((k, t20, max_doc_s, rss_delta))

    # Sort: more rescues first; among equal rescues, lower embed time
    viable.sort(key=lambda x: (-x[1], x[2]))

    print(f"  RECOMMENDATION:")
    if viable:
        best_k, best_t20, best_doc_s, best_ram = viable[0]
        best_cfg = next(c for c in EMBEDDERS if c["key"] == best_k)
        print(f"    Best quality-per-cost: {best_k} ({best_cfg['model_name']})")
        print(f"    Rescues {best_t20}/7 questions to top-20  |  "
              f"max embed {best_doc_s:.1f}s/doc  |  RAM delta ~{best_ram:.0f} MB")
        # Contextual note
        for k, t20, doc_s, ram in viable:
            if k != best_k and t20 == best_t20:
                print(f"    Note: {k} matches rescue count but costs more "
                      f"({doc_s:.1f}s/doc, +{ram:.0f} MB RAM)")
    print()
    print(SEP)


if __name__ == "__main__":
    main()
