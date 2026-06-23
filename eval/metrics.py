"""Pure metric functions for the RAG evaluation harness.

Stateless — no I/O, no network, no import of main.py.
Consumes lists of per-question result records produced by the runner.
"""
import math
import re
import string
from typing import Dict, List, Optional


def normalize_text(s: str) -> str:
    """Lowercase, collapse all whitespace to one space, and strip leading/trailing
    punctuation and whitespace.

    Applied to both sides of every substring comparison so capitalisation, extra
    spaces, and trailing periods do not cause false mismatches.
    """
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(string.punctuation + " ")


def normalize_number(s: str) -> str:
    """Strip currency text prefixes (Rs., $, ₹ …), commas, and whitespace so
    that formatted numbers compare equal to bare digit strings.

    Transforms:
        "Rs.21,433"   -> "21433"
        "Rs. 21,433"  -> "21433"
        "$1,234.56"   -> "1234.56"
        "₹1,00,000"   -> "100000"

    Apply to both the key_fact and the full answer string; non-numeric text in
    the answer is left in place — only currency marks and commas are removed so
    the numeric substring becomes findable.
    """
    s = re.sub(r"Rs\.?", "", s)        # remove "Rs" or "Rs."
    s = re.sub(r"[₹$£€¥]", "", s)      # remove Unicode currency symbols
    s = re.sub(r"[,\s]", "", s)        # remove commas and all whitespace
    return s.lstrip(".")               # clean up stray dot left by "Rs." removal


def is_retrieval_hit(chunk: Dict, record: Dict) -> bool:
    """Return True if chunk is a relevant result for the record.

    PRIMARY  — normalize_text(supporting_text_hint) is a substring of
               normalize_text(chunk["text"]), when the hint is non-empty.
               This is the main ground-truth signal for all real benchmark
               questions.
    FALLBACK — chunk["page"] is in record["supporting_pages"]. Used when the
               hint is absent or when encoding differences prevent text matching.

    Either condition is sufficient; primary is checked first.
    """
    hint = record.get("supporting_text_hint", "")
    if hint and normalize_text(hint) in normalize_text(chunk["text"]):
        return True
    return chunk["page"] in record.get("supporting_pages", [])


def first_hit_rank(record: Dict) -> Optional[int]:
    """Return the 1-indexed position of the first relevant chunk in ranked_chunks.

    Returns None if no chunk in the ranked list is a retrieval hit. Used by
    mrr() to compute the reciprocal rank contribution of a single record.
    """
    for rank, chunk in enumerate(record.get("ranked_chunks", []), start=1):
        if is_retrieval_hit(chunk, record):
            return rank
    return None


def recall_at_k(records: List[Dict], k: int) -> float:
    """Fraction of records that have at least one retrieval hit in the top-k chunks.

    The caller should pre-filter records to a single answer_type and retrieval
    mode before calling. Returns 0.0 for an empty list.
    """
    if not records:
        return 0.0
    hits = sum(
        1 for r in records
        if any(is_retrieval_hit(c, r) for c in r.get("ranked_chunks", [])[:k])
    )
    return hits / len(records)


def mrr(records: List[Dict]) -> float:
    """Mean Reciprocal Rank over the given records.

    Each record contributes 1/first_hit_rank to the sum (0 if no hit found).
    The mean is taken over all records. The caller should pre-filter to a single
    mode and answer_type. Returns 0.0 for an empty list.
    """
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        rank = first_hit_rank(r)
        if rank is not None:
            total += 1.0 / rank
    return total / len(records)


def abstention_accuracy(records: List[Dict]) -> float:
    """Fraction of UNANSWERABLE records where the system correctly abstained.

    Filters internally to answer_type == "unanswerable". A higher value means
    the system correctly refuses out-of-scope questions. Returns 0.0 when
    there are no unanswerable records in the list.
    """
    pool = [r for r in records if r.get("answer_type") == "unanswerable"]
    if not pool:
        return 0.0
    return sum(1 for r in pool if r.get("abstained", False)) / len(pool)


def false_abstention_rate(records: List[Dict]) -> float:
    """Fraction of ANSWERABLE records where the system wrongly abstained.

    Filters internally to answer_type == "answerable". A lower value is desired
    — a high rate means the system refuses questions it should be answering.
    Returns 0.0 when there are no answerable records in the list.
    """
    pool = [r for r in records if r.get("answer_type") == "answerable"]
    if not pool:
        return 0.0
    return sum(1 for r in pool if r.get("abstained", False)) / len(pool)


def _fact_present(fact: str, answer: str) -> bool:
    """True if fact appears in answer after text normalisation OR numeric normalisation."""
    if normalize_text(fact) in normalize_text(answer):
        return True
    norm_fact = normalize_number(fact)
    return bool(norm_fact) and norm_fact in normalize_number(answer)


def key_fact_match_rate(records: List[Dict]) -> float:
    """Fraction of ANSWERABLE records with a non-None answer where EVERY key_fact
    appears in the answer text.

    This is a strict, conservative presence check — NOT semantic grading:
      - Text facts  : normalize_text(fact) must be a substring of normalize_text(answer).
      - Numeric facts: normalize_number(fact) must be a substring of
        normalize_number(answer), so "21433" matches "Rs. 21,433" after stripping
        commas and currency symbols.
    A record passes only if ALL its key_facts match. An empty key_facts list
    counts as a pass (vacuously true). Filters to answerable records with a
    non-None answer; returns 0.0 when no qualifying records exist.
    """
    pool = [
        r for r in records
        if r.get("answer_type") == "answerable" and r.get("answer") is not None
    ]
    if not pool:
        return 0.0
    matches = sum(
        1 for r in pool
        if all(_fact_present(f, r["answer"]) for f in r.get("key_facts", []))
    )
    return matches / len(pool)


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Return p50, p95, and n for a list of per-query latency values in milliseconds.

    Uses the nearest-rank method: the pth percentile maps to
    sorted_values[ceil(p * n) - 1]. Note: p95 is noisy at small n (< ~20
    samples) — treat it as directional only until you have a larger sample.
    Returns {"p50": 0.0, "p95": 0.0, "n": 0} for an empty list.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "n": 0}
    n = len(latencies_ms)
    sv = sorted(latencies_ms)

    def _p(pct: float) -> float:
        return sv[min(math.ceil(pct * n) - 1, n - 1)]

    return {"p50": _p(0.50), "p95": _p(0.95), "n": n}
