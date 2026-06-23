"""Unit tests for eval/metrics.py using hand-built fake records with known outputs.

No real PDFs, no main.py import, no network. All expected values are derived
by hand and documented inline.

Run from repo root:
    python -m pytest eval/test_metrics.py -v
"""
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so eval.metrics is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import (
    abstention_accuracy,
    false_abstention_rate,
    first_hit_rank,
    is_retrieval_hit,
    key_fact_match_rate,
    latency_percentiles,
    mrr,
    normalize_number,
    normalize_text,
    recall_at_k,
)

# ── Fake chunk fixtures ───────────────────────────────────────────────────────

# Matches hint "total premium income of Rs. 21,433 crores"
C_HIT_A = {
    "text": "The total premium income of Rs. 21,433 crores was recorded in FY2024.",
    "page": 12,
    "chunk_id": 0,
}

# Matches hint "net profit margin in the motor segment was 4.2"
C_HIT_B = {
    "text": "The net profit margin in the motor segment was 4.2% in FY2024.",
    "page": 23,
    "chunk_id": 3,
}

# Miss chunks — do not match any hint used in these tests
C_MISS_1 = {"text": "Operating expenses increased year over year.", "page": 5, "chunk_id": 1}
C_MISS_2 = {"text": "Market presence expanded in rural areas.", "page": 7, "chunk_id": 2}
C_MISS_3 = {"text": "Board of directors approved the annual report.", "page": 9, "chunk_id": 4}

# Chunk for page-only fallback tests
C_PAGE = {"text": "Various financial metrics overview.", "page": 15, "chunk_id": 6}


def _rec(**kwargs) -> dict:
    """Build a minimal valid record; callers override only relevant fields."""
    base = {
        "id": "t",
        "answer_type": "answerable",
        "question_type": "lexical",
        "mode": "faiss_reranker",
        "supporting_text_hint": "",
        "supporting_pages": [],
        "key_facts": [],
        "ranked_chunks": [],
        "abstained": False,
        "answer": None,
        "latency_ms": None,
    }
    base.update(kwargs)
    return base


# ── normalize_text ────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("UPPER CASE") == "upper case"

    def test_collapses_whitespace(self):
        assert normalize_text("Multiple   spaces  here") == "multiple spaces here"

    def test_strips_trailing_punctuation(self):
        assert normalize_text("  Hello, World!  ") == "hello, world"

    def test_strips_leading_brackets(self):
        assert normalize_text("[Page 12]") == "page 12"

    def test_interior_punctuation_preserved(self):
        # Dots and commas inside the string are not stripped
        assert normalize_text("Rs. 21,433 crores") == "rs. 21,433 crores"

    def test_strips_percent(self):
        # Trailing % is punctuation and gets stripped
        assert normalize_text("98.5%") == "98.5"

    def test_empty_string(self):
        assert normalize_text("") == ""


# ── normalize_number ──────────────────────────────────────────────────────────

class TestNormalizeNumber:
    def test_rs_dot_no_space(self):
        assert normalize_number("Rs.21,433") == "21433"

    def test_rs_dot_with_space(self):
        assert normalize_number("Rs. 21,433") == "21433"

    def test_dollar_decimal(self):
        assert normalize_number("$1,234.56") == "1234.56"

    def test_rupee_symbol(self):
        assert normalize_number("₹1,00,000") == "100000"

    def test_plain_number(self):
        assert normalize_number("21433") == "21433"

    def test_empty_string(self):
        assert normalize_number("") == ""


# ── is_retrieval_hit ──────────────────────────────────────────────────────────

class TestIsRetrievalHit:
    def test_primary_text_hint_match(self):
        record = _rec(
            supporting_text_hint="total premium income of Rs. 21,433 crores",
            supporting_pages=[12],
        )
        assert is_retrieval_hit(C_HIT_A, record) is True

    def test_primary_text_hint_no_match(self):
        record = _rec(
            supporting_text_hint="total premium income of Rs. 21,433 crores",
            supporting_pages=[99],          # different page so fallback also fails
        )
        assert is_retrieval_hit(C_MISS_1, record) is False

    def test_page_fallback_empty_hint(self):
        # Empty hint → skip primary check → use page
        record = _rec(supporting_text_hint="", supporting_pages=[15])
        assert is_retrieval_hit(C_PAGE, record) is True
        assert is_retrieval_hit(C_MISS_1, record) is False   # page 5 not in [15]

    def test_page_fallback_hint_mismatch(self):
        # Non-empty hint that does NOT appear in the chunk text → falls back to page
        chunk = {"text": "Operational efficiency improved this quarter.", "page": 20, "chunk_id": 7}
        record = _rec(
            supporting_text_hint="phrase that does not appear in chunk text",
            supporting_pages=[20],
        )
        assert is_retrieval_hit(chunk, record) is True   # saved by page fallback

    def test_whitespace_normalization_newline_in_chunk(self):
        # Chunk text has a mid-phrase newline and a double space; hint uses single spaces.
        # normalize_text must collapse ALL whitespace runs (including \n) on BOTH sides
        # before the substring comparison so this is a genuine hit.
        chunk = {
            "text": "Provision \nCoverage Ratio of 67.86 per cent as of March 2024.",
            "page": 41,
            "chunk_id": 99,
        }
        record = _rec(
            supporting_text_hint="Provision Coverage Ratio of",
            supporting_pages=[99],   # wrong page — must win via text, not page
        )
        assert is_retrieval_hit(chunk, record) is True


# ── first_hit_rank ────────────────────────────────────────────────────────────

class TestFirstHitRank:
    def test_hit_at_rank_1(self):
        r = _rec(
            supporting_text_hint="total premium income of Rs. 21,433 crores",
            supporting_pages=[12],
            ranked_chunks=[C_HIT_A, C_MISS_1, C_MISS_2],
        )
        assert first_hit_rank(r) == 1

    def test_hit_at_rank_3(self):
        r = _rec(
            supporting_text_hint="net profit margin in the motor segment was 4.2",
            supporting_pages=[23],
            ranked_chunks=[C_MISS_1, C_MISS_2, C_HIT_B],
        )
        assert first_hit_rank(r) == 3

    def test_no_hit(self):
        r = _rec(
            supporting_text_hint="solvency ratio improved by 15 basis points",
            supporting_pages=[99],
            ranked_chunks=[C_MISS_1, C_MISS_2, C_MISS_3],
        )
        assert first_hit_rank(r) is None

    def test_empty_chunks(self):
        r = _rec(supporting_text_hint="anything", ranked_chunks=[])
        assert first_hit_rank(r) is None


# ── recall_at_k and mrr ───────────────────────────────────────────────────────
#
# Three records:
#   r1 — hit at rank 1  →  contributes to recall@1, recall@3, recall@5; RR = 1/1
#   r2 — hit at rank 3  →  misses recall@1; hits recall@3 and recall@5;  RR = 1/3
#   r3 — no hit         →  misses all recall; RR = 0
#
# recall@1 = 1/3
# recall@3 = 2/3
# recall@5 = 2/3   (r2 hit is at rank 3, within top 5; no new hits between 3–5)
# MRR      = (1 + 1/3 + 0) / 3 = 4/9 ≈ 0.4444

class TestRecallAndMRR:
    @pytest.fixture
    def three_records(self):
        r1 = _rec(
            id="r1",
            supporting_text_hint="total premium income of Rs. 21,433 crores",
            supporting_pages=[12],
            ranked_chunks=[C_HIT_A, C_MISS_1, C_MISS_2, C_MISS_3],
        )
        r2 = _rec(
            id="r2",
            supporting_text_hint="net profit margin in the motor segment was 4.2",
            supporting_pages=[23],
            ranked_chunks=[C_MISS_1, C_MISS_2, C_HIT_B, C_MISS_3],
        )
        r3 = _rec(
            id="r3",
            supporting_text_hint="solvency ratio improved by 15 basis points",
            supporting_pages=[99],
            ranked_chunks=[C_MISS_1, C_MISS_2, C_MISS_3],
        )
        return [r1, r2, r3]

    def test_recall_at_1(self, three_records):
        assert recall_at_k(three_records, 1) == pytest.approx(1 / 3)

    def test_recall_at_3(self, three_records):
        assert recall_at_k(three_records, 3) == pytest.approx(2 / 3)

    def test_recall_at_5(self, three_records):
        assert recall_at_k(three_records, 5) == pytest.approx(2 / 3)

    def test_mrr(self, three_records):
        # (1/1 + 1/3 + 0) / 3 = 4/9
        assert mrr(three_records) == pytest.approx(4 / 9)

    def test_recall_empty_list(self):
        assert recall_at_k([], 3) == 0.0

    def test_mrr_empty_list(self):
        assert mrr([]) == 0.0

    def test_recall_via_page_fallback_only(self):
        # A record with no hint but correct page — must count as a hit
        r = _rec(supporting_text_hint="", supporting_pages=[15], ranked_chunks=[C_PAGE])
        assert recall_at_k([r], 1) == pytest.approx(1.0)


# ── abstention_accuracy ───────────────────────────────────────────────────────
#
# 3 unanswerable records, 2 abstained → 2/3 ≈ 0.6667
# 1 answerable record mixed in       → ignored by the function

class TestAbstentionAccuracy:
    def test_two_of_three_abstained(self):
        records = [
            _rec(id="u1", answer_type="unanswerable", abstained=True),
            _rec(id="u2", answer_type="unanswerable", abstained=True),
            _rec(id="u3", answer_type="unanswerable", abstained=False),
            _rec(id="a1", answer_type="answerable",   abstained=False),   # ignored
        ]
        assert abstention_accuracy(records) == pytest.approx(2 / 3)

    def test_no_unanswerable_records(self):
        records = [_rec(answer_type="answerable", abstained=True)]
        assert abstention_accuracy(records) == 0.0

    def test_empty_list(self):
        assert abstention_accuracy([]) == 0.0

    def test_all_correct(self):
        records = [_rec(answer_type="unanswerable", abstained=True) for _ in range(4)]
        assert abstention_accuracy(records) == pytest.approx(1.0)


# ── false_abstention_rate ─────────────────────────────────────────────────────
#
# 4 answerable records, 1 wrongly abstained → 1/4 = 0.25
# 1 unanswerable record mixed in            → ignored

class TestFalseAbstentionRate:
    def test_one_of_four_wrong(self):
        records = [
            _rec(id="a1", answer_type="answerable",   abstained=False),
            _rec(id="a2", answer_type="answerable",   abstained=False),
            _rec(id="a3", answer_type="answerable",   abstained=True),    # wrong
            _rec(id="a4", answer_type="answerable",   abstained=False),
            _rec(id="u1", answer_type="unanswerable", abstained=True),    # ignored
        ]
        assert false_abstention_rate(records) == pytest.approx(1 / 4)

    def test_no_answerable_records(self):
        records = [_rec(answer_type="unanswerable", abstained=True)]
        assert false_abstention_rate(records) == 0.0

    def test_empty_list(self):
        assert false_abstention_rate([]) == 0.0

    def test_none_abstained(self):
        records = [_rec(answer_type="answerable", abstained=False) for _ in range(3)]
        assert false_abstention_rate(records) == pytest.approx(0.0)


# ── key_fact_match_rate ───────────────────────────────────────────────────────
#
# Spec requires:
#   answer "… Rs. 21,433 …"  with key_fact "21433" → MATCH (via normalize_number)
#   answer "… Rs. 20,000 …"  with key_fact "21433" → NO MATCH
#   → rate = 1/2 = 0.5

class TestKeyFactMatchRate:
    def test_numeric_fact_match_and_no_match(self):
        records = [
            _rec(
                id="k1",
                answer="The total premium was Rs. 21,433 crores in FY2024.",
                key_facts=["21433"],
            ),   # "21433" found after normalize_number strips comma + "Rs."
            _rec(
                id="k2",
                answer="The total premium was Rs. 20,000 crores in FY2024.",
                key_facts=["21433"],
            ),   # "21433" absent → no match
        ]
        assert key_fact_match_rate(records) == pytest.approx(1 / 2)

    def test_all_facts_must_match(self):
        # Both "motor" and "98.5%" must appear; missing one fails the record
        r_both = _rec(
            answer="Motor segment claims settlement ratio was 98.5% in FY2024.",
            key_facts=["98.5%", "motor"],
        )
        r_one_missing = _rec(
            answer="Claims settlement ratio was 98.5% in FY2024.",  # "motor" absent
            key_facts=["98.5%", "motor"],
        )
        assert key_fact_match_rate([r_both]) == pytest.approx(1.0)
        assert key_fact_match_rate([r_one_missing]) == pytest.approx(0.0)

    def test_empty_key_facts_is_vacuous_match(self):
        r = _rec(answer="Some answer.", key_facts=[])
        assert key_fact_match_rate([r]) == pytest.approx(1.0)

    def test_none_answer_excluded(self):
        # answer=None records must be skipped
        records = [
            _rec(answer=None, key_facts=["21433"]),
            _rec(answer="Premium was Rs. 21,433 crores.", key_facts=["21433"]),
        ]
        assert key_fact_match_rate(records) == pytest.approx(1.0)   # only second qualifies

    def test_unanswerable_excluded(self):
        records = [
            _rec(answer_type="unanswerable", answer="Information not found.", key_facts=[]),
            _rec(answer_type="answerable",   answer="Premium was Rs. 21,433 crores.", key_facts=["21433"]),
        ]
        assert key_fact_match_rate(records) == pytest.approx(1.0)

    def test_no_qualifying_records(self):
        assert key_fact_match_rate([]) == 0.0


# ── latency_percentiles ───────────────────────────────────────────────────────
#
# [10, 20, 30, 40, 50], n=5:
#   p50: ceil(0.50 × 5) = 3 → sv[2] = 30.0
#   p95: ceil(0.95 × 5) = ceil(4.75) = 5 → sv[4] = 50.0
#
# [100, 200, …, 1000], n=10:
#   p50: ceil(0.50 × 10) = 5 → sv[4] = 500.0
#   p95: ceil(0.95 × 10) = ceil(9.5) = 10 → sv[9] = 1000.0

class TestLatencyPercentiles:
    def test_five_values(self):
        result = latency_percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
        assert result["p50"] == 30.0
        assert result["p95"] == 50.0
        assert result["n"] == 5

    def test_ten_values(self):
        result = latency_percentiles([100.0 * i for i in range(1, 11)])
        assert result["p50"] == 500.0
        assert result["p95"] == 1000.0
        assert result["n"] == 10

    def test_single_value(self):
        result = latency_percentiles([42.0])
        assert result["p50"] == 42.0
        assert result["p95"] == 42.0
        assert result["n"] == 1

    def test_unsorted_input(self):
        # Function must sort internally
        result = latency_percentiles([50.0, 10.0, 30.0, 40.0, 20.0])
        assert result["p50"] == 30.0
        assert result["p95"] == 50.0

    def test_empty_list(self):
        assert latency_percentiles([]) == {"p50": 0.0, "p95": 0.0, "n": 0}
