
# Baseline Evaluation Report

**Run date:** 2026-06-23 12:27 UTC
**Questions:** 33 across 3 document(s)
**LLM (Groq):** ON

## Retrieval Metrics (answerable questions only)

| Mode | R@3 | R@5 | MRR | Retr p50 | Retr p95 |
|------|-----|-----|-----|----------|----------|
| faiss_reranker | 54.2% | 62.5% | 0.474 | 46 ms | 109 ms |
| faiss_only | 37.5% | 50.0% | 0.315 | 158 ms | 217 ms |
| bm25 | 41.7% | 41.7% | 0.329 | 47 ms | 82 ms |

## Retrieval by Question Type (faiss_reranker, answerable only)

| Question Type | n | R@3 | R@5 | MRR |
|---------------|---|-----|-----|-----|
| conceptual | 6 | 16.7% | 33.3% | 0.117 |
| distractor | 6 | 100.0% | 100.0% | 0.889 |
| lexical | 6 | 100.0% | 100.0% | 0.833 |
| paraphrase | 6 | 0.0% | 16.7% | 0.058 |

## LLM Answer Quality (faiss_reranker mode)

| Metric | Value |
|--------|-------|
| Key-fact match rate | 20.8% |
| Abstention accuracy (unanswerable Qs) | 100.0% |
| False abstention rate (answerable Qs) | 8.3% |
| LLM latency p50 | 1497 ms |
| LLM latency p95 | 10311 ms |

## Cold vs Warm Retrieval Latency (faiss_reranker)

> **Cold** = first faiss_reranker query per document (may include model-load overhead).  
> **Warm** = all subsequent queries on the same document's FAISS index.

| Bucket | n | p50 | p95 |
|--------|---|-----|-----|
| cold (1st per doc) | 3 | 54 ms | 169 ms |
| warm | 30 | 42 ms | 90 ms |

## Questions Needing Review

**7 answerable question(s)** returned zero retrieval hits across all three modes:

- **bajaj_finance_ar_2024_25_q06** (`conceptual`): Which figures indicate that troubled credit stayed very low for the company?
  - hint: `consolidated gross NPA at 0.96% and net NPA at 0.44% were amongst the lowest`
  - pages: [32]
- **hdfc_bank_ar_2024_25_q03** (`paraphrase`): How much did the Bank put toward social responsibility work during the financial year?
  - hint: `CSR Spend`
  - pages: [205]
- **hdfc_bank_ar_2024_25_q04** (`paraphrase`): How large was HDFC Bank's regular workforce as of March 31, 2025?
  - hint: `employees on the rolls of the Bank`
  - pages: [265]
- **hdfc_bank_ar_2024_25_q05** (`conceptual`): What percentage indicates how much protection the Bank had set aside against possible credit losses?
  - hint: `Provision Coverage Ratio of`
  - pages: [38]
- **hdfc_bank_ar_2024_25_q06** (`conceptual`): By which fiscal year does HDFC Bank want its own operations to stop adding net carbon impact?
  - hint: `become carbon-neutral`
  - pages: [3]
- **infosys_ar_2024_25_q03** (`paraphrase`): How much cash did Infosys report generating after business operations and investment in assets during FY2025?
  - hint: `another year of strong execution`
  - pages: [17]
- **infosys_ar_2024_25_q04** (`paraphrase`): How many people were included in Infosys's learning outreach efforts?
  - hint: `initiatives to include 13.3 million people`
  - pages: [81]
