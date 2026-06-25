# Retrieval Evaluation Summary

This document summarizes the completed retrieval-quality stage for the Intelligent Document Query Engine. It is a lightweight committed summary; generated JSON/Markdown reports under `eval/results/` remain ignored.

## Benchmark

- Corpus: annual reports for Infosys, HDFC Bank, and Bajaj Finance.
- Questions: 33 labeled benchmark questions.
- Question types: lexical, paraphrase, conceptual, and distractor.
- Modes measured: MiniLM baseline, E5-small-v2, and E5+BM25 hybrid ablation.
- Metrics: Recall@3, Recall@5, MRR, needs_review count, retrieval latency, and ingestion/indexing time.
- Evaluation setting: retrieval-only (`--no-llm`), so metrics measure retrieval/reranking without Groq answer generation.

## Final Metrics

| Configuration | R@3 | R@5 | MRR | needs_review | p50 | p95 | ingest/index time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM | 54.2% | 62.5% | 0.474 | 7 | 56 ms | 144 ms | 230.8s |
| E5-small-v2 | 58.3% | 70.8% | 0.496 | 4 | 73 ms | 156 ms | 534.0s |
| E5+BM25 hybrid | 58.3% | 70.8% | 0.504 | 5 | 216 ms | 527 ms | 510.0s |

## Decision

E5-small-v2 is the current default embedder in the GitHub repo. It improves R@5 from 62.5% to 70.8%, improves MRR, and reduces needs_review from 7 to 4. CPU ingestion is slower than MiniLM, so MiniLM remains available as a fallback/baseline for speed or cost-sensitive runs.

## Final Shipped Configuration

- Embedder: `intfloat/e5-small-v2` by default.
- MiniLM fallback: `all-MiniLM-L6-v2` via `EMBEDDING_MODEL_NAME`.
- Retrieval mode: `faiss_reranker`.
- Initial FAISS candidates: `k_initial=20`.
- Final reranked chunks: `k_final=8`.
- Reranker: `cross-encoder/ms-marco-TinyBERT-L-2-v2`.

The eval harness and production pipeline now use the same `k_final=8`, so the benchmark measures the shipped final-context configuration.

Default E5 run:

```powershell
$env:EMBEDDING_MODEL_NAME='intfloat/e5-small-v2'
```

MiniLM fallback/baseline:

```powershell
$env:EMBEDDING_MODEL_NAME='all-MiniLM-L6-v2'
```

The E5+BM25 hybrid is documented as an ablation, not as the default. It rescued one exact-table case but introduced a new regression and increased retrieval latency substantially.

## Target Checks

### E5 rescues over MiniLM

E5 rescued three MiniLM misses at hit@5:

- `bajaj_finance_ar_2024_25_q05`
- `infosys_ar_2024_25_q03`
- `infosys_ar_2024_25_q04`

These gains mostly came from better semantic/paraphrase retrieval.

### q08 exact-table regression

`bajaj_finance_ar_2024_25_q08` asks for total liabilities and equity from the consolidated balance sheet. MiniLM retrieved the exact supporting chunk at rank 1. E5-only missed it because it retrieved thematically similar balance-sheet and liability chunks instead of the exact table row.

The hybrid experiment fixed this case:

- Correct chunk: `2275`
- BM25 rank: 1
- Final hybrid reranked rank: 1
- hit@5: true

Hybrid was still not adopted because it did not improve overall R@5 and increased p50 latency from 73 ms to 216 ms versus E5-only.

### HDFC near-rescues

`hdfc_bank_ar_2024_25_q03`:

- Correct evidence entered the E5 candidate pool.
- In E5-only, correct chunks appeared below top-5 after reranking.
- In hybrid, correct chunks still finished outside top-5.
- Interpretation: reranker/final-selection limitation rather than an embedder-only miss.

`hdfc_bank_ar_2024_25_q05`:

- Correct evidence appears when the candidate pool is expanded.
- In targeted inspection, the correct chunk entered the merged pool but finished outside top-5.
- Interpretation: candidate-pool and reranker/final-selection limitation.

### Hybrid regression

Hybrid introduced a new regression versus E5-only:

- `infosys_ar_2024_25_q04`

This was one of the E5 rescues. Hybrid reranking changed the merged candidate ordering enough to lose hit@5.

## Larger Model Ablations

- GTE-base-en-v1.5 was slower and worse than E5-small-v2 in the local comparison.
- Qwen3-Embedding-0.6B was rejected because CPU ingestion was impractically slow for interactive uploads.

## Recommendation

Use E5-small-v2 as the default retrieval embedder in the GitHub repo, while keeping MiniLM configurable as the fallback/baseline. Keep E5+BM25 hybrid as an ablation until there is a better merge/rerank strategy that improves q08 without losing E5 semantic rescues or adding unacceptable latency.

The Hugging Face live demo may lag behind this GitHub repo until the Space is manually synced.
