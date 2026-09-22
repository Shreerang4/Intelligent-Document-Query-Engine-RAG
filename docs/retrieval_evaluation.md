# Retrieval Evaluation Summary

This document summarizes the completed retrieval-quality stage for the Intelligent Document Query Engine. It is a lightweight committed summary; generated JSON/Markdown reports under `eval/results/` remain ignored.

## Benchmark

- Corpus: annual reports for Infosys, HDFC Bank, and Bajaj Finance.
- Questions: 33 labeled benchmark questions: 24 answerable and 9 unanswerable, with 11 questions per document.
- Answerable question types: lexical, paraphrase, conceptual, and distractor (6 each). The remaining 9 are labeled unanswerable.
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

E5-small-v2 is the current default embedder in the GitHub repo. It improves R@5 from 62.5% to 70.8%, improves MRR, and reduces needs_review from 7 to 4. The recorded CPU ingestion was slower than MiniLM, so MiniLM remains an explicitly configured alternative/baseline; there is no automatic fallback on E5 failure.

## Final Shipped Configuration

- Embedder: `intfloat/e5-small-v2` by default.
- MiniLM alternative/baseline: `all-MiniLM-L6-v2` via `EMBEDDING_MODEL_NAME`.
- Retrieval mode: `faiss_reranker`.
- Initial FAISS candidates: `k_initial=20`.
- Final reranked chunks: `k_final=8`.
- Reranker: `cross-encoder/ms-marco-TinyBERT-L-2-v2`.

The default FAISS/reranker adapter and application path both use `k_final=8`. The experimental hybrid branch has separate defaults (`HYBRID_E5_K_INITIAL=30`, `HYBRID_BM25_K_INITIAL=20`, `HYBRID_K_FINAL=5`). Matching default retrieval settings does not mean these historical runs measure the async API, broker, database or deployed hardware.

Default E5 run:

```powershell
$env:EMBEDDING_MODEL_NAME='intfloat/e5-small-v2'
```

MiniLM alternative/baseline:

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

Use E5-small-v2 as the default retrieval embedder in the GitHub repo, while keeping MiniLM configurable as an alternative/baseline. Keep E5+BM25 hybrid as an ablation until there is a better merge/rerank strategy that improves q08 without losing E5 semantic rescues or adding unacceptable latency.

The Space is released manually through its separate `hf/main` history. Deployment commit and runtime checks are separate from retrieval evaluation; see [deployment](deployment.md).

## Metric definitions and measurement scope

The committed labels are in `eval/benchmark/questions.json`, scoring in `eval/metrics.py`, and orchestration in `eval/runner.py` / `eval/pipeline_adapter.py`.

- A retrieval hit is a normalized supporting-text substring in a chunk **or** a chunk whose page belongs to the labeled supporting pages. The page fallback can credit text that is on a relevant page without containing the exact answer.
- Recall@3/5 is the fraction of answerable questions with at least one hit in the first 3/5 ranked chunks. It is not the fraction of all relevant chunks retrieved. Unanswerable questions have no gold retrieval evidence and are excluded from recall/MRR.
- MRR averages reciprocal rank of the first hit over answerable questions, contributing zero for no hit in the returned ranking.
- `needs_review` identifies answerable questions with no hit across the returned rankings of all evaluated baseline modes. It is not simply `24 * (1 - Recall@5)` and is not an answer-factuality score.
- Latency percentiles use nearest rank. The harness times retrieval operations locally; model initialization may occur outside individual retrieval timers. Its first/subsequent-query comparison is not an HTTP cold-start measurement.
- `--no-llm` disables answer generation. It cannot establish abstention accuracy, key-fact answer correctness, citation faithfulness or claim-verification quality.

The adapter reads local PDFs and builds vectors/FAISS directly, bypassing HTTP/auth and the API document cache. The timing table preserves the previously committed measurements unchanged. There are no committed raw result reports under `eval/results/` beyond `.gitkeep`; that directory is ignored. Exact historical hardware, dependency/model revisions and full raw records are not captured in this summary, so the numbers are evidence of those recorded experiments rather than a reproducibility or capacity guarantee.

## Reproduction commands (not a production benchmark)

From an activated Python environment at the repository root, use one configuration per run. These commands run retrieval experiments; run them separately from routine tests and compare results only with their configuration and environment recorded.

```powershell
$env:RETRIEVAL_MODE='faiss_reranker'
$env:EMBEDDING_MODEL_NAME='all-MiniLM-L6-v2'
python eval/runner.py --no-llm --out eval/results/minilm

$env:EMBEDDING_MODEL_NAME='intfloat/e5-small-v2'
python eval/runner.py --no-llm --out eval/results/e5

$env:RETRIEVAL_MODE='e5_bm25_reranker'
python eval/runner.py --no-llm --out eval/results/e5_bm25
```

Model downloads and benchmark PDFs must be available locally. The GitHub tree contains the three annual-report PDFs; the HF deployment tree omits them. Keep generated reports out of commits unless intentionally selecting a sanitized, documented artifact for review.

## Limits of interpretation

This is a small, fixed financial-report corpus, with labels and repeated ablations that can favor choices specific to it. Broad page matching, extracted-table quality, cleaning of numeric/symbol-heavy lines, candidate selection, and reranker order affect results. Small-sample p95 is noisy. Recorded timings do not measure HTTP upload, object storage, queue wait, worker startup, database persistence, Groq generation/verification, concurrent users, or the live Space. Larger-model comparisons above are historical observations, not universal rankings or newly verified performance claims.
