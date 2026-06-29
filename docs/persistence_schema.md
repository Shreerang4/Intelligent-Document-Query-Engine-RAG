# Persistence Schema

This document describes the first persistence slice for the RAG backend. The
schema exists independently of the live FastAPI pipeline: ingestion, retrieval,
answer generation, and citation generation are not writing to these tables yet.

## Database Target

Intended production persistence uses a managed MySQL database supplied through
the `DATABASE_URL` environment variable, for example with a
`mysql+pymysql://...` SQLAlchemy URL.

SQLite is only the local/development fallback for smoke testing and quick schema
checks. If `DATABASE_URL` is not set, the persistence layer defaults to
`sqlite:///./rag_persistence.db`.

Set `DATABASE_URL` through environment variables in deployment. For local
development it may be placed in `.env`, but `.env` must never be committed
because it can contain database credentials.

## User Identity

The schema includes `user_id` on every user-owned table before OAuth exists. For
local development, `persistence.user_context.get_current_user_id()` returns the
stable placeholder `local-dev-user`.

Adding `user_id` now prevents a later migration where historical documents,
chunks, queries, and citations would need to be backfilled or repartitioned by
owner. When OAuth is added, it should replace only `get_current_user_id()` with a
real authenticated identity lookup; the schema stays unchanged.

## Why Eval Tables Are Excluded

Evaluation reports, benchmark questions, metrics, and experiment artifacts are
not part of the application persistence model. They remain file-based under
`eval/` and generated reports remain ignored under `eval/results/`.

Keeping eval data out of this schema avoids mixing production user history with
benchmark-only records and keeps this slice focused on app-level document and
query history.

## Why Embeddings And FAISS Blobs Are Excluded

This slice stores source text and retrieval metadata, not vector artifacts.
Embeddings and FAISS indexes are intentionally excluded because:

- they can be regenerated from chunks and the recorded embedding config,
- they are large compared with the relational metadata,
- their binary format and compatibility depend on model/index implementation,
- cache invalidation is already tied to embedding model and input format.

Future persistence can add a dedicated vector store or artifact cache if needed.

## Tables

### `users`

Stores the application user identity.

- `id` primary key
- optional profile/auth fields: `email`, `display_name`, `auth_provider`
- timestamps: `created_at`, `updated_at`

### `documents`

Stores one ingested PDF per user and the retrieval configuration used for it.

- `user_id` references `users.id`
- input metadata: `source_type`, `filename`, `source_url`, `source_hash`,
  `cache_key`
- lifecycle fields: `status`, `error_message`
- retrieval config: `embedding_model`, `embedding_format`, `retrieval_mode`,
  `reranker_model`, `k_initial`, `k_final`
- timestamps: `created_at`, `updated_at`

Index: `documents(user_id, created_at)`.

### `chunks`

Stores page-aware text chunks for a document.

- `user_id` references `users.id`
- `document_id` references `documents.id`
- chunk identity: `chunk_id`, `chunk_index`
- location/content: `page_number`, `text`, `text_hash`, `char_count`
- timestamp: `created_at`

Indexes:

- `chunks(user_id, document_id)`
- unique `chunks(document_id, chunk_id)`

### `queries`

Stores one question/answer result against a persisted document.

- `user_id` references `users.id`
- `document_id` references `documents.id`
- answer fields: `question`, `answer`, `status`, `is_abstained`
- optional structured verification payload: `claim_verifications_json`
- retrieval config: `embedding_model`, `retrieval_mode`, `reranker_model`,
  `k_initial`, `k_final`
- optional timing: `latency_ms`
- timestamp: `created_at`

Index: `queries(user_id, document_id, created_at)`.

### `citations`

Stores source chunks returned for a query.

- `user_id` references `users.id`
- `query_id` references `queries.id`
- `document_id` references `documents.id`
- optional chunk relation: `chunk_db_id`
- response-facing source fields: `chunk_id`, `rank`, `page_number`, `excerpt`
- optional scores: `retrieval_score`, `reranker_score`
- timestamp: `created_at`

Indexes:

- `citations(user_id, query_id)`
- `citations(query_id, rank)`

## Relationships

- A user has many documents, chunks, queries, and citations.
- A document has many chunks, queries, and citations.
- A query has many citations.
- A citation may point to a stored chunk through `chunk_db_id`; it also stores
  response-facing `chunk_id` and excerpt so citations remain readable even if a
  chunk relationship is unavailable.
