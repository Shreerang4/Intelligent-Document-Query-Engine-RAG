# Persistence Schema

This document describes the deployed persistence schema for the RAG backend.
The live FastAPI pipeline now writes ingested documents, page-aware chunks,
query answers, claim-verification payloads, and source citations to these
tables. The production Hugging Face Space uses this schema through managed
MySQL/Aiven.

## Database Target

Production persistence uses a managed MySQL database supplied through the
`DATABASE_URL` environment variable, for example with a `mysql+pymysql://...`
SQLAlchemy URL.

Managed MySQL requires TLS. Set `DB_CA_CERT` to the local CA certificate path so
SQLAlchemy/PyMySQL can connect with certificate verification enabled. The CA
certificate file is local deployment material and must not be committed.

For Hugging Face Docker Space deployment, do not commit the CA certificate.
Encode the CA certificate locally and store the base64 string as the
`DB_CA_CERT_B64` Space secret. On startup, the app decodes that secret into a
temporary CA file and uses it for SQLAlchemy SSL verification.

Required Hugging Face Space secrets for the deployed app are:

- `DATABASE_URL`
- `DB_CA_CERT_B64`
- `GROQ_API_KEY`
- `API_TOKEN`

Deployment to the Hugging Face Space is manual through the `hf` git remote;
there is no GitHub auto-sync workflow in this repo. The production Space is
deployed with persistence enabled, and staging remains available separately as a
rollback target.

SQLite is only the local/development fallback for smoke testing and quick schema
checks. If `DATABASE_URL` is not set, the persistence layer defaults to
`sqlite:///./rag_persistence.db`.

Set `DATABASE_URL` through environment variables in deployment. For local
development it may be placed in `.env`, but `.env` must never be committed
because it can contain database credentials.

## User Identity

The schema includes `user_id` on every user-owned table before OAuth exists. For
the current deployed app, `persistence.user_context.get_current_user_id()`
returns the stable placeholder `local-dev-user`.

Adding `user_id` from day one prevents a later migration where historical
documents, chunks, queries, and citations would need to be backfilled or
repartitioned by owner. When OAuth is added, it should replace only
`get_current_user_id()` with a real authenticated identity lookup; the schema
stays unchanged.

## Why Eval Tables Are Excluded

Evaluation reports, benchmark questions, metrics, and experiment artifacts are
not part of the application persistence model and are intentionally not stored
in MySQL. They remain file-based under `eval/` and generated reports remain
ignored under `eval/results/`.

Keeping eval data out of this schema avoids mixing production user history with
benchmark-only records and keeps this slice focused on app-level document and
query history.

## Persistent Embeddings And In-Memory FAISS

Upload ingestion stores one float32 E5 embedding beside each chunk. Each vector
is serialized with `numpy.ndarray.tobytes()` and restored with
`numpy.frombuffer(..., dtype=np.float32)`. The dimension and dtype are stored
separately and validated before vectors are assembled into a contiguous matrix.

FAISS itself remains process-local and in memory. On a RAM cache miss and MySQL
hit, chunks are loaded strictly by `chunk_index`, their embeddings are restored,
and the existing exact `IndexFlatL2` index is rebuilt without calling
`embed_documents`.

Legacy chunk rows have nullable embedding columns. The first reuse of a legacy
document embeds its stored chunk strings once and backfills those columns.

## Dedup Behavior

Document deduplication uses `(user_id, source_hash)`, but `source_hash` is
computed differently by ingestion path:

- Upload ingestion uses the SHA-256 hash of the uploaded PDF bytes.
- URL ingestion uses a hash of the existing document cache key, because the URL
  loader does not retain raw PDF bytes after chunking.

Known limitation: the same PDF ingested once by upload and once by URL will not
deduplicate across those paths because the two source hashes differ. This is an
accepted tradeoff for now to avoid refactoring the URL loader and cache flow.

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

Indexes:

- `documents(user_id, created_at)`
- `documents(user_id, source_hash)` for persistent artifact lookup

### `chunks`

Stores page-aware text chunks for a document.

- `user_id` references `users.id`
- `document_id` references `documents.id`
- chunk identity: `chunk_id`, `chunk_index`
- location/content: `page_number`, `text`, `text_hash`, `char_count`
- embedding artifact: nullable `embedding_blob`, `embedding_dimension`,
  `embedding_dtype` (`float32`)
- timestamp: `created_at`

Indexes:

- `chunks(user_id, document_id)`
- unique `chunks(document_id, chunk_id)`

### `queries`

Stores one question/answer result against a persisted document.

- `user_id` references `users.id`
- `document_id` references `documents.id`
- answer fields: `question`, `answer`, `status`, `is_abstained`
- optional retry identity: `request_id`, `request_index`
- optional structured verification payload: `claim_verifications_json`
- retrieval config: `embedding_model`, `retrieval_mode`, `reranker_model`,
  `k_initial`, `k_final`
- optional timing: `latency_ms`
- timestamp: `created_at`

Indexes/constraints:

- `queries(user_id, document_id, created_at)`
- unique `queries(user_id, request_id, request_index)`; existing rows have NULL
  request fields and do not conflict

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

## Upload Persistence And Recovery

New upload documents, chunks, embeddings, queries, and citations are committed
in one transaction after all answers have been generated. Existing documents
commit each new answer batch and its citations in one transaction before the
HTTP response is returned. A failed transaction is rolled back completely while
the already-generated answer is still returned.

When an upload includes `request_id`, its question order is persisted through
`request_index`. A retry with the same document and questions reconstructs the
committed API response without extraction, embedding, retrieval, reranking, or
Groq calls. Reusing the ID for different input returns HTTP 409.

Concurrent inserts of the same upload are serialized in MySQL with a named
advisory lock derived from `(user_id, source_hash)`. The document lookup index is
not unique because historical duplicate documents may already exist.

## Aiven MySQL Migration

`Base.metadata.create_all()` does not alter existing tables. Apply
`migrations/mysql/001_persistent_embeddings_and_request_recovery.sql` to the
Aiven database before deploying this application version.

1. Back up the production database and apply the migration to staging first.
2. Connect with the MySQL client using Aiven's host, port, username, database,
   and CA certificate. Enter the password interactively.
3. From the MySQL prompt, execute:

```sql
SOURCE migrations/mysql/001_persistent_embeddings_and_request_recovery.sql;
```

4. Verify the result:

```sql
SHOW COLUMNS FROM chunks LIKE 'embedding_%';
SHOW COLUMNS FROM queries LIKE 'request_%';
SHOW INDEX FROM documents WHERE Key_name = 'ix_documents_user_id_source_hash';
SHOW INDEX FROM queries WHERE Key_name = 'uq_queries_user_request_index';
```

The migration checks `information_schema` before every change, so it can be run
again safely after an interrupted deployment.
