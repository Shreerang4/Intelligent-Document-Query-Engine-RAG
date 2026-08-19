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
- `ACCESS_JWT_SECRET`
- `API_TOKEN` (operational `/health/db` diagnostic only)

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

The schema now supports password-authenticated users and opaque refresh
sessions. New `User` objects receive application-generated UUID IDs by default.
`email` stores the validated and normalized login identity directly and is
unique. The exact original capitalization or formatting is not preserved.
`auth_provider="password"` identifies password-authenticated accounts.

`password_hash` and `email` are nullable at the schema level during the phased
rollout. Registration requires both fields for password accounts. RAG and
history routes obtain identity exclusively from the validated access-JWT
subject, and persistence never creates users as a side effect.

New databases no longer seed `local-dev-user`. Migration
`002_multi_user_auth.sql` deletes that user and its existing citations, queries,
chunks, and documents. The legacy history is intentionally not reassigned or
preserved.

## Password Credential Services

Phase 2 adds service-layer credential handling without exposing HTTP auth
routes. Email identity follows one canonical flow:

`raw input -> validate_email(..., check_deliverability=False).normalized -> users.email`

The original capitalization/formatting is not retained, and no provider-specific
transformations such as Gmail dot removal or `+alias` removal are performed.
Registration and login both normalize before querying `users.email`.

Passwords are hashed with `argon2-cffi` using Argon2id, a library-generated
random salt, `memory_cost=19456` KiB, `time_cost=2`, and `parallelism=1`.
Successful login checks whether the stored hash needs rehashing and commits an
updated hash when parameters have changed. Password hashes are absent from the
public user schema.

## Access JWT Support

Phase 3 added short-lived HS256 access tokens and FastAPI authentication
dependencies. Phase 6B2 attaches the stateless identity dependency to every
user-facing RAG and history route. Tokens have a default 600-second lifetime and contain only
`sub`, `iat`, `exp`, `jti`, `iss`, and `aud`.

`ACCESS_JWT_SECRET` must contain at least 32 bytes and is read lazily when token
functionality is called, so existing evaluation and application imports do not
require the new secret. The normal `get_authenticated_user_id` dependency is
stateless and returns the signed JWT subject without querying MySQL. The
separate `get_current_user` dependency performs a user lookup only for operations
that require fresh account state. Ordinary stateless authentication therefore
accepts an account-state staleness window of at most approximately ten minutes.

`ACCESS_TOKEN_TTL_SECONDS` defaults to `600`. The signing secret is independent
from the operational `API_TOKEN`, which protects only `/health/db` and is never
used as application-user identity.

## Opaque Refresh-Session Services

Phase 4A adds service-layer refresh credentials.
Each credential is generated with `secrets.token_urlsafe(32)`. Only the raw
token's 32-byte SHA-256 digest is stored in `refresh_sessions.token_hash`; raw
tokens are returned to the internal caller and are never assigned to an ORM
column.

Each new login session has an absolute seven-day expiry. A user may hold
multiple independent refresh sessions for different browsers or devices.
Rotation locks the current row with `SELECT ... FOR UPDATE`, creates one
successor with the original absolute `expires_at`, revokes the predecessor, and
commits both changes atomically. Single-session revocation is idempotent; device
metadata and family-wide revocation are intentionally absent.

This guarantee requires MySQL/InnoDB and independent transactions for
concurrent requests. The locking read, validation, successor insert,
predecessor update, and commit remain one transaction. The skippable real-MySQL
race test and disposable migration rehearsal are documented in
[`production_readiness.md`](production_readiness.md).

## Authentication HTTP Boundary

Phase 4B exposes `/auth/register`, `/auth/login`, `/auth/refresh`,
`/auth/logout`, and `/auth/me`. Registration stages the normalized password
user and initial refresh session in one SQLAlchemy transaction, creates the
access token before commit, and rolls the entire unit back if any staged step
fails. Login similarly stages any Argon2 rehash and its new independent refresh
session before committing.

The raw refresh credential is transported only in the host-only
`idqe_refresh` cookie. Its shared setter/clearer configuration is `HttpOnly`,
`Secure` by default, `SameSite=Lax`, `Path=/auth`, and no `Domain`. Both
`Expires` and `Max-Age` reflect the remaining absolute database expiry;
rotation therefore does not extend the original seven-day lifetime. Plain-HTTP
local development must explicitly set `REFRESH_COOKIE_SECURE=false`.

All state-changing auth endpoints use a reusable origin validator. Browser
`Origin`, falling back to `Referer`, must match the request origin, an explicit
comma-separated `AUTH_ALLOWED_ORIGINS` entry, or one of the documented local
frontend origins. A request marked `Sec-Fetch-Site: cross-site` without either
header is rejected. Requests without browser origin metadata remain available
to non-browser clients. This is intentionally an additional CSRF layer beyond
CORS and `SameSite=Lax`.

Refresh errors collapse missing, unknown, expired, revoked, and already-rotated
credentials to the same public 401 response. Expired cookies are cleared, but
unknown or revoked cookies are not cleared by a failed refresh response: this
prevents a late failure for an old token from erasing a newer cookie installed
by a concurrent successful rotation.

## Explicit Persistent Ownership

Phase 6A removes the global persistence user context. Every document, chunk,
embedding, query, citation, history, and upload-recovery operation now receives
`user_id` explicitly. Persistence verifies that users already exist and never
creates placeholder or authentication rows as a side effect.

Document reuse is scoped by `(user_id, source_hash)`. Chunk and embedding reads
use both `user_id` and `document_id`; document-ID backfill also requires the
matching source hash. Query creation first resolves the document through
`WHERE documents.id = :document_id AND documents.user_id = :user_id`, and
citation chunk linking includes the same owner.

History document listings filter the base documents and both count subqueries
by user. Document-query and query-citation reads first resolve the parent using
the supplied owner, then apply `user_id` again to the child query. Missing and
differently owned parent identifiers therefore have identical internal
not-found behavior.

Upload idempotency remains a per-user namespace through
`(user_id, request_id, request_index)`. Recovery filters queries by
`(user_id, request_id)`, verifies the recovered document with
`(document_id, user_id)`, and filters citations by user, document, and query.
Identical PDF hashes belonging to different users produce separate logical
document rows.

The JWT-authenticated route boundary passes the validated `sub` claim into
these lower layers. Neither routes nor persistence contain a placeholder-user
fallback.

## User-Scoped RAM and FAISS State

Phase 6B1 scopes every in-memory document entry with the immutable structural
key `DocumentCacheKey(user_id, resource_key)`. The `resource_key` preserves the
existing URL/upload hash, embedding-model, and input-format namespace, while
`user_id` prevents identical URLs or PDF bytes from sharing chunks, document
IDs, or FAISS indexes across users.

`DocumentCacheEntry` also stores `user_id`. Central read/write helpers verify
that the supplied user, structural key owner, and entry owner agree. MySQL
artifact recovery already filters by user and now installs the reconstructed
FAISS index only under that user's structural key. Cache capacity, TTL, and LRU
eviction remain process-wide.

Phase 6B2 supplies the validated JWT subject to both cache and persistence
operations. The React RAG/history client uses the existing in-memory
`authenticatedFetch()` session, including its refresh-and-retry-once behavior;
there is no manual shared-token UI or client.

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

- `id` primary key; application-generated UUID for new users
- `email`, containing the validated and normalized identity, unique when non-NULL
- optional `display_name`
- `password_hash`, containing the Argon2id encoded password hash
- `auth_provider`; password accounts use `password`
- timestamps: `created_at`, `updated_at`

Constraints:

- unique `users(email)`; MySQL permits multiple NULL values, which
  keeps the phase-one schema compatible with non-authenticated transitional rows

### `refresh_sessions`

Stores only hashed opaque refresh credentials and their rotation state. No raw
refresh credential is persisted.

- `id` UUID primary key
- `user_id` references `users.id` with `ON DELETE CASCADE`
- `token_hash` is the unique 32-byte SHA-256 digest
- expiry/revocation fields: `expires_at`, nullable `revoked_at`
- `created_at`
- nullable `replaced_by_session_id` self-reference with `ON DELETE SET NULL`

Indexes and constraints:

- unique `refresh_sessions(token_hash)`
- `refresh_sessions(expires_at)` for expiry cleanup
- `refresh_sessions(user_id, revoked_at, expires_at)` for user-session lookup

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
- A user has many refresh sessions; deleting a user cascades to those sessions.
- A refresh session may identify the session that replaced it during rotation.
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

`Base.metadata.create_all()` does not alter existing tables. Existing databases
must apply the migrations in order:

1. `migrations/mysql/001_persistent_embeddings_and_request_recovery.sql`
2. `migrations/mysql/002_multi_user_auth.sql`

Migration 002 is intentionally destructive only for the legacy
`local-dev-user`. It deletes citations, queries, chunks, documents, any refresh
sessions, and finally the placeholder user in foreign-key-safe order. It does
not delete rows owned by any other user.

1. Back up the production database and apply both migrations to staging first.
2. Connect with the MySQL client using Aiven's host, port, username, database,
   and CA certificate. Enter the password interactively.
3. From the MySQL prompt, execute each required migration in numeric order:

```sql
SOURCE migrations/mysql/001_persistent_embeddings_and_request_recovery.sql;
SOURCE migrations/mysql/002_multi_user_auth.sql;
```

4. Verify the result:

```sql
SHOW COLUMNS FROM chunks LIKE 'embedding_%';
SHOW COLUMNS FROM queries LIKE 'request_%';
SHOW INDEX FROM documents WHERE Key_name = 'ix_documents_user_id_source_hash';
SHOW INDEX FROM queries WHERE Key_name = 'uq_queries_user_request_index';
SHOW COLUMNS FROM users LIKE 'password_hash';
SHOW INDEX FROM users WHERE Key_name = 'uq_users_email';
SHOW CREATE TABLE refresh_sessions;
SELECT COUNT(*) FROM users WHERE id = 'local-dev-user';
```

Both migrations are idempotent. Migration 002 checks `information_schema`
before altering `users`, uses `CREATE TABLE IF NOT EXISTS` for refresh sessions,
and can safely repeat the placeholder cleanup.
