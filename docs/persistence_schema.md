# Persistence schema and transaction boundaries

Production uses Aiven MySQL/InnoDB through SQLAlchemy. The database contains authentication state, owned document metadata/lifecycle, chunks and persisted vectors, and query/citation history. Original PDF bytes for async uploads live in private object storage; FAISS lives in API process memory.

The schema source of truth is `persistence/models.py`, with MySQL upgrades under `migrations/mysql/`. Importing the app does not create/alter tables. `python scripts/init_db.py` explicitly calls `Base.metadata.create_all()` for a new database; it cannot upgrade existing tables. SQLite is a development/test fallback and does not provide the production InnoDB row-lock semantics.

## Tables and constraints

| Table | Stored data | Key constraints/indexes |
| --- | --- | --- |
| `users` | UUID identity, normalized email, display name, Argon2id password hash, auth provider, timestamps | Primary `id`; unique `email` (nullable for historical compatibility) |
| `refresh_sessions` | Owner, 32-byte SHA-256 token digest, absolute expiry, revocation time, successor ID, creation time | Unique `token_hash`; indexes on `expires_at` and `(user_id, revoked_at, expires_at)`; user FK `ON DELETE CASCADE`, successor FK `ON DELETE SET NULL` |
| `documents` | Owner, input identity, source-object metadata, upload UUID, lifecycle, retrieval configuration, timestamps | `(user_id, created_at)`; nonunique `(user_id, source_hash)`; unique nullable `object_key`; unique `(user_id, upload_request_id)` |
| `chunks` | Owner/document, logical `chunk_id`, ordered `chunk_index`, page number, text/hash/length, vector bytes/dimension/dtype | `(user_id, document_id)`; unique `(document_id, chunk_id)` |
| `queries` | Owner/document, question, answer, status, abstention flag, verification JSON, retrieval config, timing, creation time, legacy request fields | `(user_id, document_id, created_at)`; unique nullable `(user_id, request_id, request_index)` |
| `citations` | Owner/query/document, optional stored-chunk FK, response-facing chunk ID, rank, page, excerpt, optional scores, creation time | `(user_id, query_id)` and `(query_id, rank)` |

Documents, chunks, queries and citations carry owner foreign keys; child resources also reference their parents. Most of those foreign keys do not specify database `ON DELETE CASCADE`; ORM relationships and the special refresh-session delete rules should not be confused with a universal SQL cascade policy. There is no document deletion HTTP API.

Password/email columns permit historical nulls at schema level; registration requires a valid email/password. Raw passwords and raw refresh tokens are not stored. Query/citation payloads and source text are stored as application data, not encrypted by application code. Authentication details live in [security](security.md).

## Document fields

| Group | Fields and semantics |
| --- | --- |
| Identity | `id`, `user_id`, `source_type`, `filename`, `source_url`, `source_hash`, `cache_key` |
| Source object | Nullable `object_key`, `content_type`, `byte_size`; populated by async uploads, normally null for legacy/URL rows |
| Upload recovery | Nullable `upload_request_id`, a canonical UUID scoped by user |
| Lifecycle | `status`: queued/processing/ready/failed; nullable `error_message` |
| Retrieval provenance | `embedding_model`, `embedding_format`, `retrieval_mode`, `reranker_model`, `k_initial`, `k_final` |
| Timestamps | `created_at`, `updated_at` |

The status column is a string, not a database enum or complete transition constraint. Application code governs lifecycle. Worker ingestion writes the embedding model/format actually used; query configuration is also recorded on query rows. Operators must align worker and query embedding settings.

Private keys have the form `documents/<generated-document-uuid>/source.pdf`, without original filenames or user identifiers. `object_key` is internal metadata, not a public URL, and is absent from status/upload API projections. MySQL holds no PDF blob. The object remains after successful ingestion and after terminal ingestion failure.

## Async upload idempotency and duplicate content

The service validates input and computes SHA-256 of the exact PDF bytes, then tries `(user_id, upload_request_id)` recovery **before** content detection or a new object write. Same UUID/same bytes returns the original row; different bytes conflict. A unique owned UUID constraint resolves concurrent inserts. A losing object write is deleted best effort. Multiple NULL UUIDs are allowed; clients that omit the field lose this recovery identity.

Content lookup filters `(user_id, source_hash)` to queued/processing/ready rows. Ordering is active queued/processing first, then descending creation time and document ID. Failed matches do not block new attempts. A usable match produces `duplicate_document` unless `allow_duplicate=true` was supplied. That flag bypasses only hash lookup; it never bypasses request recovery.

**`source_hash` is not unique**, globally or within a user. Intentional and historical duplicates remain valid. This async content check does not take a named advisory lock across lookup and insert, so two new UUIDs can both pass concurrently. It is a user-choice safeguard, not a strict content uniqueness guarantee.

Upload and URL ingestion use different hash identities: uploads hash raw bytes; the legacy URL path hashes its URL/model cache key. The same PDF obtained by URL and uploaded as bytes need not match. The content check is not cross-user or cross-path semantic deduplication.

## Committed artifact format

Each chunk has text, one-based page number, logical chunk ID and zero-based contiguous `chunk_index`. Vectors are contiguous float32 arrays serialized with `numpy.ndarray.tobytes()`, with dimension and dtype alongside the blob. Loading uses `numpy.frombuffer(..., dtype=np.float32)` and validates ordering, dimensions, dtype and byte length.

Worker completion locks the document row, preserves an already-ready winner, replaces that document's chunks/vectors, and commits `ready` in the same transaction. Computation occurs outside that transaction. A retry does not expose a committed partial chunk set. [Worker details](document_ingestion_worker.md) explain repeated computation and lifecycle races.

The async ready-document query loader requires matching model/input format and complete vectors. It rebuilds `IndexFlatL2` from persisted vectors on a RAM miss without re-embedding documents. It does not repair legacy/mismatched vectors. The synchronous upload route separately supports embedding backfill for stored text under a MySQL named advisory lock. These are different recovery paths.

## Transaction boundaries by API path

| Path | Persistence boundary | Recovery |
| --- | --- | --- |
| `POST /documents/upload` | Source PUT, then queued metadata commit, then broker publish; three independent boundaries | Owned upload UUID recovery and conditional queued republication |
| Celery ingestion | Processing transition commits separately; final chunks/vectors/ready commit together | Redelivery/retry; already-ready no-op |
| `POST /documents/{id}/queries` | Answer computed first; query plus citations committed in a best-effort background task | No query request key; successful answer does not prove history committed |
| `POST /hackrx/run` | Best-effort document/chunk persistence and background query history | URL/model RAM caching; no retained source object through async storage |
| `POST /hackrx/upload-run` | New document/chunks/vectors and answer batch/citations committed in one transaction after generation; existing-document answer batches are atomic | Optional legacy `request_id`; committed response reconstructed for matching bytes and ordered questions |
| Registration/login/refresh | Credential/session changes and rotation committed transactionally | InnoDB locked refresh rotation; see security guide |

For the legacy synchronous upload path, persistence failure can still return the generated answer without committed history. Its `(user_id, request_id, request_index)` constraint is distinct from `documents.upload_request_id`. Reuse with different bytes/questions returns 409; recovery only works when the earlier result committed. MySQL advisory locks serialize legacy upload persistence by user/hash; do not attribute those locks to the async hash precheck.

## Ownership and cache

Parent resource lookup uses both ID and user ID before querying children. Chunk/vector reads and citation linking filter by the owner again. Persistence requires users to exist and never creates placeholder users as a side effect. Identical content across users produces separate logical resources.

RAM/FAISS keys carry `DocumentCacheKey(user_id, resource_key)` and cache entries also carry the owner. Capacity/TTL are process-wide. Database artifacts are authoritative for reconstruction; FAISS files are not a durable application store. Query history retains excerpts and response chunk IDs even when an optional stored-chunk link is absent.

Evaluation labels/results are file-based under `eval/`, not production DB tables. Generated `eval/results/` reports are ignored; the committed [evaluation summary](retrieval_evaluation.md) preserves recorded measurements.

## MySQL migration history

Existing MySQL schemas must be checked and upgraded in numeric order. The migration files target MySQL 8+ and use metadata checks/repeatable operations; this is not proof that every migration was rehearsed against every deployed database.

| Migration | Change |
| --- | --- |
| `001_persistent_embeddings_and_request_recovery.sql` | Nullable chunk vector columns, query request fields, hash lookup and response-recovery indexes |
| `002_multi_user_auth.sql` | Password/email identity changes, refresh sessions and InnoDB requirements; removes legacy `local-dev-user` and its dependent data |
| `003_document_object_metadata.sql` | Nullable object key/content type/byte size and unique object-key index |
| `004_document_ingestion_lifecycle.sql` | Converts legacy `ingested` to `ready`; sets queued as default |
| `005_document_upload_idempotency.sql` | Nullable upload UUID and unique `(user_id, upload_request_id)` index |

Migration 002 intentionally deletes placeholder-era citations, queries, chunks, documents, refresh sessions and the placeholder user in foreign-key-safe order; it does not reassign that history. No migration was added for content duplicate detection, and historical duplicates are not merged or deleted. Comments inside older migrations describe their original scope and are retained as history.

Back up the database and rehearse migrations against a disposable/staging schema before applying them operationally. Use a MySQL client with verified TLS and enter credentials privately. From the repository root in the client:

```sql
SOURCE migrations/mysql/001_persistent_embeddings_and_request_recovery.sql;
SOURCE migrations/mysql/002_multi_user_auth.sql;
SOURCE migrations/mysql/003_document_object_metadata.sql;
SOURCE migrations/mysql/004_document_ingestion_lifecycle.sql;
SOURCE migrations/mysql/005_document_upload_idempotency.sql;
```

Useful read-only verification:

```sql
SHOW COLUMNS FROM chunks LIKE 'embedding_%';
SHOW COLUMNS FROM queries LIKE 'request_%';
SHOW INDEX FROM documents WHERE Key_name = 'ix_documents_user_id_source_hash';
SHOW INDEX FROM queries WHERE Key_name = 'uq_queries_user_request_index';
SHOW COLUMNS FROM users LIKE 'password_hash';
SHOW INDEX FROM users WHERE Key_name = 'uq_users_email';
SHOW CREATE TABLE refresh_sessions;
SHOW COLUMNS FROM documents WHERE Field IN ('object_key', 'content_type', 'byte_size');
SHOW INDEX FROM documents WHERE Key_name = 'uq_documents_object_key';
SHOW COLUMNS FROM documents LIKE 'status';
SHOW COLUMNS FROM documents LIKE 'upload_request_id';
SHOW INDEX FROM documents WHERE Key_name = 'uq_documents_user_upload_request_id';
SELECT COUNT(*) FROM documents WHERE status = 'ingested';
SELECT COUNT(*) FROM users WHERE id = 'local-dev-user';
```

The guarded real-MySQL suite rehearses migration 002 and refresh locking specifically; it is not an end-to-end rehearsal of all five migrations. See [production readiness](production_readiness.md). Database TLS/configuration and HF release commands are centralized in [deployment](deployment.md).
