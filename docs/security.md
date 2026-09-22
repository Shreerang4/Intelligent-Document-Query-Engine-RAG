# Security boundaries

This document describes implemented controls and their limits. It is not a claim of penetration testing, compliance, or formal security certification. Production still depends on correct HTTPS, database, broker, bucket, credential and network configuration.

## Passwords and access tokens

Email addresses are validated/normalized with `email-validator` and deliverability checking disabled. There is no provider-specific dot/alias rewriting and no email-ownership verification flow. Password accounts store an Argon2id hash with a generated salt (`memory_cost=19456` KiB, `time_cost=2`, `parallelism=1`). Login can rehash an old hash when parameters change. Password hashes are excluded from public user schemas.

Access tokens use HS256 with required `sub`, `iat`, `exp`, `jti`, `iss`, and `aud` claims. Validation checks signature, timestamps, issuer/audience and UUID-shaped subject/token identifiers. The default lifetime is 600 seconds. `ACCESS_JWT_SECRET` must contain at least 32 UTF-8 bytes; there is no default signing key. The operational `API_TOKEN` is separate and only authorizes `/health/db`.

React holds the access token in each tab's memory, not localStorage/sessionStorage or a shared cross-tab token cache. `authenticatedFetch` attaches it and, on a 401, refreshes and retries the request at most once. Page startup attempts refresh-cookie session restoration.

## Refresh credentials and concurrency

Each login/registration creates an opaque token with `secrets.token_urlsafe(32)`. Only the SHA-256 digest is stored as `refresh_sessions.token_hash`; the raw credential is delivered in a cookie and never returned in auth JSON. The application supports independent sessions per login/device, without device metadata or family-wide revocation.

Cookie attributes are shared between setting and clearing:

| Attribute | Behavior |
| --- | --- |
| Name | `idqe_refresh` |
| HttpOnly | Always enabled |
| Secure | Enabled by default; explicit false only for plain-HTTP development |
| SameSite | `Lax` |
| Path | `/auth` |
| Domain | Unset, making it host-only |
| Expiry | Remaining absolute seven-day session lifetime; rotation does not extend it |

There are three concurrency layers:

1. A same-tab `refreshPromise` collapses concurrent refresh calls.
2. When available, the browser Web Locks API holds `idqe-auth-refresh` around the actual refresh HTTP request, serializing cooperating same-origin tabs. Without Web Locks, requests are dispatched without that cross-tab layer.
3. The server selects the digest row `FOR UPDATE`, validates it, inserts a successor, revokes/links the predecessor, and commits in one MySQL/InnoDB transaction. A concurrent request using the already-rotated token must fail rather than create a second successor.

The row lock is the server correctness layer; SQLite tests do not prove this concurrency property. See [the guarded MySQL tests](production_readiness.md). Browser serialization is not a universal guarantee against every client or logout/refresh race.

Missing, unknown, expired, revoked and already-rotated refresh credentials produce the same public 401 message. An expired credential clears the cookie; unknown/revoked failures do not clear it, avoiding a late failed response erasing a newer cookie. Logout revokes the presented refresh session and clears the cookie. It does not revoke already-issued access JWTs.

## Transactions and response handling

Registration stages the new user and initial refresh session in one transaction; login stages any rehash and new refresh session. Both create the access token before commit. Refresh likewise creates the token before committing rotation, so configuration errors roll back rather than silently consume the refresh credential.

Successful register/login/refresh responses, logout, `/auth/me`, and invalid-refresh responses set `Cache-Control: no-store`. This is not a blanket statement about every validation/error response. Auth validation handling removes submitted `input` and `ctx` values from 422 errors. Credential-bearing configuration should never be logged during troubleshooting.

## Authorization and ownership

Ordinary document, RAG and history requests use the validated JWT subject as canonical `user_id`. They do not accept a user ID from a form, query string or task payload. Persistence filters private reads/writes by owner and parent resource; missing and differently owned resources use not-found behavior. Per-user request UUID namespaces and content hashes never deduplicate across users.

FAISS/document cache keys and entries carry owner identity. Async query entries also carry the document ID and embedding namespace. These are application-enforced boundaries; the schema does not use database row-level security. The process-wide cache capacity can be affected by other users' activity.

The normal JWT dependency does not look up current account state for every request. An issued token remains cryptographically valid until expiry even after logout or account removal; individual DB operations can still fail if the user/resource is gone. `/auth/me` uses a separate database-backed user dependency. The schema does not implement an account-disable workflow, per-token revocation list, or instant global logout.

## Browser protections

Every state-changing auth route checks Origin, falling back to Referer, against the request origin and configured exact allowed origins. Built-in allowances cover localhost/127.0.0.1 on ports 3000 and 5173. A `Sec-Fetch-Site: cross-site` request without either header is rejected. Non-browser requests without those metadata headers are permitted; this is not an authentication mechanism. CORS uses explicit origins with credentials enabled, never `*`.

FastAPI responses include `X-Content-Type-Options: nosniff` and the following enforcing CSP:

```text
default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-src 'none'; frame-ancestors 'none'; worker-src 'none'
```

There is no `unsafe-inline` or `unsafe-eval`. The production UI uses React text interpolation for document/model content, rather than raw HTML sinks. Vite serves its own development document, so production document CSP does not control HMR. CSP and escaping are defense in depth and do not make PDF/LLM content trusted.

## Storage, broker and inference trust boundaries

- Async source PDFs are stored under an opaque document UUID key, outside public frontend assets. Original filenames/user identity are not part of the object key. Local storage resolves paths under its root, writes a temporary file, fsyncs it, and replaces the destination; file permissions and mounted-volume protection remain deployment concerns.
- S3 calls do not set public ACLs or construct public URLs. Bucket/account policy must enforce privacy. There is no signed viewing/download endpoint, application encryption-key management, object lifecycle cleanup, or source deletion API.
- RabbitMQ contains document IDs, not PDF bytes, user IDs, credentials or object keys. JSON-only serialization narrows the task format; it does not authorize untrusted publishers. Broker and worker credentials are trusted internal capabilities.
- MySQL stores extracted text, vectors, questions, answers and citations as well as auth metadata. Only the password/refresh-token credentials are hashed; document content is not application-encrypted. Database and object backups/access policy matter independently.
- Selected PDF text and questions are sent to Groq for generation/verification. Model downloads and inference cross network boundaries. Do not claim data stays entirely inside the Space or that provider retention settings have been verified by this code.

Keep `.env` files, signing keys, Groq keys, database/broker URLs, B2 application credentials, operational tokens and certificate material out of git and the image build context. The repository supplies `.env.example` placeholders and development RabbitMQ credentials only. Use private hosting configuration and least-privilege database/object/broker access; the application does not provision provider policy.

The `.dockerignore` excludes the default `uploads/object-storage` tree, all directories named `uploads` or `object-storage`, local environment files except `.env.example` templates, and `certs/`. A custom storage root with a different name must also be kept outside the build context or explicitly excluded. These path rules do not identify arbitrary sensitive files placed elsewhere in the repository.

## Known security limits

The repository does not implement rate limiting, account email verification, password reset, MFA, malware scanning or OCR. The URL-ingestion route fetches user-selected HTTP(S) destinations, including redirects, without a dedicated private-address/SSRF deny policy; deployment network egress restrictions are important. Prompt instructions and claim verification do not eliminate prompt injection or false model outputs. There is no formal tenant-isolation certification or complete application audit log.

Public `/health` is a process/cache endpoint, while `/health/db` is a token-protected connectivity check. Neither attests to bucket privacy, queue health, worker progress or the security of external accounts. [Production readiness](production_readiness.md) separates repository checks from operational verification.
