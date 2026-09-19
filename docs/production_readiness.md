# Production Readiness: Authentication and Ownership

This document records the Phase 7A configuration audit and the real-MySQL
verification procedure. It does not authorize a production deployment or a
production run of migration 002.

## Refresh Rotation Transaction Invariant

Refresh rotation depends on MySQL/InnoDB row locking. One SQLAlchemy `Session`
must execute the following sequence without an intervening commit:

1. `SELECT refresh_sessions ... WHERE token_hash = :digest FOR UPDATE`
2. validate expiry and revocation state
3. load the owning user in the same transaction
4. insert and flush exactly one successor using the predecessor's absolute
   `expires_at`
5. set the predecessor's `revoked_at` and `replaced_by_session_id`
6. flush and commit

`rotate_refresh_session_in_transaction()` stages steps 1–5. The service wrapper
or HTTP refresh endpoint owns the single commit. Any refresh-service,
SQLAlchemy, or JWT-configuration failure rolls the transaction back.

Correctness requires `users` and `refresh_sessions` to use InnoDB. With two
connections rotating R1, the second locking read waits for the first
transaction, then observes the committed revocation. The required final graph
is one revoked R1 pointing to one active R2; branching to both R2 and R3 is not
valid.

## Disposable MySQL Verification

The integration suite is intentionally destructive and skips unless
`MYSQL_TEST_DATABASE_URL` is set. As an additional guard, the selected database
name must contain `test`, `testing`, or `disposable`. Never point it at
production or a database containing useful data.

For a TLS-enabled test server, set `MYSQL_TEST_CA_CERT` to its CA certificate.
The tests never print the URL, password, or certificate.

```powershell
$env:MYSQL_TEST_DATABASE_URL="mysql+pymysql://.../idqe_auth_test"
$env:MYSQL_TEST_CA_CERT="C:\path\to\test-ca.pem" # when required
.venv\Scripts\python.exe -m pytest -q -m mysql_integration
```

The suite:

- creates the auth schema and confirms `users` and `refresh_sessions` are
  InnoDB;
- forces two independent connections to overlap on the same R1 locking read;
- proves the second operation is waiting before allowing the first to commit;
- verifies exactly one success, one revoked-token failure, and one successor;
- verifies a staged rotation can be rolled back without a successor or partial
  predecessor update;
- builds a representative pre-auth schema, inserts disposable placeholder
  history, runs migration 002 twice, and inspects its columns, indexes, foreign
  keys, delete rules, cleanup, and InnoDB engines;
- mounts the real auth router against the migrated database and verifies
  registration and login.

If the environment variable is absent, skipped tests do not constitute MySQL or
migration verification.

## Production Configuration Audit

### Access JWT

- `ACCESS_JWT_SECRET` is required when JWT functionality is used and must be at
  least 32 UTF-8 bytes. There is no default signing secret.
- Only HS256 is accepted.
- `ACCESS_TOKEN_TTL_SECONDS` defaults to 600 seconds.
- `sub`, `iat`, `exp`, `jti`, `iss`, and `aud` are required.
- Signature, expiration, issued-at, issuer
  `intelligent-document-query-engine`, and audience
  `intelligent-document-query-engine-api` are validated.
- `API_TOKEN` is never reused as the JWT secret.

### Refresh Cookie

- Name: `idqe_refresh`
- `HttpOnly=true`
- `Secure=true` by default; production must not set
  `REFRESH_COOKIE_SECURE=false`
- `SameSite=Lax`
- `Path=/auth`
- no `Domain` attribute, making it host-only
- `Expires` and `Max-Age` use the refresh session's remaining absolute expiry;
  rotation does not extend the seven-day lifetime

### Origin and CORS

- Every state-changing auth endpoint retains the Origin/Referer validation
  dependency.
- Same-origin browser requests are accepted.
- A separate production frontend origin must be listed exactly in
  `AUTH_ALLOWED_ORIGINS`; wildcard values are invalid.
- Built-in local-development allowances are exact localhost/127.0.0.1 origins
  on ports 3000 and 5173. They do not match arbitrary hosts or production
  domains.
- CORS credentials remain enabled only with an explicit origin list. The list
  is never `*`.
- The production frontend remains same-origin by default; no CORS relaxation is
  required for bearer JWTs.

### Database TLS

- Every configured MySQL URL requires `DB_CA_CERT` or `DB_CA_CERT_B64`.
- PyMySQL receives the CA through `ssl.ca`, enabling certificate validation.
- `DB_ALLOW_LOCAL_TEST_CERT_HOSTNAME_MISMATCH=true` may disable only hostname
  matching for loopback databases explicitly named as test/disposable. The CA
  and signature remain verified, and the override is rejected for production
  hosts or database names.
- The base64 deployment secret is decoded to `/tmp/aiven-ca.pem`, permissioned
  `0600`, and its contents are never logged.
- Confirm the production provider's current CA is installed and that all
  user/auth tables report `ENGINE=InnoDB` before rollout.

### Operational Database Health Token

`API_TOKEN` protects only `GET /health/db`. It is read when that endpoint is
used, not while importing the application. Missing configuration returns a safe
503; a wrong token returns 401. The token is compared in constant time and
cannot authenticate RAG/history endpoints. Conversely, a user JWT cannot access
`/health/db`. Public `GET /health` remains independent of this token.

### Private Object Storage

The asynchronous document upload route retains source PDFs through the storage
abstraction. The synchronous RAG routes remain unchanged. Before enabling the
asynchronous upload path in an environment:

- With `OBJECT_STORAGE_BACKEND=local`, use an absolute durable path outside
  publicly served directories and mount the same contents into API and worker
  processes.
- With `OBJECT_STORAGE_BACKEND=s3`, enforce private access using the provider's
  public-access controls and bucket/account policy. The application sends no
  object ACL, never requests `public-read`, and constructs no public URL.
- Store explicit S3 credentials as secrets. If they are absent, boto3 uses its
  standard provider chain, allowing IAM roles and workload identity.
- Grant the runtime principal only the required object-prefix get/put/delete
  permissions. Database backups contain metadata, not the PDF bytes.
- Source PDFs are intended to remain available for a later authenticated,
  short-lived signed-URL flow. No viewing or signed-URL endpoint exists yet.

### RabbitMQ, Celery, and Async Upload

`POST /documents/upload` stores private PDFs and publishes document IDs for the
separate worker. Existing synchronous routes remain unchanged.

- Run the ingestion worker as a separate process using
  `celery -A backend.app.celery_app worker --loglevel=INFO --concurrency=1`.
- Configure `CELERY_BROKER_URL` independently for every publisher and worker;
  do not assume RabbitMQ is colocated with FastAPI.
- Keep the initial concurrency at one because each worker child can load an E5
  model and PDF embedding is CPU- and RAM-intensive.
- With local object storage, mount the same durable object volume at the same
  configured path in the API and worker. With S3-compatible storage, give both
  processes access to the same private bucket and prefix.
- RabbitMQ messages contain only the opaque `document_id`. MySQL remains the
  authority for ownership, object metadata, lifecycle status, and artifacts.
- Celery has no result backend. Alert on worker/task failures and inspect the
  document's MySQL status rather than polling Celery results.
- A retry countdown is represented by `queued`; active work is represented by
  `processing`. The default permits three retries after the initial attempt.
- Broker publisher errors return a structured 503 while leaving the committed
  document queued and retaining its PDF. Do not mark ambiguous publication
  failures as failed.
- Clients should always send a stable `upload_request_id`. Retrying it repairs
  the remaining database-commit/publish crash window without creating another
  document. No transactional outbox or queued-row reconciler exists yet.
- Apply migration 005 before enabling the endpoint against an existing MySQL
  database.

See [`document_ingestion_worker.md`](document_ingestion_worker.md) for the full
task contract and local Compose instructions.

## Accepted Account-State Staleness

Normal RAG/history requests use `get_authenticated_user_id()`. This validates
the signed JWT and returns its canonical UUID without querying `users`.
Consequently, a deleted or later-disabled account may continue using an
already-issued token until its remaining lifetime expires, bounded by the
configured access-token TTL (600 seconds by default).

This is intentional. `/auth/me` and future sensitive account/security
operations can use the separate DB-backed `get_current_user()` dependency when
fresh account state is required. Do not add a user lookup to every ordinary
request solely to remove this accepted window.

## Browser Content Security Policy

FastAPI adds the following enforcing header to production responses:

```text
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-src 'none'; frame-ancestors 'none'; worker-src 'none'
```

- `default-src 'self'` supplies a restrictive same-origin fallback.
- `script-src 'self'` permits only the built Vite JavaScript bundle.
- `style-src 'self'` permits only the built CSS asset. The frontend uses system
  fonts and has no inline or third-party stylesheet requirement.
- `connect-src 'self'` permits the same-origin auth, RAG, history, and health
  requests. Groq and model traffic is server-side and is intentionally absent.
- `img-src 'self'` and `font-src 'self'` prevent third-party resource loading.
- `object-src 'none'`, `frame-src 'none'`, and `worker-src 'none'` disable
  browser capabilities the application does not use.
- `base-uri 'self'`, `form-action 'self'`, and `frame-ancestors 'none'` prevent
  hostile base URLs, cross-origin form targets, and framing.

Neither `'unsafe-inline'` nor `'unsafe-eval'` is allowed. FastAPI also returns
`X-Content-Type-Options: nosniff`. Vite development documents are served by
Vite and do not receive this production CSP, so HMR does not require weakening
the deployed policy. Proxied API responses may include the header, but a CSP is
enforced from the document response.

CSP is defense-in-depth. User questions, PDF excerpts, LLM answers, filenames,
URLs, and backend messages must continue to be rendered through React text
interpolation; CSP does not replace avoiding raw HTML execution sinks.

## Pre-Rollout Gates

Before production rollout:

1. Run the real InnoDB concurrency test and migration rehearsal against the
   dedicated disposable database.
2. Back up production and verify a rollback path.
3. Confirm production environment variables without printing their values.
4. Verify InnoDB engines and MySQL TLS with read-only queries.
5. Apply migration 002 under a separately approved change window.
6. Apply migration 003 before enabling any route that persists object metadata.
7. Apply migration 004 before invoking queue-independent ingestion so completed
   legacy rows use `ready` and newly created rows default to `queued`.
8. Verify the object bucket or mounted directory is private and durable from
   every process that will use it.
9. Perform staging registration, login, refresh, logout, upload, URL query, and
   two-user ownership smoke tests before deployment.
