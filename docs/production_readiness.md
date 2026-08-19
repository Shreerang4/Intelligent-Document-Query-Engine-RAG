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
6. Perform staging registration, login, refresh, logout, upload, URL query, and
   two-user ownership smoke tests before deployment.
