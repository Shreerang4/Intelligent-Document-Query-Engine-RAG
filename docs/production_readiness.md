# Production readiness and verification

The application implements async upload, object-backed source retention, RabbitMQ/Celery ingestion, persisted embeddings, document queries, duplicate choice, multi-user ownership and rotating refresh sessions. The current HF deployment co-locates API and a concurrency-1 worker, with external CloudAMQP, Aiven, Backblaze B2 and Groq. See [deployment](deployment.md) for topology/configuration and [security](security.md) for implemented controls.

This document distinguishes repository checks from live operational evidence. A passing unit suite or successful git push does not establish provider configuration, worker progress, backup restorability, or production capacity. No current throughput/availability target is claimed.

## What the tests cover

| Area | Existing suite |
| --- | --- |
| Passwords, JWTs, auth HTTP, refresh transactions/cookies | `tests/test_auth_*.py`, `test_access_jwt.py`, `test_refresh_sessions.py` |
| Ownership, request recovery, stored vectors/history | `test_persistence_ownership.py`, `test_persistent_upload_storage.py`, `test_rag_jwt_authorization.py`, `test_upload_cache_flow.py` |
| Storage implementations and metadata | `test_object_storage.py`, `test_document_object_persistence.py` |
| Upload acceptance, duplicate checks, lifecycle/status/query | `test_document_upload_*.py`, `test_document_query_api.py` |
| Ingestion, task settings/publication/retry | `test_document_ingestion_*.py`, `test_celery_app.py` |
| Supervisor, Compose, TLS, security headers | `test_production_start.py`, `test_compose.py`, `test_database_tls_config.py`, `test_security_headers.py` |
| Retrieval metric calculations | `eval/test_metrics.py`; pure metric tests, not a performance run |
| Frontend | Node tests in `frontend/tests/` for auth/session/API helpers and source-level UI/security assertions |

Models, transport and storage are mocked in relevant focused tests. Frontend tests include static source checks; they are not a real-browser end-to-end suite. Supervisor tests use fake child processes. No ordinary test run proves live CloudAMQP/B2/Groq behavior. Test counts are intentionally not frozen into this document.

## Routine validation

Activate the repository's Python environment and run from the root:

```powershell
python -m pytest -q
python -m compileall -q main.py backend persistence eval scripts tests
python -m compileall -q start.py production_start.py
python -m pip check
python -c "import main, production_start, backend.app.celery_app; print('Imports OK')"
git diff --check
docker compose config --quiet
Push-Location frontend
npm test
npm run build
Pop-Location
```

Imports should use a valid local/test configuration; MySQL import requires its CA settings. They do not create tables or load embedding/reranker models, but database configuration initialization can materialize the configured CA file. Use `DATABASE_URL=sqlite:///:memory:` and clear deployment CA settings for an isolated import-only check when appropriate. `docker compose config --quiet` requires the referenced local `.env` file and validates the Compose model without printing resolved secrets or starting services.

`pip check` concerns the selected Python environment, not the contents of a built Docker image. Do not alter dependency pins merely to hide unrelated environment drift. The Docker build compiles the frontend; `npm run build` verifies it separately without a full image build.

## Guarded MySQL verification

The `mysql_integration` tests are intentionally destructive and skip unless `MYSQL_TEST_DATABASE_URL` names a MySQL database containing `test`, `testing`, or `disposable`. Never select production or any database with useful data. Use a dedicated database and private environment injection; do not paste a credential-bearing URL into committed commands.

```powershell
# Configure MYSQL_TEST_DATABASE_URL privately for a disposable database.
# Set MYSQL_TEST_CA_CERT to a local CA path when TLS is required.
python -m pytest -q -m mysql_integration
```

The integration file `tests/test_mysql_auth_integration.py`:

- Confirms auth tables use InnoDB, overlaps independent refresh-rotation connections, and verifies one successor rather than a branched token chain.
- Verifies rollback of staged refresh changes.
- Builds a representative pre-auth schema, runs migration 002 twice, checks its schema/delete rules/placeholder cleanup, then exercises registration and login against it.

The rotation invariant is one transaction containing locking read, validation, user lookup, successor insert, predecessor revocation/link and commit. The second transaction must observe the committed revocation after waiting. Browser locks reduce races but do not replace this invariant. Skipped integration tests are not evidence of InnoDB verification, and migration-002 rehearsal is not coverage of all migrations or a production migration run.

## Operational checks for a release

These are checks to perform in a controlled deployment, not claims that this documentation pass performed them:

1. Review the release diff and secret scan. Keep real `.env`, provider URLs/IDs, CA material and credentials out of git/images.
2. Back up the database and source-store metadata, establish restore/rollback steps, and rehearse pending migrations. Confirm schema through migration 005 and InnoDB where row locking is required.
3. Verify database TLS using the provider CA. `DB_ALLOW_LOCAL_TEST_CERT_HOSTNAME_MISMATCH` is only for loopback disposable databases and must not be set in production.
4. Confirm API/worker share database, broker queue, private object configuration and embedding model/input format. Confirm Secure cookies and exact allowed production origin.
5. Verify bucket privacy and required object-prefix read/write/delete permissions. Retention/backup settings belong to the provider; the app does not enforce them.
6. Build/deploy through the existing HF release branch workflow, inspect hosting build/runtime status, and confirm both child processes stay running. A push alone is insufficient runtime evidence.
7. In a test account, exercise register/login/refresh/logout, upload/poll/query, citations/history, duplicate choices, and cross-user 404 behavior. Check worker logs and the source object through authorized operational tools without exposing credentials.
8. Exercise publication failure/recovery and duplicate delivery in a disposable environment. Confirm that an original upload UUID can recover queued work; do not treat polling as republication.

## Health and observability limits

`GET /health` returns process version, model/client-loaded flags and document-cache count. It does not test DB, broker, storage, Groq or worker progress. `GET /health/db` checks database connectivity using a separate operational bearer token; a user JWT cannot substitute. An absent token setting yields 503 and an incorrect token yields 401.

Celery has no result backend. MySQL document status is the application-facing record, but queued does not imply a broker message exists and processing does not prove a worker is alive. Logs expose some failure types and document IDs; they are not a complete distributed tracing or audit system. There is no built-in queue-age alert, metrics dashboard, dead-letter administration or automated recovery service.

## Remaining operational limits

- Separate source PUT, metadata commit and message publication leave failure windows. There is no transactional outbox or queued-row reconciler.
- Late acknowledgement supports worker-loss redelivery, but task execution is not exactly once. Concurrent deliveries may repeat expensive work.
- The HF supervisor couples API and worker lifetime and has a ten-second shutdown grace period. Concurrency is currently 1; no independent production autoscaling is implemented.
- Source PDFs remain in private storage. There is no deletion/viewing endpoint or automatic orphan collector; database and object backups must be considered separately.
- FAISS is process-local. Cache size bounds entries rather than total RAM. Compatible persisted vectors support restart recovery but incompatible model changes need an operational plan.
- Query history is best effort after response. A successful answer can be missing from history.
- Content checks are user-scoped but do not serialize concurrent new UUIDs. Browser upload-recovery state is not persisted across reloads.
- Ordinary user JWT validation is stateless. Logout does not revoke access JWTs; account-state changes are not immediately reflected in every request.
- No app rate limiting, MFA, email verification, password-reset flow, OCR, formal security certification or current load benchmark is provided.

The detailed [failure table](failure_modes.md) and [security boundaries](security.md) describe practical implications. The [retrieval evaluation](retrieval_evaluation.md) records historical retrieval quality and local timing, not end-to-end production performance.
