# Deployment, local setup, and configuration

## Current production topology

| Service | Deployment role |
| --- | --- |
| Hugging Face Docker Space | One container with FastAPI, the built React frontend, and one Celery worker |
| CloudAMQP | External RabbitMQ broker used by both publisher and worker |
| Aiven MySQL / InnoDB | Users/sessions, document lifecycle, chunks/vectors, queries/citations |
| Backblaze B2 | Private S3-compatible source-PDF storage |
| Groq | Answer generation and claim verification |

These are the project's current provider assignments. The code verifies the integration interfaces, not live account configuration. No account identifiers, private endpoints, credentials or bucket names are published here.

The Dockerfile builds the frontend with Node 20, then uses Python 3.11, CPU PyTorch and the Python requirements in the runtime image. It copies the frontend build into `frontend/dist`; FastAPI serves that build from the same origin as the API. `EXPOSE 7860` and README `app_port: 7860` agree with the default `PORT`.

The `.dockerignore` excludes `uploads/` (including the default `uploads/object-storage` root), nested directories named `uploads` or `object-storage`, local environment files except `.env.example` templates, and `certs/`. Runtime PDFs and these private configuration files are excluded from `COPY . ./`. If you configure a differently named storage root inside the repository, keep it outside the build context or explicitly exclude it before building. Git ignore rules alone do not protect an image build.

`CMD ["python", "production_start.py"]` launches:

```text
python start.py
python -m celery -A backend.app.celery_app worker --loglevel=INFO --concurrency=1
```

Both children inherit the container environment. `start.py` binds `0.0.0.0` at `PORT` (default 7860). The supervisor forwards SIGTERM/SIGINT, waits up to 10 seconds, then kills remaining child processes. If either child exits unexpectedly, even with code zero, it terminates its sibling and returns nonzero. It does not restart children itself, wait for their readiness, or manage a Celery result backend. Container restart is a hosting responsibility. A forced shutdown during ingestion can require broker redelivery.

Co-residence is a low-cost/demo choice. API and worker have separate roles but share CPU, memory and container lifetime. For separate production deployments, override the API command with `python start.py`, run the worker command in its own deployment, supply matching database/broker/storage/model settings, and provision resources independently. The existing ID-only task contract is suitable for that split; independent production scaling is not currently deployed. More workers require capacity validation, not just a concurrency flag change.

## Configuration reference

Runtime settings are read from environment variables; Python modules also load `.env` via python-dotenv. Values are often cached or initialized at import, so restart affected processes after changing them. `.env.example` contains local examples and empty secret fields. Never put secrets in `VITE_*` variables: those become browser-build inputs.

### Identity, database and HTTP

| Variable | Default / requirement | Meaning |
| --- | --- | --- |
| `ACCESS_JWT_SECRET` | Required for auth; no default; at least 32 UTF-8 bytes | **Secret.** HS256 signing key; independent of `API_TOKEN`. |
| `ACCESS_TOKEN_TTL_SECONDS` | `600` | Positive access-token lifetime in seconds. |
| `REFRESH_COOKIE_SECURE` | `true` | Secure cookie flag. Use `false` only for plain-HTTP local development. |
| `AUTH_ALLOWED_ORIGINS` | Empty | Additional exact browser origins. Same request origin and built-in localhost/127.0.0.1 ports 3000/5173 are accepted by auth checks. No wildcard support. Also supplies the CORS allowlist. |
| `API_TOKEN` | None; needed only for `/health/db` | **Secret.** Operational diagnostic token; cannot authenticate user routes. |
| `GROQ_API_KEY` | Required for generation/verification | **Secret.** Used by the Groq SDK. Ingestion itself does not call Groq. |
| `DATABASE_URL` | `sqlite:///./rag_persistence.db` | **Secret when credential-bearing.** Production/shared local topology requires reachable MySQL, normally `mysql+pymysql`. |
| `DB_CA_CERT` | None | Local path to MySQL CA certificate; every MySQL URL requires CA configuration. Mount it into containers if using a file path. |
| `DB_CA_CERT_B64` | None | Deployment-managed base64 CA material; takes precedence over the path and is decoded to `/tmp/aiven-ca.pem` with mode 0600. No certificate contents are logged by this bootstrap. |
| `DB_ALLOW_LOCAL_TEST_CERT_HOSTNAME_MISMATCH` | `false` | Restricted hostname-check exception for loopback disposable/test MySQL databases. CA/signature verification remains. Rejected for production hosts/database names. |
| `PORT` | `7860` | API bind port. Keep 7860 for HF and the current Compose port mapping. |

The refresh cookie name/path and absolute seven-day lifetime are code constants, not environment settings. Database tables are **not** created on application startup. Use `scripts/init_db.py` for a fresh schema; use [migrations](persistence_schema.md) for existing databases. SQLite is useful for host-local smoke tests but does not establish InnoDB locking guarantees.

### Source object storage

| Variable | Default / requirement | Meaning |
| --- | --- | --- |
| `OBJECT_STORAGE_BACKEND` | `local` | Active async upload/worker backend: `local` or `s3`. Production B2 uses `s3`. |
| `OBJECT_STORAGE_LOCAL_ROOT` | `./uploads/object-storage` | Local private root, resolved to an absolute path. API/worker must see the same contents. Compose overrides to `/var/lib/idqe/object-storage`. |
| `OBJECT_STORAGE_S3_ENDPOINT_URL` | boto3 service default | Custom S3-compatible endpoint, required for a non-AWS provider such as B2. Supply actual provider values privately. |
| `OBJECT_STORAGE_S3_BUCKET` | Required for `s3` | Private bucket name; public access must be disabled by provider policy. |
| `OBJECT_STORAGE_S3_REGION` | boto3/provider default | Bucket region if required by the provider. |
| `OBJECT_STORAGE_S3_ACCESS_KEY_ID` | boto3 credential chain | **Credential.** Optional explicit key identifier; must accompany its secret. |
| `OBJECT_STORAGE_S3_SECRET_ACCESS_KEY` | boto3 credential chain | **Secret.** Optional explicit secret; must accompany the key identifier. |
| `OBJECT_STORAGE_S3_SESSION_TOKEN` | None | **Secret.** Optional temporary credential token; requires explicit key/secret. |
| `OBJECT_STORAGE_S3_ADDRESSING_STYLE` | `auto` | `auto`, `path`, or `virtual`. |

For B2, configure `s3`, endpoint, bucket, region and application credentials in the hosting environment. The implementation calls S3 PUT/GET/DELETE without setting public ACLs. It does not validate bucket policy or implement client-side encryption, object retention rules, signed viewing URLs, or a deletion API. Without explicit credentials, boto3 uses its standard provider chain. Do not copy a local `.env` into a production image.

### RabbitMQ and Celery

| Variable | Default | Meaning |
| --- | --- | --- |
| `CELERY_BROKER_URL` | Local RabbitMQ guest URL on `localhost:5672` | **Secret in production.** Same broker URL for API publisher and worker; configure the external CloudAMQP URL privately. |
| `DOCUMENT_INGESTION_QUEUE` | `document_ingestion` | Queue for task `documents.ingest`; must match across processes. |
| `DOCUMENT_INGESTION_MAX_RETRIES` | `3` | Retries after the initial attempt; nonnegative. Does not cap all worker-loss redeliveries. |
| `DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS` | `5` | Initial full-jitter upper bound in seconds; positive. |
| `DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS` | `300` | Maximum upper bound; at least the initial value. |
| `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` | `idqe`, `idqe-local` | Local Compose broker defaults only; never production credentials. |
| `COMPOSE_CELERY_BROKER_URL` | Local `idqe` broker URL at service host `rabbitmq:5672` | Compose interpolation variable overriding `CELERY_BROKER_URL` for both app services. Match the local broker credentials. |

URL-encode reserved characters in credentials. A host-run process reaches local RabbitMQ via `localhost`; a Compose process uses `rabbitmq`. Compose does not provide a broker management UI port. Its RabbitMQ health dependency gates initial API/worker startup, not ongoing readiness. The application has no Celery result backend; inspect document status rather than Celery results.

### Retrieval and cache

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_PDF_BYTES` | `15728640` (15 MiB) | PDF size cap. |
| `HTTP_TIMEOUT_SECONDS` | `30` | Legacy URL-download timeout. |
| `EMBEDDING_MODEL_NAME` | `intfloat/e5-small-v2` | Shared worker/query embedding configuration. MiniLM can be selected explicitly as `all-MiniLM-L6-v2`. |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | CrossEncoder model. |
| `LLM_MODEL_NAME` | `openai/gpt-oss-20b` | Groq generation/verification model. |
| `RETRIEVAL_MODE` | `faiss_reranker` | Default path; `e5` aliases it. Experimental `e5_bm25_reranker` selects hybrid retrieval. |
| `RETRIEVAL_K_INITIAL`, `RETRIEVAL_K_FINAL` | `20`, `8` | Default FAISS candidates and reranked context chunks. |
| `HYBRID_E5_K_INITIAL`, `HYBRID_BM25_K_INITIAL`, `HYBRID_K_FINAL` | `30`, `20`, `5` | Experimental hybrid branch's independent limits. |
| `MAX_CONCURRENT_QUESTIONS` | `4` | Semaphore bound per multi-question request; not a global request or Celery worker limit. |
| `DOCUMENT_CACHE_MAX_ITEMS` | `8` | Process-wide document-index entry bound, not per-user or byte-based. |
| `DOCUMENT_CACHE_TTL_SECONDS` | `3600` | Expiration measured from entry creation, checked on cache operations. Zero effectively prevents reuse as time advances. |
| `VITE_API_BASE_URL` | Empty / same origin | Frontend build setting. The local Vite proxy already maps API paths to port 8000. |

Chunk size 500, overlap 50, maximum questions 10, maximum extracted claims 5 and normal claim-verification final chunks 3 are code constants. `allow_duplicate`, `upload_request_id`, and legacy `request_id` are API fields, not environment variables. Model names/input formats must agree with stored vectors for async queries; changing models is not an automatic migration.

## Local Compose: complete application image

Requirements: Docker with Compose and a shared MySQL database reachable from both app containers. The Compose file provisions RabbitMQ, API, worker and named object/broker volumes; it does **not** provision MySQL or mount a shared SQLite database.

```powershell
Copy-Item .env.example .env
# Edit .env with local settings before starting services.
docker compose config --quiet
docker compose build
# Fresh database only; existing schemas need migrations instead.
docker compose run --rm --no-deps api python scripts/init_db.py
docker compose up rabbitmq api worker
```

In `.env`, set a shared MySQL `DATABASE_URL`, its CA setting, a generated `ACCESS_JWT_SECRET`, a Groq key, and `REFRESH_COOKIE_SECURE=false` for plain HTTP. Keep `PORT=7860`. If MySQL runs on the Docker Desktop host, use a host reachable from containers rather than container-local `localhost`; its TLS certificate must match the chosen hostname. For TLS material, the base64 CA setting avoids needing an additional certificate bind mount.

Open `http://localhost:7860` for the built UI. Both app services explicitly override the image CMD, so the API service does not start an extra worker. They receive the same Compose broker URL and share `document-objects:/var/lib/idqe/object-storage`. Optional S3 configuration replaces local-object reads/writes but does not remove the declared volume. Normal Compose shutdown retains named volumes; deleting volumes removes their data.

## Host Python + Vite development

From the repo root in PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Configure secrets locally. Use REFRESH_COOKIE_SECURE=false for HTTP.
python scripts/init_db.py
```

For a simple host smoke setup, SQLite may be shared by API/worker launched from the same repository directory using the same absolute SQLite file path. Prefer MySQL for transaction/concurrency validation. Use an absolute `OBJECT_STORAGE_LOCAL_ROOT` in both processes to avoid different working directories selecting different stores.

Start the Compose broker only, then run the API in the host environment:

```powershell
docker compose up -d rabbitmq
# Host client uses localhost and the Compose broker's development credentials.
$env:CELERY_BROKER_URL='amqp://idqe:idqe-local@localhost:5672//'
$env:PORT='8000'
python start.py
```

In a second activated terminal, use the same `.env`/database/storage configuration. A Linux/WSL environment or the Compose worker is the preferred way to exercise the production prefork worker:

```powershell
$env:CELERY_BROKER_URL='amqp://idqe:idqe-local@localhost:5672//'
python -m celery -A backend.app.celery_app worker --loglevel=INFO --concurrency=1
```

For a Windows host smoke run, `--pool=solo` can be added to avoid prefork; this is a development variant, not the deployed command or a test of worker-loss recovery. Do not mix a host-local SQLite file with a container worker that cannot access it.

In a third terminal:

```powershell
Push-Location frontend
Copy-Item .env.example .env
npm ci
npm run dev
Pop-Location
```

Leave `VITE_API_BASE_URL` empty for the Vite same-origin proxy. `/auth`, `/documents`, `/hackrx`, `/history`, and `/health` target `http://localhost:8000`. This is why the host API command above uses port 8000. Frontend builds served by FastAPI/HF use port 7860 by default. To serve a local production frontend, run `npm run build` before starting FastAPI.

## Existing HF release workflow

Deployment is a manual push to the already configured `hf` remote's `main` branch. There is no GitHub-to-Space auto-sync workflow in the tracked repository. GitHub `main` and Space `main` have separate histories; the Space tree intentionally omits the three benchmark PDF binaries. A direct `main:main` push to HF is not the established history-preserving workflow.

After a reviewed GitHub commit, fetch the configured Space branch and apply the intended commit in an isolated worktree. The following is a template; replace the commit placeholder and choose an unused worktree/branch name:

```powershell
git fetch hf main
git worktree add -b hf-release .hf-release-worktree hf/main
git -C .hf-release-worktree cherry-pick <reviewed-github-commit>
git -C .hf-release-worktree diff --check HEAD~1 HEAD
git -C .hf-release-worktree show --stat --oneline HEAD
# Review the complete deployment diff, then push only the release branch.
git push hf hf-release:main
git worktree remove .hf-release-worktree
git branch -d hf-release
```

Keep README Docker metadata and the existing Space settings intact. Do not force-push to join the unrelated histories. A successful git push starts the hosting build; it is not evidence that the build or runtime is healthy. Inspect the host's build/runtime state and verify the upload/poll/query flow separately.

Provide production secret categories through the hosting secret store: JWT signing key, Groq key, DB URL/CA material, broker URL, S3 application credentials, and the optional operational diagnostic token. Configure public origins, bucket/endpoint/region and backend mode privately as appropriate. Never commit real provider URLs, account IDs, certificate files or credentials in examples.

For rollback, retain a reviewed application image/commit and compatible database backup. An older Space branch or staging remote alone is not proof of a valid rollback target; migrations and persisted embedding compatibility must also be considered. See [production readiness](production_readiness.md) and [failure modes](failure_modes.md).
