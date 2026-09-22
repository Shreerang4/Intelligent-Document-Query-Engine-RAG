# Repository and deployment map

This is the source repository for the full application, not a generated deployment-only folder. [README.md](README.md) is the entry point; [FILE_TREE.txt](FILE_TREE.txt) is a curated directory overview, not an exhaustive generated file list.

| Path | Role |
| --- | --- |
| `main.py` | FastAPI application, shared query pipeline, process-local FAISS/cache, frontend serving |
| `backend/app/auth/` | Password/JWT/refresh authentication and HTTP boundary |
| `backend/app/documents/`, `schemas/`, `services/` | Async upload/status/query contracts and ingestion orchestration |
| `backend/app/celery_app.py`, `celery_config.py`, `tasks/` | RabbitMQ/Celery task delivery |
| `backend/app/rag/`, `storage/` | Shared PDF/embedding logic and private local/S3 storage |
| `persistence/`, `migrations/mysql/`, `scripts/init_db.py` | Relational state/artifacts, upgrades and explicit fresh-schema initialization |
| `frontend/` | React/Vite source, locked npm dependencies and Node tests |
| `start.py`, `production_start.py`, `Dockerfile`, `compose.yaml` | API startup, HF supervision, image build and separate local services |
| `tests/`, `eval/` | Backend tests, fixed retrieval labels, metric tests and evaluation tools |
| `docs/` | Architecture, failure, worker, schema, security, deployment and evaluation guides |

The Dockerfile builds the React assets and copies application source into a Python runtime image. `.dockerignore` excludes runtime upload/object-storage directories, local environment files except example templates, `certs/`, caches, node_modules and other development artifacts. Review custom storage paths and any new sensitive file before building; ignore rules are not a secret scanner.

GitHub `main` and the existing HF `main` have separate histories. The Space tree omits the three benchmark PDFs; application changes are applied on top of its current branch before pushing. The canonical procedure is in [docs/deployment.md](docs/deployment.md). Neither this manifest nor the presence of a staging branch certifies a live deployment or rollback target.

Generated `frontend/dist`, model/runtime caches and `eval/results` reports are not source deliverables. Only obvious configuration templates belong in git; `.env`, real keys, connection strings and provider account details do not.
