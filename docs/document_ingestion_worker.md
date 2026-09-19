# RabbitMQ and Celery ingestion worker

This phase adds task delivery around the queue-independent
`ingest_document(document_id)` service. No HTTP route publishes tasks yet, so
the existing synchronous browser flow is unchanged.

## Message contract

The registered task is `documents.ingest`. Its JSON body has one positional
argument, the opaque document UUID, and an empty keyword-argument object:

```text
args: [document_id]
kwargs: {}
```

User IDs, object keys, filenames, PDF bytes, chunks, embeddings, and model
configuration must never be added to the message. The worker resolves all
authoritative state through MySQL and private object storage by calling
`ingest_document(document_id)`.

Celery uses JSON serialization exclusively, late acknowledgement, worker-loss
rejection/redelivery, a prefetch multiplier of one, and no result backend.
Document status in MySQL is authoritative.

## Retry lifecycle

`DOCUMENT_INGESTION_MAX_RETRIES=3` means three retries after the initial
attempt, for at most four attempts. A retryable failure with budget remaining
conditionally moves `processing` back to `queued` before the countdown. The
update never overwrites `ready` or `failed`. If that best-effort state update
fails, the retry is still scheduled and only the exception type is logged.

Retries use full-jitter exponential backoff capped by
`DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS`. When the retry count is already
at the configured maximum, the application ingestion service records a fixed,
sanitized terminal failure. That write is conditional and cannot overwrite a
concurrently completed `ready` document. If the terminal database update
itself fails, the Celery task fails visibly.

Permanent PDF/content failures are recorded by the existing ingestion service
and are not retried. Duplicate or redelivered tasks rely on the existing
idempotent ingestion lifecycle and atomic artifact persistence.

## Local Compose

Copy `.env.example` to the ignored `.env` file and provide the existing database,
authentication, Groq, and storage settings. Compose expects `DATABASE_URL` to
identify a database reachable from both application containers; production-like
use should point both processes at the same MySQL database.

```powershell
docker compose up --build rabbitmq api worker
```

Compose builds one application image and starts it with separate commands:

```text
api:    python start.py
worker: celery -A backend.app.celery_app worker --loglevel=INFO --concurrency=1
```

When `OBJECT_STORAGE_BACKEND=local`, both services use
`/var/lib/idqe/object-storage` and mount the same `document-objects` volume.
For S3-compatible storage, both processes instead receive the same private
bucket/endpoint/credential-provider configuration through environment values.
No public object URL or ACL is used.

The Compose broker credentials are development defaults only. Override them for
shared environments and URL-encode special characters in the broker URL. A
host-run worker normally uses `localhost`; containers use the `rabbitmq` service
hostname. RabbitMQ, the API, and the worker can later be deployed separately by
changing only environment settings.

## Manual worker startup

With RabbitMQ reachable through `CELERY_BROKER_URL`:

```powershell
.venv\Scripts\celery.exe -A backend.app.celery_app worker --loglevel=INFO --concurrency=1
```

Keep concurrency at one initially. Each worker process may load its own E5
model, and PDF ingestion is CPU- and memory-intensive. There is no result
backend and no Flower deployment in this phase.

## Publication boundary

`POST /documents/upload` calls `enqueue_document_ingestion(document_id)` after
the private object and queued row are durable. The helper publishes only the UUID,
uses RabbitMQ publisher confirmations, and propagates normal publication errors.
It does not create or change the document row.

If publication raises, the endpoint returns a structured 503 containing the
document ID but leaves the row `queued` and retains the PDF. Confirmation
failures can be ambiguous, so this permits an already-accepted task to run and
also permits the same upload request UUID to republish safely. The future
frontend should always supply an `upload_request_id`, even though the field is
currently optional.

The remaining crash window is a process exit after the queued-row commit and
before publication. A client retry with the same request UUID repairs it; a
transactional outbox and queued-row reconciler are intentionally absent.
