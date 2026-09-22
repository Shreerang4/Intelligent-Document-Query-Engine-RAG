# RabbitMQ and Celery ingestion worker

The async upload endpoint publishes to the Celery task `documents.ingest`. The task invokes `ingest_document(document_id)`, whose parsing/persistence logic is independent of the transport and shared with the application's PDF processing components. The React upload flow already uses this queue and polls lifecycle status.

## Message and queue contract

```text
task: documents.ingest
queue: document_ingestion (DOCUMENT_INGESTION_QUEUE)
args: [document_id]
kwargs: {}
```

The helper normalizes the UUID before sending. PDF bytes, user IDs, object keys, filenames, chunks, vectors and model configuration are not task arguments. The worker resolves the authoritative document row and private source metadata from MySQL. Queue publication is a trusted internal capability, not a user-facing authorization endpoint.

`backend/app/celery_app.py` explicitly configures:

| Setting | Value / implication |
| --- | --- |
| `accept_content`, task/result serializers | JSON only |
| `task_acks_late` | `True`; acknowledgement follows task execution |
| `task_reject_on_worker_lost` | `True`; worker-process loss can lead to broker redelivery |
| `worker_prefetch_multiplier` | `1`; limits prefetched work per concurrency slot |
| `task_ignore_result` | `True` |
| `task_store_errors_even_if_ignored` | `False` |
| `result_backend` | `None`; lifecycle is read from MySQL |
| `broker_connection_retry_on_startup` | `True` |
| `broker_transport_options` | `confirm_publish=True` |
| task routing/default queue | `DOCUMENT_INGESTION_QUEUE`, default `document_ingestion` |

There is no application-configured task time limit, dead-letter workflow, result store, Flower deployment or automatic queued-document reconciler. At-least-once delivery must be tolerated; these settings do not promise exactly-once work or recovery from every arbitrary exception.

## Worker lifecycle and artifact transaction

1. Lock the document row. Missing documents are ignored by the delivery wrapper; ready/failed documents return without work.
2. Validate source metadata and commit `processing`. A processing document can be started again after redelivery.
3. GET the private object and check its byte length against metadata. The worker does not recompute `source_hash` as a cryptographic integrity check.
4. Parse with PyMuPDF, clean text, split per page into 500-character chunks with 50-character overlap, and embed with the configured model. E5 document input uses `passage: ` prefixes.
5. Lock the document again. If already ready, preserve that winner. Otherwise replace chunks, store each vector as float32 bytes with dimension/dtype, and commit `ready` plus artifacts together.

The source PDF remains in storage after success or failure. No committed partial artifact set is exposed by the final transaction. The start lock is not held while embedding: duplicate deliveries can repeat computation. A previously started success may finish after a different invocation marked failed; final persistence only short-circuits an already-ready row. See [architecture](architecture.md) for the state diagram.

## Failure classification and retry policy

| Failure | Service/task behavior |
| --- | --- |
| Invalid/missing source metadata; source-size mismatch | Record a bounded permanent failure |
| `PdfContentError`: oversize, invalid/encrypted/unreadable PDF, no usable text | Record failed; do not retry the content error |
| Storage GET failure (including missing object), unexpected parser error, model load, embedding or artifact DB failure | Raise `RetryableDocumentIngestionError` |
| Failure to record a permanent failure | Becomes retryable, because DB state is not authoritative yet |
| Missing document | Log that it is gone and return |
| Unhandled exception outside classified paths | Task fails; there is no catch-all autoretry |

With retries remaining, the wrapper conditionally moves processing back to queued before `self.retry`. It never changes ready/failed to queued. If that state update fails, it logs only the exception type and still schedules the retry. If the document disappeared or reached a terminal state, retry is suppressed.

`DOCUMENT_INGESTION_MAX_RETRIES=3` allows three Celery retries after the initial attempt. For retry number `r` (starting at zero), countdown is an integer selected uniformly from:

```text
0 .. min(DOCUMENT_INGESTION_RETRY_BACKOFF_MAX_SECONDS,
         DOCUMENT_INGESTION_RETRY_BACKOFF_SECONDS * 2**r)
```

Defaults are 5 and 300 seconds. This is full jitter, so a retry can be immediate. Retry counts do not cap repeated broker redeliveries caused by worker loss. When retry budget is exhausted, the service records a fixed safe failure message while preserving a ready winner. If that terminal DB write fails, the task fails visibly; no result backend or reconciler repairs its document state automatically.

## Publication boundary and recovery

`POST /documents/upload` writes the object and commits the queued row before calling `enqueue_document_ingestion`. Publisher confirmation cannot make these operations atomic. An ordinary publish failure returns 503 with the authoritative document ID, leaves the row queued, and retains the source because the broker may already have accepted the message.

Clients should supply and reuse `upload_request_id` for the same logical upload. Recovery occurs before content-duplicate checking and republishes only when current state is queued. Processing/ready/failed recovery does not publish. The React client uses a UUID retained in memory during the selected-file attempt, including **Upload again** after an uncommitted duplicate response; it does not save the UUID across reloads.

A crash after row commit and before publication can leave a queued document without a message. Retrying the same upload UUID can repair it. Status polling and opening a queued duplicate do not republish. No outbox or periodic reconciler closes this window automatically. [Failure modes](failure_modes.md) covers compensation and ambiguity in detail.

## Production and local startup

HF uses `production_start.py` to run FastAPI and a concurrency-1 Celery worker together. Both use the external CloudAMQP broker, shared Aiven database, and private B2 storage. It is a deployment compromise, not a combined API/task implementation.

Local Compose starts `rabbitmq`, `api`, and `worker` services from the configured environment:

```powershell
docker compose up --build rabbitmq api worker
```

The API command is `python start.py`. The worker command is:

```text
celery -A backend.app.celery_app worker --loglevel=INFO --concurrency=1
```

Compose requires a shared reachable database; the example SQLite file is not mounted between containers. For local source storage, both services mount `document-objects` at `/var/lib/idqe/object-storage`. For S3, both receive the same bucket/endpoint/credential configuration. The RabbitMQ health dependency gates initial startup, and API port mapping remains `7860:7860`.

A host-run worker uses the same command with a broker address reachable from the host. Configure matching database/storage/model settings before launching. Each worker process may load its own embedding model; concurrency 1 is the current deployment configuration, not a measured scaling limit. [Deployment and configuration](deployment.md) provides complete setup commands.
