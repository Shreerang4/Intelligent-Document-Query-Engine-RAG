# Failure modes and recovery

This table describes the current async upload, ingestion, and ready-document query paths. It is based on `backend/app/services/document_upload.py`, `backend/app/tasks/ingest_document.py`, `persistence/document_ingestion.py`, and `main.py`. Recovery here is application behavior, not a guarantee of provider availability or exactly-once execution.

| Failure / event | Visible behavior | Persisted state | Recovery and duplicate risk | Current limit |
| --- | --- | --- | --- | --- |
| Invalid upload, size, signature, or request UUID | 400; form/schema validation may be 422 | No new object or row | Correct the input | Signature validation is not full PDF parsing |
| Object-storage PUT raises | 503, safe message | No new document row; remote PUT outcome may be ambiguous | Retry the upload attempt | A completed remote PUT followed by an error can leave an orphan; no automatic orphan sweeper |
| Metadata write fails after object PUT | 503; conflict races can instead recover the winning request | DB transaction normally rolls back; new object deletion is attempted | Retry with the same UUID; the owned unique constraint arbitrates request races | Cleanup is best effort; an ambiguous DB commit outcome is not atomically coordinated with object deletion |
| Concurrent same upload UUID | Winner recovered or 409 for different bytes | At most one owned row for that non-null UUID | Losing object is deleted best effort; publication may repeat | Same request idempotency does not make task delivery exactly once |
| Status read fails after metadata commit | 503 with authoritative document ID/status when known | Row and PDF retained | Retry the same upload UUID | This attempt may not yet have published a task |
| Broker publication/confirmation raises | 503 with `message`, `document_id`, `status: queued` | Queued row and PDF retained | Same UUID retry can republish; original message may already exist | Confirmation failure can be ambiguous; duplicate tasks must be tolerated |
| Hard API crash between row commit and publish | Lost/error HTTP response; later status can remain queued | Queued row and source PDF | Same upload UUID retry republishes if still queued | No transactional outbox or automatic queued-row reconciler; polling alone does not publish |
| Lost HTTP response after successful publication | Client cannot know the outcome from that response | Existing row/PDF; worker may have advanced | Same UUID recovers current state; republishes only if queued | Browser retains the UUID only during the current page's upload attempt |
| Worker process lost before acknowledgement | UI may remain processing until broker redelivery | Last committed state retained | Late ack plus `reject_on_worker_lost` permits redelivery and repeated work | Broker availability/persistence and worker restart are external dependencies; no processing lease or heartbeat |
| Retryable object GET, model, embedding, unexpected processing, or DB error | Usually queued during retry; safe failed status when exhausted | Completed artifacts remain atomic; processing can remain if retry-state write fails | Three configured retries by default; full-jitter delay capped at 300 seconds | If retry publication or terminal state persistence fails, task failure is visible to operations; no automatic reconciliation guarantee |
| Invalid/encrypted/unreadable PDF, no meaningful text, invalid metadata, or stored-size mismatch | Status becomes failed with a bounded message | Source retained; no successful artifact transaction | New upload UUID for a corrected/fresh attempt | A delivery to failed is a no-op; no retry-failed HTTP endpoint |
| Duplicate Celery delivery | Usually no user-visible difference | Ready row is preserved by locked persistence | Ready document is a no-op; processing deliveries may recompute | Artifact idempotency does not prevent duplicate CPU work; an in-flight success may supersede failed |
| Same user's usable content uploaded with a new UUID | 409 `duplicate_document` and existing document projection | No new object, row, or task | Open existing, or explicitly upload again | Different concurrent UUIDs can pass the hash check before either commits |
| Intentional duplicate upload | Normal 202 path with `allow_duplicate=true` | New object and row if request UUID is new | Subsequent retries must reuse that new attempt's UUID | The flag bypasses content checking only; reusing an old committed UUID recovers the old row |
| Failed content match | Fresh attempt can proceed | Existing failed row/source unchanged | New UUID permits normal upload | Original UUID still recovers failed state |
| Query of queued/processing/failed document | 409 | No lifecycle change | Poll until ready; for failed, inspect status and start a new upload as appropriate | No query-triggered ingestion or source repair |
| Missing or another user's private resource | 404 after JWT validation | No change | Use an owned identifier | Worker IDs are trusted internal input, not user authorization |
| Incompatible/missing persisted query vectors | 503 | Existing DB/source unchanged | Restore compatible configuration/artifacts operationally | Async query route does not regenerate vectors |
| No retrieved context | Successful query response with `status: no_context` | Best-effort query history | Ask a question supported by available text | Does not prove the original PDF lacks the information |
| Query model/Groq failure | Ready-document endpoint returns safe 503 for a failed answer | Document stays ready | Client may retry | No query idempotency; generation can repeat and a repeated successful query can add history |
| Claim verification failure | Answer can still return with failed/partial verification evidence | Document unchanged | Review source evidence manually | Verification is another model call, not a correctness guarantee |
| Post-response history DB failure | Answer may already be successful | Query/citations may be missing | Failure is logged by exception type | No durable history-write queue or replay |
| API/container restart | RAM cache and in-flight work are lost | DB and external source objects remain subject to provider durability | Rebuild FAISS from matching stored vectors; broker can redeliver unacknowledged work | Local filesystem sources require a persistent shared mount |

## The publication gap

The upload order is object PUT, document-row commit, then confirmed broker publication. There is no distributed transaction. Publisher confirmations improve observability of publication but cannot prove atomicity with the preceding MySQL commit. An error is deliberately not converted into a failed document because an accepted task may still run.

The recovery key is `(user_id, upload_request_id)`. Retry with the original bytes and UUID; a new UUID can produce a content-duplicate prompt without repairing the original publication. Opening a queued duplicate only resumes polling. The current UI does not durably retain upload UUIDs after a browser reload, and there is no outbox, queue inspector endpoint, or reconciler to close this gap automatically.

## At-least-once boundaries

Celery is configured with JSON serialization, late acknowledgement, worker-loss rejection, publisher confirmations and no result backend. Treat deliveries as at least once and processing as repeatable, subject to broker/worker availability. Only classified retryable service exceptions enter the configured retry loop. Arbitrary unhandled exceptions are not promised automatic retry; a failed task and document status can diverge when state recording itself fails.

The row locks at start and artifact commit are short transactions. They preserve a ready winner and avoid partial committed chunk sets, but do not serialize the entire parsing/embedding interval. Retry counters apply to Celery retries; they do not cap all broker redeliveries after repeated worker loss.

## Operational interpretation

`GET /health` reports API process/model-cache information; it does not exercise RabbitMQ, storage, the worker or Groq. `GET /health/db` tests only the database and needs an operational token. Long-lived queued/processing documents need operational investigation; there is no built-in age threshold, alert, dead-letter workflow, or automatic repair job. See [production readiness](production_readiness.md) for validation scope.
