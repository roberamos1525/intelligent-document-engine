# Operations runbook

## Running

| Mode | Command |
| --- | --- |
| Development server | `idengine serve --host 127.0.0.1 --port 8000` |
| Production-style | `uvicorn idengine.api.app:build_app_from_env --factory --host 0.0.0.0 --port 8000 --no-access-log` |
| Container | `docker compose up --build` (not verified locally; Docker was unavailable) |

Run **one API process per data directory**. The job queue and SQLite tolerate several worker
threads, but in-memory indexes are per process: a second process would not see the first one's
index updates until restart. Scale up with `jobs.workers` and Uvicorn's thread pool rather than
multiple processes.

Configuration is validated at start-up; a bad value aborts with `configuration_error` and the
offending field. Print the effective configuration (secrets masked) with `idengine config`.

## Health and monitoring

- `GET /health` (no auth): database round-trip latency, index status (vectors available, chunk
  count), worker threads running, LLM adapter reachability. `200` = ok, `503` = degraded.
- `GET /metrics`: Prometheus text format. Metrics (all prefixed `idengine_`):

| Metric | Type | Labels |
| --- | --- | --- |
| `http_requests_total` | counter | method, route (template), status |
| `http_request_seconds` | histogram | method, route |
| `documents_ingested_total` | counter | type, outcome (`ok`, `failed`, `duplicate`) |
| `chunks_indexed_total` | counter | |
| `ingest_seconds` | histogram | |
| `search_seconds` | histogram | mode |
| `ask_seconds` | histogram | |
| `llm_calls_total` | counter | provider, outcome |
| `jobs_total` | counter | type, outcome (`submitted`, `succeeded`, `retried`, `failed`) |
| `job_seconds` | histogram | type |
| `documents`, `chunks`, `job_queue_depth` | gauge | refreshed on `/metrics` and `/v1/stats` |

Suggested alerts: `/health` non-200; rising `jobs_total{outcome="failed"}`;
`job_queue_depth` growing for more than 15 minutes; p95 of `http_request_seconds` for `/v1/ask`
above your SLO; `documents_ingested_total{outcome="failed"}` spikes.

- **Logs** are JSON lines on stderr (`logging.json_format=false` for plain text). Each request log
  has `request_id`, route, status, duration and the caller's key fingerprint (`key_id`); job logs
  carry `job_id`. Use the `request_id` from an error response to find the matching log lines.

## Background jobs

- Inspect: `idengine jobs list --status failed` or `GET /v1/jobs?status=failed`.
- Retry: `POST /v1/jobs/{id}/retry` (admin). Cancel a queued job: `POST /v1/jobs/{id}/cancel`.
- Process the queue without the API (e.g. after a bulk `ingest --background`): `idengine jobs run`.
- Remove old finished jobs: `idengine jobs purge --older-than-days 30`.
- Retries use exponential backoff with jitter (`jobs.backoff_base_seconds`,
  `jobs.backoff_max_seconds`) up to `jobs.max_attempts`. Validation, not-found and conflict errors
  fail immediately.
- A job running longer than `jobs.lease_seconds` keeps running, but if the process dies its lease
  expires and the job is requeued at the next start. Size `lease_seconds` above your slowest job.

## Index maintenance

- Indexes are rebuilt automatically at start-up when missing, corrupt, built for a different
  embedding configuration, or inconsistent with the database.
- Manual rebuild: `idengine reindex` (refits LSA) or `POST /v1/admin/reindex` with
  `{"refit": true}`. Use `--no-refit` to rebuild without refitting.
- After changing `chunking.*`, reprocess documents: `POST /v1/documents/{id}/reprocess` per
  document (chunk boundaries are only recomputed on reprocessing).
- Switching `index.kind` or embedding settings needs no manual action; the next start rebuilds.
- Deleted vectors are kept as tombstones (HNSW still routes through them); the index compacts
  itself automatically when more than half of its rows are tombstones, and a reindex also compacts.

## Backups and restore

1. Stop the service (or pause writes).
2. Copy `engine.sqlite3` (or use `sqlite3 engine.sqlite3 ".backup backup.sqlite3"` for an online
   copy), `blobs/` and `models/`. `indexes/` is optional because it can be rebuilt.
3. Restore by putting the files back into an empty data directory and starting the service; the
   indexes rebuild if absent.

## Capacity notes

Indexes are fully in memory and each flush rewrites the index files. Memory is roughly
`chunks x dim x 4 bytes` for vectors (256-dim: about 1 KiB per chunk) plus the BM25 postings and,
for HNSW, the adjacency lists. Pure-Python HNSW inserts took about 2 ms each (64-dim, `ef_construction=100`) on the development
machine, so bulk loads of many thousands of chunks are noticeably slower than with native
libraries; `index.kind = "flat"` is the fastest option for small corpora.

## Troubleshooting

| Symptom | Likely cause / action |
| --- | --- |
| `configuration_error` at start-up | Fix the named setting; `idengine config` shows effective values |
| Server refuses to start: "no keys are configured" | Set `IDE_API_KEYS` (or `IDE_AUTH_ENABLED=false` locally) |
| `415 unsupported_file_type` | File content is not one of the supported formats, or the extension lies |
| `422 extraction_failed` "no extractable text" | Scanned PDF without a text layer (OCR not supported) |
| Hybrid search warning "vector index unavailable" | LSA configured but not fitted: run `idengine reindex` |
| `semantic` search returns `409 model_not_trained` | Same as above |
| `502 llm_error` | External LLM unreachable or returned an invalid response; check `IDE_LLM_BASE_URL`, or switch back to `extractive` |
| `PermissionError` / WinError 32 on Windows | Antivirus holding a file; the engine retries renames, re-run if it persists |
| Slow first start after upgrade | Indexes are being rebuilt from the database (logged as "rebuilding indexes") |
