# HTTP API reference

Base URL: `http://<host>:<port>`. All request and response bodies are JSON unless noted.
Interactive OpenAPI documentation is served at `/docs` and the schema at `/openapi.json`.

## Authentication and roles

Send an API key in `X-API-Key: <key>` or `Authorization: Bearer <key>`. Keys are configured with
`IDE_API_KEYS="key1:admin,key2:reader"` (a key without a role is `admin`).

| Role | Allowed |
| --- | --- |
| `reader` | GET endpoints, search, ask, summaries, classify text, stats, metrics |
| `writer` | reader + upload, update, delete, reprocess, classify document, cancel job |
| `admin` | writer + train classifier, reindex, retry job |

`GET /health` needs no key. `GET /metrics` needs a reader key unless `api.metrics_public=true`.
Setting `IDE_AUTH_ENABLED=false` disables authentication (every caller is admin); the server
refuses to start with authentication enabled and no keys configured.

Each key has a token bucket (`api.rate_limit_per_minute`, `api.rate_limit_burst`); exceeding it
returns `429` with a `Retry-After` header.

## Errors

Every error uses the same envelope and carries the request id (also returned in the
`X-Request-ID` response header; a valid incoming `X-Request-ID` is propagated):

```json
{"error": {"code": "unsupported_file_type",
           "message": "executable content rejected (Windows executable)",
           "details": {}, "request_id": "5f2c..."}}
```

| HTTP | `code` values |
| --- | --- |
| 401 | `unauthorized` |
| 403 | `forbidden` |
| 404 | `not_found` |
| 405 | `method_not_allowed` |
| 409 | `conflict`, `model_not_trained` |
| 413 | `file_too_large`, `request_too_large` |
| 415 | `unsupported_file_type` |
| 422 | `validation_error` (request schema or business rule), `extraction_failed` |
| 429 | `rate_limited` |
| 500 | `internal_error`, `storage_error`, `index_corrupted`, `configuration_error` |
| 502 | `llm_error` (external LLM adapter failed) |

Request validation errors list the offending fields in `details.errors`.

## Limits

| Limit | Setting | Default |
| --- | --- | --- |
| Request body | `api.max_request_bytes` | 25 MiB |
| Uploaded file | `ingest.max_file_bytes` | 20 MiB |
| PDF pages | `ingest.max_pdf_pages` | 500 |
| DOCX expansion | `ingest.max_docx_uncompressed_bytes` / `max_docx_compression_ratio` | 200 MiB / 150 |
| Query length | `search.max_query_chars` | 2000 |
| `top_k` | `search.max_top_k` | 100 |
| Inline text (summarize/classify/train) | fixed | 200,000 characters |
| Tags per document | fixed | 20 (letters, digits, `-_.:`, max 40 chars) |

## Documents

### `POST /v1/documents` (writer)

Multipart form: `file` (required), `tags` (comma separated), `category` (sets a manual category).
Query: `wait=true` to process synchronously.

| Status | Meaning |
| --- | --- |
| 202 | Registered; an `ingest` job was queued (`job` in the body) |
| 201 | Processed synchronously (`wait=true`) |
| 200 | Identical bytes already stored; `duplicate: true`, existing document returned |

```json
{"document": {"id": "doc_8e1f...", "filename": "invoice.txt", "doc_type": "txt",
              "status": "ready", "title": "NORTHWIND SUPPLIES INVOICE", "authors": [],
              "document_date": "2024-04-02", "language": "en", "category": "finance",
              "category_confidence": 0.97, "tags": ["finance"], "near_duplicate_of": null,
              "chunk_count": 3, "metadata": {"keywords": ["..."], "entities": {"MONEY": ["USD 8940.00"]}, "...": "..."}},
 "duplicate": false, "job": null}
```

### `GET /v1/documents` (reader)

Query: `status`, `category`, `language`, `doc_type`, `tag`, `q` (title substring), `sort`
(`created_at`, `updated_at`, `title`, `size_bytes`, `document_date`), `order` (`asc`/`desc`),
`limit` (1-200), `offset`. Returns `{"items": [...], "total": n, "limit": l, "offset": o}`
(items omit the large `metadata` field).

### `GET /v1/documents/{id}` (reader)

Full document including `metadata` (keywords, entities, dates, word count, reading time, language
confidence, page count, category probabilities).

### `PATCH /v1/documents/{id}` (writer)

Body: any of `{"title": "...", "tags": ["..."], "category": "..."}`. Index metadata is updated
too, so filters reflect the change immediately. Setting `category` marks it as manual.

### `DELETE /v1/documents/{id}` (writer)

Removes the document, its chunks, summaries, index entries and stored file. `204`.

### `GET /v1/documents/{id}/chunks` (reader)

Ordered chunks with `text`, `token_count`, `heading_path`, `page_start`, `page_end`.

### `GET /v1/documents/{id}/content` (reader)

The original file as an attachment (`Content-Disposition`, `X-Content-Type-Options: nosniff`).

### `POST /v1/documents/{id}/reprocess` (writer)

Queues an `ingest` job that re-extracts and re-indexes the document (e.g. after changing
chunking settings). `202` with the job.

### `POST /v1/documents/{id}/summarize` (reader)

Body: `{"method": "textrank" | "mapreduce", "max_sentences": 5, "background": false}`.
Returns `{"method", "summary", "sentences", "provider", "stages"}`; cached summaries report
`provider: "cache"`. With `background: true` a `summarize` job is returned (`202`).

### `POST /v1/documents/{id}/classify` (writer)

Query `apply=true|false`. Returns `{"label", "confidence", "probabilities"}`; with `apply` the
category is stored unless it was set manually. `409 model_not_trained` without a model.

## Search and question answering

### `POST /v1/search` (reader)

```json
{"query": "late payment fee", "mode": "hybrid", "top_k": 5,
 "filters": {"tags": "finance", "document_date": {"$gte": "2024-01-01"}},
 "group_by_document": false}
```

Query parameter `include_text=true` adds the full chunk text. Each hit has `chunk_id`,
`document_id`, `title`, `snippet`, `highlights` (character offsets into the snippet), `score`
(RRF score for hybrid, cosine for semantic, BM25 for lexical), `vector_score`, `bm25_score`,
`ranks`, `heading_path`, `page_start`, `page_end` and `metadata` (category, language, doc_type,
tags). `warnings` explains degraded modes (e.g. vectors not available yet).

**Filter language.** Fields: `document_id`, `doc_type`, `language`, `category`, `document_date`,
`tags`, `title`, `filename`, `section`, `chunk_ordinal`. Operators: equality (list fields match
membership), `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$exists`,
`$contains` (case-insensitive substring), and the combinators `$and`, `$or`, `$not`
(nesting depth limited to 8).

### `POST /v1/ask` (reader)

```json
{"question": "What is the total amount due on invoice INV-7781?",
 "filters": {"category": "finance"}, "top_k": 5, "mode": "hybrid"}
```

```json
{"question": "...", "answer": "The total amount due is $8,940.00. [1]",
 "answerable": true, "short_answer": "$8,940.00",
 "citations": [{"number": 1, "document_id": "doc_...", "chunk_id": "chk_...",
                "title": "NORTHWIND SUPPLIES INVOICE",
                "source": "NORTHWIND SUPPLIES INVOICE, Payment Terms",
                "page_start": null, "page_end": null, "heading_path": ["..."], "snippet": "..."}],
 "provider": "extractive", "model": "extractive-v1", "took_ms": 6.1,
 "retrieval": {"mode": "hybrid", "retrieved": 9, "reranked": 5, "context_passages": 3,
               "context_tokens": 152, "context_truncated": false, "warnings": []},
 "usage": {}}
```

When nothing relevant is found, `answerable` is `false`, the answer says so and there are no
citations.

### `POST /v1/summarize` (reader)

`{"text": "...", "method": "textrank", "max_sentences": 3}` summarises inline text.

## Classification

| Endpoint | Role | Description |
| --- | --- | --- |
| `GET /v1/classifier` | reader | `{"trained": false}` or model info: id, algorithm, labels, label counts, hold-out and CV metrics |
| `POST /v1/classifier/train` | admin | Body `{"examples": [{"label", "text"} or {"label", "document_id"}], "algorithm": "naive_bayes"/"logistic", "test_ratio": 0.2, "folds": 0, "background": false}` |
| `POST /v1/classify` | reader | `{"text": "..."}` -> `{"label", "confidence", "probabilities"}` |

Training needs at least 4 examples and 2 labels. The returned `holdout` report (when
`test_ratio > 0` and every label has 2+ examples) contains accuracy, macro/micro/weighted F1,
per-class precision/recall/F1 and the confusion matrix. The deployed model is then trained on all
examples.

## Jobs

| Endpoint | Role | Description |
| --- | --- | --- |
| `GET /v1/jobs?status=&type=&limit=&offset=` | reader | List jobs, newest first |
| `GET /v1/jobs/{id}` | reader | Job with `status` (`queued`, `running`, `succeeded`, `failed`, `cancelled`), `attempts`, `result`, `error` |
| `POST /v1/jobs/{id}/cancel` | writer | Cancel a queued job (`409` otherwise) |
| `POST /v1/jobs/{id}/retry` | admin | Requeue a failed or cancelled job with a fresh attempt budget |

Job types: `ingest`, `summarize`, `reindex`, `train_classifier`.

## System

| Endpoint | Role | Description |
| --- | --- | --- |
| `GET /health` | public | Database, index, worker and LLM checks; `200` when ok, `503` when degraded |
| `GET /v1/stats` | reader | Counts by status/category, chunks, job counts, index and classifier details |
| `GET /metrics` | reader* | Prometheus text format (see docs/operations.md) |
| `POST /v1/admin/reindex` | admin | `{"refit": true, "background": true}`; rebuilds indexes from the database (refits LSA) |
