# Intelligent Document Engine

A local-first document intelligence platform written in Python 3.11. It ingests PDF, DOCX,
Markdown, HTML and plain-text files, extracts structure and metadata, classifies documents,
indexes them for lexical, semantic and hybrid search, and answers questions with cited sources
(retrieval-augmented generation). Everything runs offline with no paid APIs: the default
"LLM" is a deterministic extractive answerer, and the embeddings, vector indexes, BM25,
classifiers and summarisers are implemented in the project with numpy and the standard library.

It ships as a Python library, a FastAPI REST service and a command line tool that share the
same storage (SQLite plus files on disk).

## Contents

- [Implemented functionality](#implemented-functionality)
- [Technology stack](#technology-stack)
- [Architecture at a glance](#architecture-at-a-glance)
- [Installation and quick start](#installation-and-quick-start)
- [Configuration](#configuration)
- [Using the API](#using-the-api)
- [Command line](#command-line)
- [Testing and evaluation](#testing-and-evaluation)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [AI-assisted development](#ai-assisted-development)

## Implemented functionality

**Ingestion**
- Formats: PDF (pypdf, page numbers kept), DOCX (python-docx: headings, list styles, tables,
  core properties), HTML (stdlib parser that drops scripts, navigation, forms and footers),
  Markdown (front matter, ATX/setext headings, code fences, tables, lists) and plain text.
- Type detection from magic bytes, not from the file name. Executables, images, archives, legacy
  Office files and non-DOCX ZIP containers are rejected; a `.pdf`/`.docx` extension that does not
  match the content is an error. Size limit, PDF page limit, encrypted-PDF handling and DOCX
  zip-bomb checks (uncompressed size and compression ratio).
- Deduplication: SHA-256 of the bytes (identical uploads return the existing document), a hash of
  the normalised text (same text in another format), and 64-bit SimHash with LSH banding in SQLite
  for near duplicates (flagged as `near_duplicate_of`).
- Cleaning: Unicode NFKC, ligatures, smart quotes, zero-width and control characters,
  de-hyphenation of line breaks.
- Structure recovery for unstructured text (numbered headings, ALL-CAPS and Title Case lines,
  setext underlines, list items).
- Chunking: token-aware, sentence-level overlap, heading-aware (a chunk never crosses a section and
  keeps its heading path and page range), hard splitting of over-long sentences, table row splitting.

**Metadata and classification**
- Title, authors (native properties or "By ..." lines), document date (native or labelled date
  lines), language (Cavnar-Trenkle character n-gram profiles for en/es/fr/de/it/pt/nl),
  keywords (recurring phrases + RAKE + corpus-aware TF-IDF), entities (EMAIL, URL, PHONE, MONEY
  with normalised amounts, PERCENT, DATE, ORG, PERSON, LOCATION via regexes and gazetteers).
- Trainable classifier: multinomial naive Bayes or softmax logistic regression (Adam, L2) in numpy,
  stratified hold-out and k-fold cross-validation, persisted model with metadata; optional
  automatic classification on ingest (manual categories are never overwritten).

**Retrieval and generation**
- `EmbeddingProvider` interface with three implementations: a stateless hashing-trick embedder with
  an implicit sparse random projection (default), LSA (TF-IDF + randomized truncated SVD over a
  CSR matrix, needs a fit/reindex), and an optional `sentence-transformers` adapter (not installed
  by default).
- Own vector indexes: exact (flat), IVF (spherical k-means, adaptive probing) and HNSW (heuristic
  neighbour selection, tombstones), all with metadata filtering and crash-safe persistence.
- BM25 inverted index; hybrid search with reciprocal rank fusion; query-focused snippets with
  highlight offsets; MongoDB-style filter language (`$in`, `$gte`, `$or`, `$not`, ...).
- RAG pipeline: hybrid retrieval -> reranking (retrieval score, IDF-weighted term coverage, term
  proximity, embedding similarity) with MMR diversification -> numbered context within a token
  budget -> LLM adapter -> answer with citations (document, page range, section). The extractive
  adapter declines to answer when the context does not support an answer.
- LLM adapters: extractive (default, offline), Ollama and any OpenAI-compatible chat API (disabled
  by default; timeouts, retries with backoff; tested with mocked HTTP only).
- Summarisation: TextRank (local) and map-reduce over chunks through the LLM adapter with
  recursive collapse; summaries are cached per document.

**Platform**
- FastAPI service: upload, document CRUD, chunks, original content download, search, ask,
  summarise, classify/train, jobs, stats, health and Prometheus metrics. API keys with
  reader/writer/admin roles, per-key token-bucket rate limiting, request body limit, consistent
  error envelope with request ids.
- Background job queue persisted in SQLite: worker threads, atomic claim with leases, recovery of
  expired leases, retries with exponential backoff and jitter, permanent-error detection,
  cancel/retry.
- Structured JSON logging with request/job correlation ids; metrics registry with Prometheus
  exposition.
- Evaluation: retrieval (recall@k, precision@3, MRR, nDCG@10), QA (exact match, token F1, answer recall,
  citation accuracy, abstention accuracy) and classification metrics over a golden dataset, a CLI
  `eval` command and a committed baseline enforced by a regression test.
- CLI for ingest, search, ask, summarize, docs, classify, eval, jobs, reindex, stats, config, serve.

## Technology stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.11 |
| HTTP API | FastAPI 0.143, Starlette, Uvicorn |
| Validation / config | pydantic 2, TOML (`tomllib`), environment variables |
| Parsing | pypdf, python-docx, stdlib `html.parser`, own Markdown parser |
| Numerics | numpy (embeddings, SVD, k-means, classifiers, TextRank) |
| Storage | SQLite (stdlib `sqlite3`, WAL), content-addressed files, JSON/NPZ index files |
| HTTP client | httpx (optional LLM adapters) |
| Quality | pytest, ruff |

All runtime dependencies are pinned in `pyproject.toml`; the full resolved set is in
`requirements.lock`.

## Architecture at a glance

```
            +-------------+      +-------------+
 files ---> | FastAPI API | ---> |             | ---> SQLite (documents, chunks, jobs, summaries)
            +-------------+      |   Engine    | ---> blobs/ (original files, by SHA-256)
            |     CLI     | ---> | (composition| ---> indexes/ (vectors-N.npz + vectors.json, bm25.json)
            +-------------+      |    root)    | ---> models/ (classifier)
                                 +-------------+
   ingest: detect -> extract -> clean -> metadata -> chunk -> embed + BM25 -> classify
   ask:    hybrid retrieve -> rerank + MMR -> context with citations -> LLM adapter
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map, data flow and design decisions, and
[docs/](docs/) for the data model, search/RAG internals, operations runbook and testing guide.

## Installation and quick start

Requirements: Python 3.11. No C compiler, external database or network service is needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .

idengine ingest tests/fixtures/golden/corpus --tags sample
idengine search "how long is personal data kept"
idengine ask "What is the monthly rent for the office lease?"
```

Run the API (keys are required unless authentication is explicitly disabled):

```bash
export IDE_API_KEYS="dev-admin-key:admin,dev-reader-key:reader"   # PowerShell: $env:IDE_API_KEYS="..."
idengine serve --port 8000
curl -s localhost:8000/health
```

[GETTING_STARTED.md](GETTING_STARTED.md) walks through a full session, including the classifier
and the evaluation command. `examples/quickstart.py` shows library usage.

## Configuration

Settings come from defaults, an optional TOML file (`--config` or `IDE_CONFIG_FILE`, see
`examples/engine.toml`) and environment variables `IDE_<SECTION>__<FIELD>` (for example
`IDE_INDEX__KIND=ivf`), plus short aliases such as `IDE_DATA_DIR`, `IDE_API_KEYS`,
`IDE_LLM_PROVIDER`. Everything is validated at start-up; invalid values stop the process with a
readable message. `idengine config` prints the effective configuration with secrets masked.
`.env.example` lists the most important variables.

Key sections: `storage`, `ingest`, `chunking`, `embeddings`, `index`, `search`, `rag`, `llm`,
`jobs`, `classifier`, `api`, `logging` (see `src/idengine/config.py` for every field and default).

## Using the API

All `/v1/*` endpoints require an API key (`X-API-Key` header or `Authorization: Bearer`).
`/health` is public; `/metrics` requires a key unless `api.metrics_public` is enabled.

```bash
KEY=dev-admin-key
curl -s -X POST "localhost:8000/v1/documents?wait=true" -H "X-API-Key: $KEY" \
     -F "file=@tests/fixtures/golden/corpus/f_invoice.txt" -F "tags=finance"
curl -s -X POST localhost:8000/v1/search -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"query": "late payment fee", "mode": "hybrid", "top_k": 3}'
curl -s -X POST localhost:8000/v1/ask -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"question": "What is the total amount due on invoice INV-7781?"}'
```

Uploads are processed in the background by default (`202` with a job); `?wait=true` processes
synchronously (`201`); an identical file returns the existing document (`200`). Full reference:
[API.md](API.md). Interactive OpenAPI docs are served at `/docs`.

## Command line

```
idengine [--config FILE] [--data-dir DIR] [--json] <command>
  ingest PATH... [--recursive] [--tags ...] [--category C] [--background]
  search QUERY [--mode semantic|lexical|hybrid] [--top-k N] [--filter JSON] [--group]
  ask QUESTION [--filter JSON]
  summarize DOC_ID | --file FILE [--method textrank|mapreduce] [--sentences N]
  docs list|show|delete
  classify train FILE.jsonl | predict TEXT | document DOC_ID
  eval [--dataset DIR] [--baseline FILE] [--update-baseline] [--tolerance 0.02]
  jobs list|run|purge     reindex [--no-refit]     stats     config     serve
```

## Testing and evaluation

```bash
python -m pytest -q          # 422 passed (unit + integration, ~11 s on the dev machine)
ruff check . && ruff format --check .
idengine eval                # golden dataset; exits 1 if a metric regresses beyond tolerance
```

The suite contains unit tests per module, seeded property tests, ASGI-level middleware tests and
integration tests that run the FastAPI app in-process with `TestClient`, the CLI and the engine
against real SQLite files. PDF and DOCX fixtures are generated programmatically
(`tests/builders.py`). External LLM adapters are tested with `httpx.MockTransport`; no test touches
the network.

The golden dataset (`tests/fixtures/golden/`) is small and synthetic: 12 documents in four
categories, 22 retrieval queries, 17 QA cases (2 unanswerable) and 56 labelled texts. Current
baseline (`tests/fixtures/golden/baseline.json`), all deterministic:

| Metric | Lexical | Semantic (hashing) | Hybrid |
| --- | --- | --- | --- |
| recall@1 | 0.977 | 0.841 | 0.977 |
| recall@5 | 1.000 | 1.000 | 1.000 |
| MRR | 1.000 | 0.932 | 1.000 |
| nDCG@10 | 0.986 | 0.931 | 0.986 |

QA: exact match 0.933, token F1 0.988, citation accuracy 1.0, abstention accuracy 1.0.
Classification (naive Bayes, 40 train / 16 test texts): accuracy 1.0. These numbers come from a
tiny corpus that was also used while developing the heuristics, so they show that the pipeline
works end to end and guard against regressions; they are **not** a benchmark of real-world
quality. See [docs/testing.md](docs/testing.md).

### Verification log (2026-10-12, Windows 11, Python 3.11.9)

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 422 passed |
| `ruff check .` / `ruff format --check .` | clean |
| `python -m compileall -q src` | ok |
| `idengine eval` | no regressions against `baseline.json` |
| Live smoke test: `uvicorn idengine.api.app:build_app_from_env --factory`, then `examples/smoke.sh` | `/health` ok; upload `201` (ready, 3 chunks); search returned the invoice chunk first; ask answered "$8,940.00" citing "NORTHWIND SUPPLIES INVOICE, Payment Terms"; async upload `202` -> job succeeded; request without key `401`; `/metrics` exposed request counters |
| `examples/api_client.py` against the live server | both uploads processed by jobs; question answered with citation |

## Deployment

A multi-stage `Dockerfile` (non-root user, health check, data volume) and `compose.yaml` (with an
optional Ollama profile) are provided. **They were written carefully but not built or run locally,
because Docker is not installed on the development machine.** The CI workflow
(`.github/workflows/ci.yml`) runs lint, tests and the evaluation gate on Linux and Windows and
builds the image; it is a local file that has not been executed on GitHub. Operational guidance
(backups, reindexing, monitoring) is in [docs/operations.md](docs/operations.md).

## Limitations

- Single-node only: SQLite and in-process indexes; no horizontal scaling or shared index service.
  Indexes are held in memory and rewritten in full on flush, which suits corpora up to tens of
  thousands of chunks, not millions.
- The HNSW and IVF implementations are pure Python/numpy: correct and tested for recall, but far
  slower to build than FAISS/hnswlib (about 2 ms per HNSW insert with the default `ef_construction=100`, measured on 64-dimensional vectors on the dev machine).
- No OCR: scanned PDFs without a text layer are rejected with an explanatory error.
- The default embedder is lexical in nature (hashed bag of words and bigrams); it does not capture
  synonyms the way neural embeddings do. LSA helps somewhat; real semantic models require the
  optional sentence-transformers adapter, which is untested beyond a mocked module.
- The extractive answerer only quotes sentences from the context; it cannot synthesise, compute or
  reason across documents. Short answers depend on regex entity types.
- Language detection covers seven European languages; keyword extraction and stemming are
  English-only (keywords are skipped for other languages).
- Entity recognition is rule based (regexes and small gazetteers) and will miss or mislabel many
  names.
- Ollama/OpenAI-compatible adapters were only exercised against mocked HTTP responses, never a
  live server.
- Jobs whose runtime exceeds the lease (`jobs.lease_seconds`) are only requeued when a worker pool
  restarts; there is no lease renewal heartbeat.
- API keys are static configuration; there is no user management, key rotation endpoint or audit
  log.

## Roadmap

- Incremental index persistence (append-only segments) instead of full rewrites.
- Optional native ANN backend (hnswlib/FAISS) behind the existing `VectorIndex` interface.
- OCR for scanned PDFs (Tesseract adapter).
- Cross-encoder reranking and query expansion when a neural model is available.
- Job lease heartbeats, scheduled jobs and a dead-letter view.
- Multi-tenant collections with per-key document scoping.
- A larger, independently written evaluation set.

## AI-assisted development

This project was developed with AI assistance (Claude by Anthropic). The code, tests and
documentation were generated in an AI-assisted workflow and reviewed and verified by running the
test suite, the linters, the evaluation command and a live smoke test of the API (upload, search,
ask) with Uvicorn. Claims in this document are limited to what was actually run; the Docker image
and the CI workflow were not executed.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Robert Ramos.
