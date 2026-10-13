# Architecture

This document describes how the Intelligent Document Engine is structured, how data flows
through it and why the main design decisions were made. Deeper material lives in
[docs/data-model.md](docs/data-model.md), [docs/search-and-rag.md](docs/search-and-rag.md),
[docs/operations.md](docs/operations.md) and [docs/testing.md](docs/testing.md).

## Goals and constraints

- Run entirely on one machine with no paid or external services; every AI component has a local,
  deterministic implementation.
- Build without a C compiler: only pure-Python wheels plus numpy.
- One source of truth (SQLite + original files) from which every derived structure (indexes,
  summaries) can be rebuilt.
- The same engine is used by the HTTP API, the CLI, the evaluation runner and the tests.

## Package map (`src/idengine`)

| Package / module | Responsibility |
| --- | --- |
| `config.py` | Pydantic settings, TOML + environment loading, validation, roles |
| `errors.py` | Typed error hierarchy; each error has a stable `code` and HTTP status |
| `engine.py` | Composition root: builds services, registers job handlers, facade methods |
| `observability/` | JSON logging with request/job ids, metrics registry + Prometheus text |
| `text/` | Normalisation, tokenisation, sentence splitting, Porter stemmer, analyzer |
| `ingest/` | Type detection, extractors (pdf/docx/html/markdown/txt), structure heuristics, hashing, chunker, ingestion service |
| `analysis/` | Language detection, dates, entities, keywords, document metadata |
| `classify/` | Vectorizer, naive Bayes, softmax regression, persistence, training service |
| `embeddings/` | `EmbeddingProvider` interface; hashing, LSA, optional sentence-transformers |
| `vectorindex/` | Filter language, shared vector storage, flat / IVF / HNSW, k-means |
| `search/` | BM25, rank fusion, snippets, `IndexManager`, `SearchService` |
| `llm/` | `LLMAdapter` interface, extractive model, Ollama and OpenAI-compatible adapters, prompts |
| `summarize/` | TextRank, map-reduce summariser |
| `rag/` | Reranker (features + MMR), context assembly with citations, `RagPipeline` |
| `storage/` | SQLite connection management and migrations, repositories, blob store, job table |
| `jobs/` | Job queue: worker threads, retries with backoff, lease recovery |
| `evaluation/` | Retrieval/QA/classification metrics, golden dataset loader, runner, baseline |
| `api/` | FastAPI app factory, middleware, security, schemas, routers |
| `cli.py` | `idengine` command |

Dependencies point downwards: `api`/`cli` -> `engine` -> services (`ingest`, `search`, `rag`,
`classify`, `summarize`, `jobs`) -> primitives (`text`, `analysis`, `embeddings`, `vectorindex`,
`llm`, `storage`). The metric modules in `evaluation` have no engine dependency.

## Ingestion flow

```
upload bytes
  -> sanitize filename, detect type from magic bytes (+ extension cross-check, size limit)
  -> SHA-256: existing document? return it (duplicate)
  -> store blob (content-addressed), insert document row (status=processing)
  -> [inline, or as an "ingest" job]
  -> extract blocks (heading / paragraph / list / table / code, with page numbers)
  -> metadata: title, authors, date, language, keywords (TF-IDF uses live BM25 statistics), entities
  -> text hash + SimHash -> duplicate / near-duplicate lookup (LSH bands in SQLite)
  -> chunk (heading-aware, token budget, sentence overlap)
  -> replace chunk rows atomically; drop stale chunks from indexes
  -> classify (if a model exists and the category was not set manually)
  -> document row status=ready; index chunks (BM25 + embeddings -> vector index); flush indexes
```

Any failure marks the document `failed` with the error message; the upload response (or the job)
reports the typed error. Re-uploading the same bytes re-processes a failed document.

## Query flow

```
search(query, mode, filters)
  -> validate, compile filter predicate
  -> vector candidates (embed query -> vector index search with predicate)
  -> BM25 candidates (analyzed terms -> inverted index with predicate)
  -> hybrid: reciprocal rank fusion of the two rankings
  -> hydrate chunk texts from SQLite, build snippets + highlight offsets

ask(question)
  -> hybrid search (retrieve_k candidates)
  -> rerank: retrieval score, IDF-weighted coverage, proximity, embedding similarity
  -> MMR selection (rerank_k) to avoid redundant overlapping chunks
  -> drop passages below min_relevance; assemble numbered context within the token budget
  -> LLMAdapter.generate(task=answer) -> answer text with [n] markers
  -> map markers to citations (document, title, page range, section, snippet)
```

## Key design decisions

**SQLite is the source of truth.** Documents, chunks, jobs and summaries live in SQLite (WAL mode,
one connection per thread, explicit transactions). Index files are derived data: on start-up the
`IndexManager` loads them, checks they were built for the current embedding space (provider
signature) and that their size matches the chunk table, and otherwise rebuilds them from the
stored chunk texts. A corrupt index file therefore costs a rebuild, never data.

**Crash-safe persistence.** JSON files are written to a temporary file, fsynced and renamed.
Vector data uses generations: `vectors-N.npz` is written first, then the manifest
`vectors.json` is atomically switched to it, then the previous generation is deleted. On Windows
the rename is retried briefly because antivirus scanners can hold freshly written files.

**Stateless default embedder.** The hashing embedder maps stemmed unigrams and bigrams through a
keyed BLAKE2b hash onto a few signed dimensions (an implicit very sparse random projection). It
needs no training, so documents can be indexed incrementally and vectors never go stale. LSA is
available when a corpus-fitted space is preferable; because it must be fitted, it is enabled via
`reindex`, and hybrid search degrades to BM25 with a warning until then.

**Own vector indexes behind one interface.** `VectorIndex` provides storage (dense matrix, ids,
metadata, tombstones), locking, filtering and persistence; subclasses add structure. Flat is exact.
IVF trains spherical k-means lazily and retrains when the collection doubles; filtered queries
probe more partitions until enough candidates exist. HNSW follows Malkov & Yashunin with the
neighbour-selection heuristic; deletions are tombstones, compacted by `rebuild()`. Filtered HNSW
queries widen the beam and fall back to an exact scan when too few matches survive, so filters never
silently reduce recall.

**Hybrid by default.** BM25 handles exact identifiers and rare terms; the vector side helps with
morphology and phrasing. Reciprocal rank fusion needs no score calibration between the two.

**Deterministic local LLM.** The extractive adapter scores context sentences by IDF-weighted
overlap with the question, phrase matches, source rank and whether the sentence contains the entity
type the question asks for (date, amount, person, ...). It cites every sentence and abstains below a
relevance threshold. External adapters (Ollama, OpenAI-compatible) implement the same interface and
use the same numbered-context prompt, so citations work identically; they are opt-in.

**Background jobs in SQLite.** Jobs are claimed with an atomic `UPDATE` inside `BEGIN IMMEDIATE`
and leased to a worker. Retries use exponential backoff with equal jitter; validation, not-found and
conflict errors are treated as permanent. Expired leases (crashed workers) are requeued at start-up.

**Typed errors everywhere.** Library code raises `EngineError` subclasses carrying `code`,
`http_status` and `details`; the API turns them into a single error envelope, and the CLI prints
them and exits non-zero. Unexpected exceptions become `internal_error` without leaking internals.

## Concurrency model

- FastAPI runs synchronous endpoints in its thread pool; uploads are read asynchronously with a
  size cap and processed in the pool.
- Each thread uses its own SQLite connection. Writers serialise on SQLite's lock (busy timeout 10 s).
- `IndexManager` mutations hold a re-entrant lock; BM25 and vector index searches hold their own
  locks, so searches run concurrently with ingestion (covered by a concurrency test).
- The job queue runs `jobs.workers` daemon threads that wait on a condition variable and poll at
  `jobs.poll_interval_seconds`.

## Security model (summary)

API keys map to roles (`reader` < `writer` < `admin`); keys are compared as SHA-256 digests in
constant time, and only short fingerprints appear in logs. Request bodies, uploads, queries,
filters, training sets and inline texts are size-limited; uploaded files are type-checked by
content. See [SECURITY.md](SECURITY.md).

## Extension points

- New file format: add an extractor returning `ExtractedDocument` and register it in
  `ingest/extractors/__init__.py` and `ingest/detect.py`.
- New embedding model: implement `EmbeddingProvider` (`dim`, `embed`, optional `fit/save/load`,
  `signature`) and add it to `create_embedder`.
- New ANN structure: subclass `VectorIndex` and implement `_search` (plus `_on_add`/`_on_remove`
  and persistence hooks as needed).
- New LLM backend: implement `LLMAdapter.generate`; reuse `prompts.render_messages` and
  `parse_citations`.
- New background task: `engine.jobs.register("name", handler)`.
