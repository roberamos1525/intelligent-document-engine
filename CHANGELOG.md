# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [0.1.0] - 2026-10-12

### Added
- Ingestion of PDF, DOCX, HTML, Markdown and plain text with magic-byte type detection, size and
  page limits, zip-bomb checks, text normalisation and structure recovery.
- Exact (SHA-256), normalised-text and SimHash near-duplicate detection with LSH banding.
- Heading-aware, token-aware chunking with sentence overlap and page tracking.
- Metadata extraction: title, authors, dates, language (n-gram profiles), keywords
  (RAKE + TF-IDF + recurring phrases), regex/gazetteer entities.
- Trainable classifier (multinomial naive Bayes, softmax regression) with hold-out and
  cross-validation reports and automatic classification on ingest.
- Embedding providers: hashing (default), LSA, optional sentence-transformers.
- Vector indexes: flat, IVF and HNSW with metadata filtering and generation-based persistence.
- BM25 index, hybrid search with reciprocal rank fusion, snippets with highlights.
- RAG pipeline with reranking, MMR, token-budgeted context and citations; extractive LLM plus
  opt-in Ollama and OpenAI-compatible adapters.
- TextRank and map-reduce summarisation with per-document caching.
- SQLite-backed job queue with worker threads, leases, retries and backoff.
- FastAPI service with API-key roles, rate limiting, body limits, error envelope, request ids,
  health, stats and Prometheus metrics.
- CLI (`idengine`) for ingest, search, ask, summarize, docs, classify, eval, jobs, reindex,
  stats, config and serve.
- Evaluation framework (retrieval, QA, classification metrics), golden dataset, committed
  baseline and regression test.
- Dockerfile, compose file and CI workflow (not executed locally).
