# Testing and evaluation

## Commands

```bash
python -m pytest -q                      # whole suite
python -m pytest tests/unit -q           # fast unit tests only
python -m pytest tests/integration -q    # engine, API (TestClient), CLI, regression gate
python -m pytest -k hnsw -q              # by keyword
ruff check . && ruff format --check .    # lint and formatting
idengine eval                            # golden evaluation with baseline comparison
```

The suite needs no network, no external services and no pre-built fixtures. Each test that needs
storage gets its own temporary data directory.

## Layout

```
tests/
  conftest.py               settings/engine fixtures, API keys used by the API tests
  builders.py               programmatic PDF writer and DOCX builder, sample texts
  fixtures/golden/          evaluation corpus, dataset.json, baseline.json
  unit/                     one module per component
  integration/              engine end-to-end, FastAPI TestClient, CLI, regression gate
```

| Area | Unit tests | Integration tests |
| --- | --- | --- |
| Config | layering, aliases, validation errors, secret masking | CLI `config` |
| Text | Porter stemmer reference pairs, sentence splitting, normalisation | |
| Ingestion | magic bytes, extension mismatch, zip bombs, encrypted PDFs, every extractor, chunker invariants | all formats through engine/API/CLI, dedup, failure recovery, reprocessing |
| Analysis | 7 languages, date formats, every entity label, keywords, metadata | metadata via engine/API |
| Storage | migrations, transactions, repositories, LSH near-dup lookup, blobs, job table | persistence across restarts |
| Vector indexes | filter DSL, recall of HNSW/IVF vs exact, tombstones, persistence, corruption | index rebuild after corruption, LSA reindex |
| Search | BM25 scoring/persistence, RRF, snippets | modes, filters, grouping, concurrency |
| Classification | NB/softmax, vectorizer, splits, CV, reports, service | training via engine/API/CLI, auto-classify |
| LLM | extractive answers/abstention, prompts, Ollama/OpenAI adapters with `httpx.MockTransport` (success, retries, timeouts, malformed responses) | `/v1/ask` |
| Summaries/RAG | PageRank, TextRank, map-reduce stages, rerank features, MMR, context budget | summaries + cache, ask with filters |
| Jobs | retries/backoff, permanent errors, exhaustion, worker threads, lease recovery | async uploads, job endpoints |
| API | middleware at ASGI level (body limits, request ids, metrics) | auth/roles, rate limits, size limits, error envelope, every endpoint |

`tests/unit/test_properties.py` contains seeded randomised invariant tests (chunker never loses
sentences or exceeds its budget, HNSW agrees with exact search, BM25 is monotonic in term
frequency, compiled filters match Python semantics, RRF is order invariant).

## Golden dataset and regression gate

`tests/fixtures/golden/dataset.json` defines:

- 12 short synthetic documents (finance, legal, HR, engineering) in TXT, Markdown and HTML,
- 22 retrieval queries with document-level relevance (two with graded relevance),
- 17 QA cases with accepted answers and the expected source document, including 2 questions that
  must be declined,
- a classification split of 40 training and 16 test texts.

`idengine eval` ingests the corpus into a temporary engine and computes:

| Group | Metrics |
| --- | --- |
| Retrieval (per mode: lexical, semantic, hybrid) | recall@1/3/5/10, precision@3, MRR, nDCG@10 (document level) |
| QA | exact match and token F1 (short answer, or the answer without citation markers), answer recall, citation accuracy, abstention accuracy |
| Classification | accuracy, macro F1 |

`baseline.json` stores the expected values. `tests/integration/test_regression.py` fails if any
metric drops more than 0.02 below its baseline and also enforces absolute floors. The CLI command
exits with status 1 on a regression, so CI can use it directly.

Interpretation: the dataset is tiny and was used while tuning the extractive heuristics, so its
numbers demonstrate correct wiring and protect against regressions; they do not estimate
real-world accuracy. Extend the dataset (or point `--dataset` at your own directory with the same
schema) before drawing quality conclusions.

## Writing tests

- Prefer real components over mocks; mock only the network (`httpx.MockTransport`) and, where a
  failure must be simulated, a single method via `monkeypatch`.
- Generate PDFs/DOCX with `tests/builders.py`.
- Use seeds for anything random.
- Integration tests that need the API use `TestClient(create_app(settings))` with
  `make_settings(tmp_path / ...)` from `conftest.py`.
