# Getting started

This guide takes you from a fresh checkout to ingesting documents, searching, asking questions,
training the classifier and running the evaluation, first with the CLI and then over HTTP.

## 1. Install

Python 3.11 is required.

```bash
cd intelligent-document-engine
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
python -m pip install -r requirements.lock      # exact, pinned versions (includes pytest and ruff)
python -m pip install --no-deps -e .            # the idengine package and CLI
idengine --version
```

Data is written to `./var` by default. Use `--data-dir` or `IDE_DATA_DIR` to put it elsewhere.

## 2. Ingest and explore with the CLI

```bash
idengine ingest tests/fixtures/golden/corpus --recursive --tags sample
idengine docs list
idengine docs show <document-id>          # metadata: language, keywords, entities, dates...
```

Search in the three modes, with and without a filter:

```bash
idengine search "how long is personal data kept"
idengine search "rollback" --mode lexical
idengine search "policy" --filter '{"doc_type": "markdown"}' --group
```

Ask questions; answers cite their sources and the command exits with code 2 when the documents
do not contain an answer:

```bash
idengine ask "What is the monthly rent for the office lease?"
idengine ask "Who approved the FY2025 budget?"
idengine ask "What is the boiling point of mercury?"
```

Summaries (TextRank, or map-reduce through the configured LLM adapter):

```bash
idengine summarize <document-id> --sentences 2
idengine summarize --file some-long-text.txt --method mapreduce
```

Add `--json` before the command for machine-readable output, e.g. `idengine --json stats`.

## 3. Train the classifier

Training data is JSON Lines with a `label` and either `text` or `document_id`:

```bash
idengine classify train examples/training.jsonl --folds 3
idengine classify predict "Please pay the attached invoice within 30 days."
idengine classify document <document-id>          # stores the predicted category
```

Once a model exists, newly ingested documents are classified automatically
(`ingest.auto_classify`), except when a category was given explicitly on upload.

## 4. Run the API

```bash
# bash
export IDE_API_KEYS="dev-admin-key:admin,dev-reader-key:reader"
# PowerShell
$env:IDE_API_KEYS = "dev-admin-key:admin,dev-reader-key:reader"

idengine serve --host 127.0.0.1 --port 8000
```

Or with Uvicorn directly (reads all settings from the environment):

```bash
uvicorn idengine.api.app:build_app_from_env --factory --port 8000
```

In another terminal:

```bash
curl -s localhost:8000/health
curl -s -X POST "localhost:8000/v1/documents?wait=true" -H "X-API-Key: dev-admin-key" \
     -F "file=@tests/fixtures/golden/corpus/l_lease.txt"
curl -s -X POST localhost:8000/v1/ask -H "X-API-Key: dev-reader-key" \
     -H "Content-Type: application/json" -d '{"question": "What is the monthly rent?"}'
```

`examples/smoke.sh` runs upload -> search -> ask with curl, `examples/api_client.py` does the
same with httpx (including waiting for background jobs), and `examples/requests.http` contains
ready-made requests for REST-client editors. OpenAPI documentation is at
`http://localhost:8000/docs`.

## 5. Optional components

- **IVF or exact index:** `IDE_INDEX__KIND=ivf` or `flat`. Indexes are rebuilt automatically when
  the kind changes.
- **LSA embeddings:** `IDE_EMBEDDINGS__PROVIDER=lsa`, then `idengine reindex` (or
  `POST /v1/admin/reindex`) to fit the model. Until then hybrid search uses BM25 only.
- **sentence-transformers:** `pip install sentence-transformers` (not part of the lock file), then
  `IDE_EMBEDDINGS__PROVIDER=sentence-transformers` and `idengine reindex`.
- **Ollama / OpenAI-compatible LLM:** set `IDE_LLM_PROVIDER=ollama` (or `openai`),
  `IDE_LLM_BASE_URL`, `IDE_LLM_MODEL` and, for OpenAI-compatible servers, `IDE_LLM_API_KEY`.
  These adapters are disabled by default and were only tested against mocked HTTP.

## 6. Test and evaluate

```bash
python -m pytest -q
ruff check . && ruff format --check .
idengine eval                         # compares with tests/fixtures/golden/baseline.json
idengine eval --output report.json    # full per-query details
```

If you intentionally change retrieval or answering behaviour, review the new metrics and refresh
the baseline with `idengine eval --update-baseline`.
