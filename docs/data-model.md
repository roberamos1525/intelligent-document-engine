# Data model and on-disk layout

Everything lives under the data directory (`storage.data_dir`, default `./var`):

```
var/
  engine.sqlite3            SQLite database (WAL mode: also engine.sqlite3-wal / -shm)
  blobs/ab/cd/<sha256>      original uploaded files, content-addressed
  indexes/
    vectors.json            vector index manifest (current generation, ids, metadata, graph)
    vectors-<N>.npz         vector data for generation N (+ IVF centroids)
    bm25.json               BM25 document table (term frequencies, lengths, metadata)
    lsa.npz, lsa_vocab.json LSA model (only with embeddings.provider = "lsa")
  models/
    classifier.json         classifier manifest: algorithm, labels, vocabulary, settings, info
    classifier.npz          classifier arrays (IDF, weights or log-probabilities)
```

SQLite and the blobs are the source of truth. Everything under `indexes/` can be deleted at any
time; it is rebuilt from the database on the next start (or with `idengine reindex`).

## SQLite schema

Migrations are applied in order at start-up and recorded in `schema_migrations(version,
applied_at)`. Current version: 2.

### `documents`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PK | `doc_<uuid4 hex>` |
| `filename` | TEXT | sanitised client file name |
| `doc_type` | TEXT | `pdf`, `docx`, `txt`, `markdown`, `html` |
| `mime_type`, `size_bytes` | TEXT, INTEGER | |
| `content_hash` | TEXT UNIQUE | SHA-256 of the bytes; exact dedup and blob key |
| `text_hash` | TEXT, indexed | SHA-256 of the canonical text (format-independent dedup) |
| `simhash` | INTEGER | 64-bit SimHash stored as signed 64-bit |
| `sim_band0..3` | INTEGER, indexed | 16-bit bands of the SimHash for LSH candidate lookup |
| `status` | TEXT, indexed | `processing`, `ready`, `failed` |
| `title`, `document_date`, `language` | TEXT | extracted metadata |
| `authors`, `tags` | TEXT (JSON list) | |
| `category`, `category_confidence` | TEXT, REAL | manual or model category |
| `metadata` | TEXT (JSON) | keywords, entities, dates, word count, reading time, category source/probabilities, extractor properties |
| `near_duplicate_of` | TEXT | id of an earlier document with the same text or SimHash distance <= threshold |
| `chunk_count`, `error` | INTEGER, TEXT | |
| `created_at`, `updated_at` | TEXT | ISO-8601 UTC with milliseconds |

Near-duplicate lookup: with four 16-bit bands, any two fingerprints within Hamming distance 3
share at least one band exactly (pigeonhole), so `WHERE sim_band0 = ? OR ... sim_band3 = ?`
returns every candidate via indexes; distances are then checked in Python. Thresholds above 3
fall back to a full scan.

### `chunks`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PK | `chk_<uuid4 hex>` (also the id in both indexes) |
| `document_id` | TEXT FK -> documents ON DELETE CASCADE | indexed |
| `ordinal` | INTEGER | position in the document; `UNIQUE(document_id, ordinal)` |
| `text`, `token_count` | TEXT, INTEGER | chunk body and approximate token count |
| `heading_path` | TEXT (JSON list) | e.g. `["Contract", "Payment"]` |
| `page_start`, `page_end` | INTEGER | PDF pages covered (NULL for other formats) |

Chunks of a document are replaced atomically in one transaction when a document is (re)processed.
The text that is embedded and BM25-indexed is `heading path + "\n" + text`, so section titles
contribute to retrieval.

### `jobs`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PK | `job_<uuid4 hex>` |
| `type`, `payload` | TEXT, TEXT (JSON) | `ingest`, `summarize`, `reindex`, `train_classifier` |
| `status` | TEXT | `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `priority` | INTEGER | higher first (-100..100) |
| `attempts`, `max_attempts` | INTEGER | |
| `run_after` | REAL (epoch seconds) | scheduling and retry backoff; `(status, run_after)` indexed |
| `locked_by`, `lease_expires` | TEXT, REAL | worker lease |
| `result`, `error` | TEXT (JSON), TEXT | |
| `created_at`, `updated_at`, `started_at`, `finished_at` | TEXT | |

### `summaries`

`(document_id, method)` primary key, `summary`, `created_at`. The `method` key encodes method,
sentence count and LLM provider (e.g. `textrank:5:extractive`). Rows are deleted with the
document and invalidated when it is reprocessed.

## Index files

**Vector index (`vectors.json` + `vectors-N.npz`).** The manifest stores the format version, index
kind, generation number, the embedding *signature* (provider, dimension, seed, ...), the external
ids and per-row metadata (document id, type, language, category, date, tags, title, filename,
section, chunk ordinal) and kind-specific structure: IVF inverted lists and trained size, or the
HNSW levels, adjacency lists, entry point and maximum level. The NPZ holds the vector matrix, the
alive mask (tombstones) and IVF centroids. A save writes `vectors-(N+1).npz`, atomically replaces
the manifest, then deletes generation N.

**BM25 (`bm25.json`).** Analyzer version plus, per chunk id, the term-frequency map, document
length and filter metadata. Postings are rebuilt in memory on load.

**Staleness rules.** An index is rebuilt when its file is missing or corrupt, when its kind or
embedding signature differs from the configuration, or when its entry count differs from the
number of chunks belonging to `ready` documents.

## Document metadata (`documents.metadata`)

```json
{
  "title": "...", "authors": ["..."], "document_date": "2024-04-02",
  "language": "en", "language_confidence": 1.0,
  "keywords": ["late payments", "total amount due", "invoice"],
  "entities": {"MONEY": ["USD 8940.00"], "DATE": ["2024-04-02"], "EMAIL": ["..."], "ORG": ["..."]},
  "dates": ["2024-04-02"], "word_count": 92, "reading_minutes": 0.4,
  "heading_count": 3, "page_count": null,
  "category_source": "model", "category_probabilities": {"finance": 0.97, "legal": 0.01},
  "extra": {"encoding": "utf-8"}
}
```
