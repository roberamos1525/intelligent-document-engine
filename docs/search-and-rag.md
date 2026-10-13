# Search and RAG internals

## Text analysis

`text.tokenize.analyze` is shared by BM25, the embedders, keyword TF-IDF, the classifier and the
reranker: lowercase, word tokenisation (internal apostrophes, hyphens and dots kept, e.g. `u.s`,
`e-mail`), English stopword removal, Porter stemming for alphabetic tokens; tokens containing digits
are kept verbatim so identifiers like `inv-7781` and years match exactly.

Chunk budgets use `count_tokens`, a model-agnostic approximation (words plus punctuation marks).
It over-counts relative to typical sub-word tokenizers for plain prose, which keeps chunks safely
inside LLM context limits.

## Chunking

1. Sections are derived from heading blocks with a level stack (`ExtractedDocument.sections()`),
   so every chunk carries its full heading path.
2. Blocks become sentence-sized units (tables split by row, code kept whole when it fits, units
   longer than the budget hard-split into token windows).
3. Units are packed greedily up to `chunking.max_tokens`. When a chunk is emitted, the next one
   starts with trailing units of the previous chunk worth at most `overlap_tokens` (sentence-level
   overlap, never the whole previous chunk).
4. A trailing chunk smaller than `min_tokens` is merged into its predecessor when it fits.

## Embeddings

| Provider | How | Fit | Notes |
| --- | --- | --- | --- |
| `hashing` | Each feature (stemmed unigram weight `1+log tf`, bigram at half weight) is hashed with keyed BLAKE2b to 6 signed positions; the sum is L2-normalised | none | Implicit Achlioptas-style sparse random projection of an unbounded hashed bag of words; deterministic per seed |
| `lsa` | TF-IDF rows (sublinear tf, smoothed idf, L2) in CSR form -> randomized truncated SVD (Halko et al., 4 power iterations) -> project onto the top `dim` right singular vectors | `reindex` | Captures co-occurrence ("latent topics"); refit when the corpus changes substantially |
| `sentence-transformers` | Optional adapter around an installed model | none | Not installed by default |

All vectors are unit length, so a dot product equals cosine similarity.

## Vector indexes

- **Flat**: one matrix-vector product over live rows, `argpartition` for top-k. Exact.
- **IVF**: spherical k-means (k-means++ seeding, empty-cluster re-seeding) trained once
  `ivf_train_threshold` vectors exist and retrained when the collection doubles. Queries visit the
  `nprobe` nearest partitions, then keep probing while a metadata filter leaves fewer than `3k`
  candidates; if still fewer than `k`, the query is answered exactly.
- **HNSW**: levels drawn with `floor(-ln U / ln M)`; greedy descent through upper layers, beam
  search with `ef_construction` on insertion, neighbour selection with the diversity heuristic
  (a candidate is kept only if it is closer to the new node than to any already selected
  neighbour, then pruned candidates fill remaining slots); `M` links per upper layer, `2M` on layer
  0, with re-pruning when a neighbour list overflows. Queries use `ef_search` (at least `k`, and
  `4k` with a filter); if fewer than `k` filtered hits survive, an exact scan is used.

Measured on the development machine (pure Python): HNSW recall@10 was 1.0 against the exact index
on 2,000 clustered 64-dimensional vectors, with inserts taking about 2 ms each (`ef_construction=100`); the unit tests
assert recall >= 0.9 on smaller synthetic sets. These are indicative numbers, not benchmarks.

## BM25

Okapi BM25 with `k1 = 1.5`, `b = 0.75` and `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` (always
positive). Query term frequency multiplies the term contribution. Metadata filters are applied to
the scored candidates.

## Hybrid search

The vector and BM25 rankings (each `top_k * candidate_multiplier` long, x3 when grouping by
document) are fused with reciprocal rank fusion: `score(d) = sum 1 / (rrf_k + rank_r(d))`,
`rrf_k = 60` by default. Ties break on the best individual rank, then id. Snippets select the
40-word window containing the most query-term hits and return highlight offsets relative to the
snippet.

## Filters

Filters are compiled once into a predicate (see API.md for the operator list). The same predicate
runs inside the vector index (during candidate selection) and in BM25 (on scored candidates), so
both sides of a hybrid query see the same subset.

## RAG pipeline

1. **Retrieve**: hybrid search for `rag.retrieve_k` chunks (filters apply).
2. **Rerank**: each candidate gets four features in [0, 1]: min-max normalised retrieval score,
   IDF-weighted query-term coverage, term proximity (smallest window containing every matched term)
   and embedding cosine. Weighted sum: 0.35 / 0.30 / 0.10 / 0.25.
3. **Diversify**: Maximal Marginal Relevance with `rag.mmr_lambda` (default 0.7) selects
   `rag.rerank_k` chunks, penalising similarity to already selected ones (embedding cosine, or term
   Jaccard when no embedder is available). This matters because overlapping chunks repeat text.
4. **Filter**: chunks with coverage below `rag.min_relevance` and semantic similarity below 0.3 are
   dropped; if none remain the pipeline answers "not found" without calling the LLM.
5. **Assemble**: passages are numbered `[1]..[n]` in rank order until `rag.context_token_budget`;
   the last passage may be cut at a sentence boundary. Each passage has a source label
   (`title, pp. x-y, section path`).
6. **Generate**: the adapter receives the question and passages. HTTP adapters render a system
   prompt that requires `[n]` citations and an exact "not found" sentence; the extractive adapter
   works on the structured passages directly.
7. **Cite**: `[n]` markers that refer to supplied passages become citation records; if an answer
   has no valid markers, all context passages are returned as citations so the answer can be
   verified.

### Extractive answerer

Every context sentence is scored as

```
IDF-weighted question-term overlap
+ 0.10 per matching question bigram
+ 0.25 if the sentence contains the expected answer type and matches the question
+ 0.08 / (1 + passage rank)
- small penalty for very long sentences
```

The expected answer type is inferred from the question ("how many/how long" -> number, "who" ->
person/organisation, "when" -> date or a relative duration such as "within 45 days", "where" /
"which law" -> location, "how much/amount/rent/fee" -> money, e-mail, phone, URL, percent). The best
sentence must reach an overlap of 0.2, otherwise the adapter abstains. Up to `llm.answer_sentences`
sentences scoring at least 60% of the best are returned in document order, each followed by its
citation. The `short_answer` is the typed value in the best sentence closest to the question words.

## Summarisation

- **TextRank**: sentences with at least three content terms become nodes; edges are TF-IDF cosine
  similarities; PageRank (damping 0.85, power iteration with dangling-node handling) ranks them;
  the top sentences are returned in document order. Document summaries exclude headings.
- **Map-reduce**: each chunk is summarised through the LLM adapter (map, 2 sentences), partial
  summaries are grouped under a token budget and summarised again while they exceed it (collapse,
  at most 4 levels), and a final reduce call produces the summary. Stage statistics are returned.
