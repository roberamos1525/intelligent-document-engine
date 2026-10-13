"""Semantic, lexical and hybrid search over indexed chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from idengine.config import SearchSettings
from idengine.errors import ModelNotTrainedError, ValidationError
from idengine.observability.metrics import EngineMetrics
from idengine.search.fusion import reciprocal_rank_fusion
from idengine.search.highlight import make_snippet
from idengine.search.indexes import IndexManager
from idengine.storage.repositories import ChunkRepository
from idengine.text.tokenize import analyze
from idengine.vectorindex.filters import compile_filter


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    title: str | None
    filename: str | None
    text: str
    snippet: str
    highlights: list[tuple[int, int]]
    score: float
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    vector_score: float | None = None
    bm25_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        parts = [self.title or self.filename or self.document_id]
        if self.page_start is not None:
            pages = (
                f"p. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"pp. {self.page_start}-{self.page_end}"
            )
            parts.append(pages)
        path = list(self.heading_path)
        if path and self.title and path[0].casefold() == self.title.casefold():
            path = path[1:]  # the top heading usually repeats the title
        if path:
            parts.append(" > ".join(path))
        return ", ".join(parts)


@dataclass(slots=True)
class SearchResponse:
    query: str
    mode: SearchMode
    hits: list[SearchHit]
    took_ms: float
    warnings: list[str] = field(default_factory=list)


class SearchService:
    def __init__(
        self,
        settings: SearchSettings,
        indexes: IndexManager,
        chunks: ChunkRepository,
        metrics: EngineMetrics,
    ) -> None:
        self.settings = settings
        self.indexes = indexes
        self.chunks = chunks
        self.metrics = metrics

    def _validate(self, query: str, top_k: int) -> str:
        query = query.strip()
        if not query:
            raise ValidationError("query must not be empty")
        if len(query) > self.settings.max_query_chars:
            raise ValidationError(
                f"query exceeds {self.settings.max_query_chars} characters",
                details={"limit": self.settings.max_query_chars},
            )
        if not 1 <= top_k <= self.settings.max_top_k:
            raise ValidationError(
                f"top_k must be between 1 and {self.settings.max_top_k}",
                details={"top_k": top_k},
            )
        return query

    def search(
        self,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.HYBRID,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        group_by_document: bool = False,
        min_score: float | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        mode = SearchMode(mode)
        top_k = self.settings.default_top_k if top_k is None else top_k
        query = self._validate(query, top_k)
        predicate = compile_filter(filters)
        warnings: list[str] = []
        candidates = top_k * self.settings.candidate_multiplier * (3 if group_by_document else 1)
        terms = analyze(query)

        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        rankings: dict[str, list[str]] = {}

        if mode in (SearchMode.SEMANTIC, SearchMode.HYBRID):
            if not self.indexes.vectors_available:
                if mode is SearchMode.SEMANTIC:
                    raise ModelNotTrainedError(
                        "semantic search is unavailable until the embedder is fitted (reindex)"
                    )
                warnings.append("vector index unavailable; hybrid search used BM25 only")
            else:
                query_vector = self.indexes.embedder.embed_query(query)
                vector_hits = self.indexes.vectors.search(query_vector, candidates, predicate)
                vector_scores = {h.id: h.score for h in vector_hits if h.score > 0}
                rankings["vector"] = list(vector_scores)
        if mode in (SearchMode.LEXICAL, SearchMode.HYBRID):
            if not terms:
                warnings.append("query has no searchable terms after stopword removal")
            lexical_hits = self.indexes.bm25.search(terms, candidates, predicate)
            bm25_scores = {h.id: h.score for h in lexical_hits}
            rankings["bm25"] = list(bm25_scores)

        if mode is SearchMode.HYBRID:
            fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)
            ordered = [(item_id, score, ranks) for item_id, score, ranks in fused]
        elif mode is SearchMode.SEMANTIC:
            ordered = [(i, s, {"vector": r}) for r, (i, s) in enumerate(vector_scores.items(), 1)]
        else:
            ordered = [(i, s, {"bm25": r}) for r, (i, s) in enumerate(bm25_scores.items(), 1)]

        if min_score is not None:
            ordered = [item for item in ordered if item[1] >= min_score]

        hits = self._hydrate(ordered, set(terms), vector_scores, bm25_scores)
        if group_by_document:
            seen: set[str] = set()
            grouped: list[SearchHit] = []
            for hit in hits:
                if hit.document_id not in seen:
                    seen.add(hit.document_id)
                    grouped.append(hit)
            hits = grouped
        hits = hits[:top_k]

        elapsed = time.perf_counter() - started
        self.metrics.search_latency.observe(elapsed, {"mode": mode.value})
        return SearchResponse(query, mode, hits, round(elapsed * 1000, 2), warnings)

    def _hydrate(
        self,
        ordered: list[tuple[str, float, dict[str, int]]],
        query_terms: set[str],
        vector_scores: dict[str, float],
        bm25_scores: dict[str, float],
    ) -> list[SearchHit]:
        records = self.chunks.get_many([item_id for item_id, _, _ in ordered])
        hits: list[SearchHit] = []
        for chunk_id, score, ranks in ordered:
            record = records.get(chunk_id)
            if record is None:  # deleted concurrently
                continue
            meta = dict(self.indexes.bm25.metadata.get(chunk_id, {}))
            snippet = make_snippet(record.text, query_terms)
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    document_id=record.document_id,
                    title=meta.get("title"),
                    filename=meta.get("filename"),
                    text=record.text,
                    snippet=snippet.text,
                    highlights=snippet.highlights,
                    score=round(float(score), 6),
                    heading_path=record.heading_path,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    vector_score=round(vector_scores[chunk_id], 6)
                    if chunk_id in vector_scores
                    else None,
                    bm25_score=round(bm25_scores[chunk_id], 6) if chunk_id in bm25_scores else None,
                    ranks=ranks,
                    metadata={k: meta.get(k) for k in ("category", "language", "doc_type", "tags")},
                )
            )
        return hits
