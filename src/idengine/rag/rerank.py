"""Second-stage reranking of retrieved chunks.

Features per candidate (all in [0, 1]):

* ``retrieval`` - min-max normalised first-stage (fused) score,
* ``coverage``  - IDF-weighted share of query terms present in the chunk,
* ``proximity`` - how tightly the matched query terms cluster in the chunk,
* ``semantic``  - cosine similarity between query and chunk embeddings.

The weighted sum is then diversified with Maximal Marginal Relevance so the
context window is not filled with near-identical overlapping chunks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from idengine.embeddings.base import EmbeddingProvider
from idengine.search.fusion import normalize_scores
from idengine.search.service import SearchHit
from idengine.text.tokenize import analyze

DEFAULT_WEIGHTS = {"retrieval": 0.35, "coverage": 0.3, "proximity": 0.1, "semantic": 0.25}


@dataclass(slots=True)
class RankedChunk:
    hit: SearchHit
    score: float
    features: dict[str, float] = field(default_factory=dict)


def proximity_score(chunk_terms: Sequence[str], query_terms: set[str]) -> float:
    """1.0 when all matched query terms are adjacent, decaying with the span width."""
    positions: dict[str, list[int]] = {}
    for i, term in enumerate(chunk_terms):
        if term in query_terms:
            positions.setdefault(term, []).append(i)
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    # Smallest window covering one occurrence of each matched term (sliding window).
    events = sorted((pos, term) for term, plist in positions.items() for pos in plist)
    need = len(positions)
    counts: dict[str, int] = {}
    best = math.inf
    left = 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == need:
            best = min(best, pos - events[left][0] + 1)
            left_term = events[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                del counts[left_term]
            left += 1
    return need / best if best != math.inf else 0.0


class Reranker:
    def __init__(
        self,
        embedder: EmbeddingProvider | None,
        *,
        weights: dict[str, float] | None = None,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.weights = weights or DEFAULT_WEIGHTS
        self.mmr_lambda = mmr_lambda

    def score(self, query: str, hits: Sequence[SearchHit]) -> list[RankedChunk]:
        return self._score(query, hits)[0]

    def _score(
        self, query: str, hits: Sequence[SearchHit]
    ) -> tuple[list[RankedChunk], np.ndarray | None]:
        if not hits:
            return [], None
        query_terms = set(analyze(query))
        retrieval = normalize_scores({h.chunk_id: h.score for h in hits})
        chunk_terms = [analyze(h.text) for h in hits]
        n = len(hits)
        df = {t: sum(1 for terms in chunk_terms if t in set(terms)) for t in query_terms}
        idf = {t: math.log(1 + n / (1 + df[t])) + 0.1 for t in query_terms}
        total_idf = sum(idf.values()) or 1.0

        semantic = [0.0] * n
        embeddings: np.ndarray | None = None
        if self.embedder is not None and self.embedder.is_fitted:
            embeddings = self.embedder.embed([h.text for h in hits])
            query_vector = self.embedder.embed_query(query)
            semantic = [max(0.0, float(v)) for v in embeddings @ query_vector]

        ranked: list[RankedChunk] = []
        for i, hit in enumerate(hits):
            present = query_terms & set(chunk_terms[i])
            features = {
                "retrieval": retrieval[hit.chunk_id],
                "coverage": sum(idf[t] for t in present) / total_idf if query_terms else 0.0,
                "proximity": proximity_score(chunk_terms[i], query_terms),
                "semantic": semantic[i],
            }
            score = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
            ranked.append(
                RankedChunk(hit, round(score, 6), {k: round(v, 4) for k, v in features.items()})
            )
        return ranked, embeddings

    def select(self, query: str, hits: Sequence[SearchHit], k: int) -> list[RankedChunk]:
        """Score candidates and pick ``k`` with MMR diversification."""
        ranked, embeddings = self._score(query, hits)
        if len(ranked) <= 1:
            return ranked
        similarity = self._pairwise_similarity(ranked, embeddings)
        remaining = list(range(len(ranked)))
        chosen: list[int] = []
        while remaining and len(chosen) < k:

            def mmr(i: int) -> float:
                redundancy = max((similarity[i][j] for j in chosen), default=0.0)
                return self.mmr_lambda * ranked[i].score - (1 - self.mmr_lambda) * redundancy

            best = max(remaining, key=lambda i: (mmr(i), -i))
            chosen.append(best)
            remaining.remove(best)
        return [ranked[i] for i in chosen]

    @staticmethod
    def _pairwise_similarity(
        ranked: Sequence[RankedChunk], embeddings: np.ndarray | None
    ) -> list[list[float]]:
        if embeddings is not None and len(embeddings) == len(ranked):
            return (embeddings @ embeddings.T).tolist()
        sets = [set(analyze(r.hit.text)) for r in ranked]
        return [[len(a & b) / len(a | b) if a | b else 0.0 for b in sets] for a in sets]
