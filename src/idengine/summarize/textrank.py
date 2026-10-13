"""TextRank extractive summarisation (Mihalcea & Tarau, 2004).

Sentences are graph nodes; edges are weighted by TF-IDF cosine similarity;
PageRank (power iteration) scores centrality. The summary keeps the top
sentences in their original order.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from idengine.text.tokenize import analyze, split_sentences

_MIN_SENTENCE_TERMS = 3


@dataclass(frozen=True, slots=True)
class RankedSentence:
    index: int
    text: str
    score: float


def pagerank(
    weights: np.ndarray, *, damping: float = 0.85, iterations: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """PageRank over a weighted adjacency matrix (rows normalised to transitions)."""
    n = weights.shape[0]
    if n == 0:
        return np.zeros(0)
    out_strength = weights.sum(axis=1, keepdims=True)
    # Dangling nodes (no edges) jump uniformly.
    transition = np.where(
        out_strength > 0, weights / np.where(out_strength == 0, 1, out_strength), 1.0 / n
    )
    scores = np.full(n, 1.0 / n)
    for _ in range(iterations):
        updated = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(updated - scores).sum() < tol:
            return updated
        scores = updated
    return scores


def similarity_matrix(sentence_terms: Sequence[list[str]]) -> np.ndarray:
    vocabulary: dict[str, int] = {}
    for terms in sentence_terms:
        for term in terms:
            vocabulary.setdefault(term, len(vocabulary))
    n = len(sentence_terms)
    if not vocabulary:
        return np.zeros((n, n))
    df = Counter(term for terms in sentence_terms for term in set(terms))
    matrix = np.zeros((n, len(vocabulary)))
    for row, terms in enumerate(sentence_terms):
        for term, tf in Counter(terms).items():
            matrix[row, vocabulary[term]] = (1 + math.log(tf)) * math.log(
                (1 + n) / (1 + df[term]) + 1
            )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, 0.0)
    return sim


def rank_sentences(sentences: Sequence[str]) -> list[RankedSentence]:
    """Score sentences by TextRank centrality; sentences that are too short score 0."""
    terms = [analyze(s) for s in sentences]
    eligible = [i for i, t in enumerate(terms) if len(t) >= _MIN_SENTENCE_TERMS]
    if not eligible:
        return [RankedSentence(i, s, 0.0) for i, s in enumerate(sentences)]
    sim = similarity_matrix([terms[i] for i in eligible])
    scores = pagerank(sim)
    score_of = {idx: float(scores[pos]) for pos, idx in enumerate(eligible)}
    return [RankedSentence(i, s, round(score_of.get(i, 0.0), 6)) for i, s in enumerate(sentences)]


def textrank_summary(text: str | Sequence[str], *, max_sentences: int = 5) -> list[RankedSentence]:
    """Top ``max_sentences`` sentences in document order."""
    sentences = [s.text for s in split_sentences(text)] if isinstance(text, str) else list(text)
    if not sentences:
        return []
    ranked = rank_sentences(sentences)
    # Mild lead bias: earlier sentences break ties (documents front-load key facts).
    best = sorted(ranked, key=lambda r: (-(r.score + 1e-6 / (1 + r.index)), r.index))
    return sorted(best[:max_sentences], key=lambda r: r.index)
