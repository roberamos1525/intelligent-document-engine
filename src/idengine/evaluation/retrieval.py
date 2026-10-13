"""Ranking metrics for retrieval evaluation (binary or graded relevance)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len([r for r in ranked[:k] if r in relevant]) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    return sum(
        (2.0 ** gains.get(item, 0.0) - 1.0) / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
    )


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalised DCG with exponential gain; ``gains`` maps relevant ids to grades."""
    ideal_order = sorted(gains, key=lambda item: gains[item], reverse=True)
    ideal = dcg_at_k(ideal_order, gains, k)
    return 0.0 if ideal == 0 else dcg_at_k(ranked, gains, k) / ideal


def dedupe(ranked: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each id (chunk hits -> document ranking)."""
    return list(dict.fromkeys(ranked))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
