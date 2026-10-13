"""Rank fusion for hybrid search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists with Reciprocal Rank Fusion (Cormack et al., 2009).

    ``score(d) = sum_r w_r / (k + rank_r(d))`` with 1-based ranks. Returns
    ``(id, fused_score, {ranking_name: rank})`` sorted best first; ties are
    broken by the best individual rank, then by id for determinism.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for position, item_id in enumerate(ids, start=1):
            if name in ranks.get(item_id, {}):
                continue  # duplicate inside one ranking: keep the best position
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + position)
            ranks.setdefault(item_id, {})[name] = position
    return sorted(
        ((item_id, score, ranks[item_id]) for item_id, score in scores.items()),
        key=lambda item: (-item[1], min(item[2].values()), item[0]),
    )


def normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalise scores into [0, 1] (all-equal inputs map to 1)."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {key: (value - low) / (high - low) for key, value in scores.items()}
