"""Spherical k-means (cosine) with k-means++ seeding, used to train IVF centroids."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from idengine.embeddings.base import l2_normalize

Array = npt.NDArray[np.float32]


def kmeans_plus_plus(data: Array, k: int, rng: np.random.Generator) -> Array:
    """Choose ``k`` initial centroids spread out by cosine distance."""
    n = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centroids[0] = data[first]
    closest = 1.0 - data @ centroids[0]
    for i in range(1, k):
        weights = np.clip(closest, 0.0, None) ** 2
        total = float(weights.sum())
        index = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=weights / total))
        centroids[i] = data[index]
        closest = np.minimum(closest, 1.0 - data @ centroids[i])
    return centroids


def spherical_kmeans(
    data: Array, k: int, *, iterations: int = 25, seed: int = 0, tol: float = 1e-4
) -> tuple[Array, npt.NDArray[np.int64]]:
    """Cluster unit vectors; returns ``(centroids, assignments)``.

    Empty clusters are re-seeded with the point that is currently worst
    served by its centroid, so exactly ``k`` non-empty clusters are produced
    whenever ``n >= k``.
    """
    n = data.shape[0]
    if n == 0:
        raise ValueError("cannot cluster an empty dataset")
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus(data, k, rng)
    assignments = np.zeros(n, dtype=np.int64)
    previous_objective = -np.inf
    for _ in range(iterations):
        similarity = data @ centroids.T
        assignments = np.argmax(similarity, axis=1)
        best = similarity[np.arange(n), assignments]
        objective = float(best.sum())
        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, assignments, data)
        counts = np.bincount(assignments, minlength=k)
        for empty in np.flatnonzero(counts == 0):
            worst = int(np.argmin(best))
            new_centroids[empty] = data[worst]
            best[worst] = np.inf
        centroids = l2_normalize(new_centroids)
        if objective - previous_objective < tol * max(abs(objective), 1.0):
            break
        previous_objective = objective
    assignments = np.argmax(data @ centroids.T, axis=1)
    return centroids, assignments
