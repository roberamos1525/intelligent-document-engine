"""Inverted-file (IVF) approximate index.

Vectors are partitioned by their nearest k-means centroid. A query scores
only the ``nprobe`` closest partitions. Until enough vectors exist to train
meaningful centroids the index answers exactly; it retrains automatically
when the collection has doubled since the last training.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from idengine.vectorindex.base import Vector, VectorHit, VectorIndex, top_k
from idengine.vectorindex.filters import Predicate
from idengine.vectorindex.kmeans import spherical_kmeans


class IVFIndex(VectorIndex):
    kind = "ivf"

    def __init__(
        self,
        dim: int,
        *,
        nlist: int = 16,
        nprobe: int = 4,
        train_threshold: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(dim)
        if nprobe > nlist:
            raise ValueError("nprobe must not exceed nlist")
        self.nlist = nlist
        self.nprobe = nprobe
        self.train_threshold = max(train_threshold, nlist)
        self.seed = seed
        self.centroids: npt.NDArray[np.float32] | None = None
        self.lists: list[list[int]] = []
        self.trained_size = 0

    @property
    def trained(self) -> bool:
        return self.centroids is not None

    def train(self) -> None:
        """(Re)train centroids on all live vectors and rebuild the inverted lists."""
        with self.lock:
            rows = self.storage.live_rows()
            if rows.size < max(self.nlist, 2):
                self.centroids, self.lists, self.trained_size = None, [], 0
                return
            centroids, assignments = spherical_kmeans(
                self.storage.vectors[rows], self.nlist, seed=self.seed
            )
            self.centroids = centroids
            self.lists = [[] for _ in range(centroids.shape[0])]
            for row, cluster in zip(rows.tolist(), assignments.tolist(), strict=True):
                self.lists[cluster].append(row)
            self.trained_size = int(rows.size)

    def _on_add(self, row: int) -> None:
        if self.centroids is None:
            if self.storage.live_count >= self.train_threshold:
                self.train()
            return
        if self.storage.live_count >= 2 * self.trained_size:
            self.train()
            return
        cluster = int(np.argmax(self.centroids @ self.storage.vectors[row]))
        self.lists[cluster].append(row)

    def _rebuild_structure(self) -> None:
        if self.storage.live_count >= self.train_threshold:
            self.train()
        else:
            self.centroids, self.lists, self.trained_size = None, [], 0

    def _search(self, query: Vector, k: int, predicate: Predicate | None) -> list[VectorHit]:
        if self.centroids is None:
            return self.exact_search(query, k, predicate)
        order = np.argsort(-(self.centroids @ query))
        # Adaptive probing: visit at least `nprobe` partitions, and keep going while a
        # filter leaves too few candidates (filtered queries need wider coverage).
        wanted = k * (3 if predicate is not None else 1)
        candidates: list[int] = []
        for probed, cluster in enumerate(order, start=1):
            for row in self.lists[cluster]:
                if self.storage.alive[row] and (
                    predicate is None or predicate(self.storage.metadata[row])
                ):
                    candidates.append(row)
            if probed >= self.nprobe and len(candidates) >= wanted:
                break
        if len(candidates) < k:
            # Selective filters or tiny partitions: fall back to an exact answer.
            return self.exact_search(query, k, predicate)
        rows = np.asarray(candidates, dtype=np.int64)
        scores = self.storage.vectors[rows] @ query
        return [self.storage.hit(int(rows[i]), scores[i]) for i in top_k(rows, scores, k)]

    def stats(self) -> dict[str, Any]:
        stats = super().stats()
        sizes = [len(lst) for lst in self.lists]
        stats.update(
            {
                "trained": self.trained,
                "nlist": len(self.lists) if self.trained else 0,
                "nprobe": self.nprobe,
                "largest_list": max(sizes) if sizes else 0,
            }
        )
        return stats

    def _extra_arrays(self) -> dict[str, npt.NDArray[Any]]:
        if self.centroids is None:
            return {}
        return {"centroids": self.centroids}

    def _extra_manifest(self) -> dict[str, Any]:
        return {"lists": self.lists, "trained_size": self.trained_size}

    def _restore_extra(self, manifest: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        if "centroids" in arrays and manifest.get("lists") is not None:
            self.centroids = np.asarray(arrays["centroids"], dtype=np.float32)
            self.lists = [list(map(int, lst)) for lst in manifest["lists"]]
            self.trained_size = int(manifest.get("trained_size", 0))
        else:
            self._rebuild_structure()
