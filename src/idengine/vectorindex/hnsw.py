"""Hierarchical Navigable Small World graph index (Malkov & Yashunin, 2018).

Implementation notes:

* Similarity is the dot product of unit vectors (cosine); larger is closer.
* Neighbour selection uses the paper's heuristic (Algorithm 4), which keeps
  the graph navigable across clusters better than "closest M".
* Removal marks a tombstone: the node keeps routing traffic but is never
  returned. :meth:`rebuild` compacts tombstones.
* Filtered queries widen the beam; if the filtered beam still yields fewer
  than ``k`` hits, the index falls back to an exact scan of matching rows.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from idengine.vectorindex.base import Vector, VectorHit, VectorIndex
from idengine.vectorindex.filters import Predicate


class HNSWIndex(VectorIndex):
    kind = "hnsw"

    def __init__(
        self,
        dim: int,
        *,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__(dim)
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = max(ef_construction, m)
        self.ef_search = ef_search
        self.level_multiplier = 1.0 / math.log(max(m, 2))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.levels: list[int] = []
        self.neighbors: list[list[list[int]]] = []  # node -> level -> neighbour rows
        self.entry_point: int | None = None
        self.max_level = -1

    # -- primitives ------------------------------------------------------------
    def _similarities(self, query: Vector, rows: list[int]) -> list[float]:
        return (self.storage.vectors[rows] @ query).tolist()

    def _random_level(self) -> int:
        return int(-math.log(max(self._rng.random(), 1e-12)) * self.level_multiplier)

    def _search_layer(
        self, query: Vector, entry_points: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """Beam search on one layer; returns up to ``ef`` (similarity, row), best first."""
        visited = set(entry_points)
        sims = self._similarities(query, entry_points)
        candidates = [(-s, r) for s, r in zip(sims, entry_points, strict=True)]
        heapq.heapify(candidates)
        results = [(s, r) for s, r in zip(sims, entry_points, strict=True)]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        while candidates:
            neg_sim, row = heapq.heappop(candidates)
            if -neg_sim < results[0][0] and len(results) >= ef:
                break
            fresh = [n for n in self.neighbors[row][level] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for sim, neighbor in zip(self._similarities(query, fresh), fresh, strict=True):
                if len(results) < ef or sim > results[0][0]:
                    heapq.heappush(candidates, (-sim, neighbor))
                    heapq.heappush(results, (sim, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted(results, reverse=True)

    def _select_neighbors(self, candidates: list[tuple[float, int]], limit: int) -> list[int]:
        """Heuristic selection: keep a candidate only if it is closer to the base
        node than to every neighbour already selected (diversifies directions)."""
        ordered = sorted(candidates, reverse=True)
        if len(ordered) <= 1:
            return [row for _, row in ordered]
        rows = [row for _, row in ordered]
        block = self.storage.vectors[rows]
        pairwise = block @ block.T  # one product instead of one per candidate
        chosen: list[int] = []  # positions into `rows`
        discarded: list[int] = []
        # closest_chosen[p] = similarity of candidate p to its nearest selected neighbour.
        closest_chosen = np.full(len(rows), -np.inf, dtype=np.float32)
        for pos, (sim, _) in enumerate(ordered):
            if len(chosen) >= limit:
                break
            if closest_chosen[pos] > sim:
                discarded.append(pos)
                continue
            chosen.append(pos)
            np.maximum(closest_chosen, pairwise[pos], out=closest_chosen)
        # Fill remaining slots with the best discarded candidates (keepPrunedConnections).
        for pos in discarded:
            if len(chosen) >= limit:
                break
            chosen.append(pos)
        return [rows[pos] for pos in chosen]

    def _shrink(self, row: int, level: int) -> None:
        limit = self.m0 if level == 0 else self.m
        links = self.neighbors[row][level]
        if len(links) <= limit:
            return
        base = self.storage.vectors[row]
        scored = list(zip(self._similarities(base, links), links, strict=True))
        self.neighbors[row][level] = self._select_neighbors(scored, limit)

    # -- incremental construction -----------------------------------------------
    def _on_add(self, row: int) -> None:
        level = self._random_level()
        while len(self.levels) <= row:
            self.levels.append(-1)
            self.neighbors.append([])
        self.levels[row] = level
        self.neighbors[row] = [[] for _ in range(level + 1)]
        query = self.storage.vectors[row]

        if self.entry_point is None:
            self.entry_point, self.max_level = row, level
            return

        entry = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            entry = [self._search_layer(query, entry, 1, layer)[0][1]]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(query, entry, self.ef_construction, layer)
            limit = self.m0 if layer == 0 else self.m
            chosen = self._select_neighbors(found, limit)
            self.neighbors[row][layer] = chosen
            for neighbor in chosen:
                self.neighbors[neighbor][layer].append(row)
                self._shrink(neighbor, layer)
            entry = [r for _, r in found]
        if level > self.max_level:
            self.entry_point, self.max_level = row, level

    def _rebuild_structure(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.levels, self.neighbors = [], []
        self.entry_point, self.max_level = None, -1
        for row in range(self.storage.size):
            if self.storage.alive[row]:
                self._on_add(row)
            else:
                self.levels.append(-1)
                self.neighbors.append([])

    # -- query ---------------------------------------------------------------
    def _search(self, query: Vector, k: int, predicate: Predicate | None) -> list[VectorHit]:
        if self.entry_point is None:
            return []
        entry = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            entry = [self._search_layer(query, entry, 1, layer)[0][1]]
        ef = max(self.ef_search, k)
        if predicate is not None:
            ef = max(ef, 4 * k)
        found = self._search_layer(query, entry, ef, 0)
        hits: list[VectorHit] = []
        for sim, row in found:
            if not self.storage.alive[row]:
                continue
            if predicate is not None and not predicate(self.storage.metadata[row]):
                continue
            hits.append(self.storage.hit(row, sim))
            if len(hits) == k:
                break
        if len(hits) < k and (predicate is not None or len(hits) < len(self)):
            exact = self.exact_search(query, k, predicate)
            if len(exact) > len(hits):
                return exact
        return hits

    def stats(self) -> dict[str, Any]:
        stats = super().stats()
        alive_levels = [lvl for row, lvl in enumerate(self.levels) if self.storage.alive[row]]
        degrees = [
            len(self.neighbors[r][0]) for r in range(len(self.levels)) if self.levels[r] >= 0
        ]
        stats.update(
            {
                "m": self.m,
                "ef_search": self.ef_search,
                "max_level": self.max_level,
                "levels_histogram": {
                    str(lvl): alive_levels.count(lvl) for lvl in sorted(set(alive_levels))
                },
                "mean_degree_level0": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
            }
        )
        return stats

    # -- persistence -----------------------------------------------------------
    def _extra_manifest(self) -> dict[str, Any]:
        return {
            "levels": self.levels,
            "neighbors": self.neighbors,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }

    def _restore_extra(self, manifest: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        levels = manifest.get("levels")
        neighbors = manifest.get("neighbors")
        if (
            isinstance(levels, list)
            and isinstance(neighbors, list)
            and len(levels) == len(neighbors) == self.storage.size
        ):
            self.levels = [int(v) for v in levels]
            self.neighbors = [[list(map(int, layer)) for layer in node] for node in neighbors]
            entry = manifest.get("entry_point")
            self.entry_point = int(entry) if entry is not None else None
            self.max_level = int(manifest.get("max_level", -1))
        else:
            self._rebuild_structure()
