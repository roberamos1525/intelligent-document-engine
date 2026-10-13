"""Exact brute-force cosine index."""

from __future__ import annotations

from idengine.vectorindex.base import Vector, VectorHit, VectorIndex
from idengine.vectorindex.filters import Predicate


class FlatIndex(VectorIndex):
    """Scores every live vector with one matrix-vector product.

    Exact and fast up to tens of thousands of vectors; also the ground truth
    used to measure the recall of the approximate indexes.
    """

    kind = "flat"

    def _search(self, query: Vector, k: int, predicate: Predicate | None) -> list[VectorHit]:
        return self.exact_search(query, k, predicate)
