"""Vector indexes: exact (flat), IVF and HNSW, with metadata filtering."""

from __future__ import annotations

from idengine.config import IndexSettings
from idengine.vectorindex.base import VectorHit, VectorIndex
from idengine.vectorindex.filters import Predicate, compile_filter
from idengine.vectorindex.flat import FlatIndex
from idengine.vectorindex.hnsw import HNSWIndex
from idengine.vectorindex.ivf import IVFIndex


def create_index(settings: IndexSettings, dim: int, *, seed: int = 0) -> VectorIndex:
    if settings.kind == "flat":
        return FlatIndex(dim)
    if settings.kind == "ivf":
        return IVFIndex(
            dim,
            nlist=settings.ivf_nlist,
            nprobe=settings.ivf_nprobe,
            train_threshold=settings.ivf_train_threshold,
            seed=seed,
        )
    return HNSWIndex(
        dim,
        m=settings.hnsw_m,
        ef_construction=settings.hnsw_ef_construction,
        ef_search=settings.hnsw_ef_search,
        seed=seed,
    )


__all__ = [
    "FlatIndex",
    "HNSWIndex",
    "IVFIndex",
    "Predicate",
    "VectorHit",
    "VectorIndex",
    "compile_filter",
    "create_index",
]
