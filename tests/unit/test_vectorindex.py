from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from idengine.config import IndexSettings
from idengine.embeddings import l2_normalize
from idengine.errors import IndexCorruptedError, ValidationError
from idengine.vectorindex import FlatIndex, HNSWIndex, IVFIndex, compile_filter, create_index
from idengine.vectorindex.base import VectorIndex
from idengine.vectorindex.kmeans import spherical_kmeans

DIM = 32


def clustered(n_clusters: int = 10, per_cluster: int = 40, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, DIM))
    points = np.vstack([c + 0.5 * rng.standard_normal((per_cluster, DIM)) for c in centers])
    return l2_normalize(points.astype(np.float32))


def fill(index: VectorIndex, vectors: np.ndarray) -> list[str]:
    ids = [f"v{i}" for i in range(len(vectors))]
    meta = [
        {"parity": "even" if i % 2 == 0 else "odd", "n": i, "tags": [f"t{i % 3}"]}
        for i in range(len(vectors))
    ]
    index.add(ids, vectors, meta)
    return ids


def recall(
    index: VectorIndex, truth: FlatIndex, queries: np.ndarray, k: int = 10, flt=None
) -> float:  # type: ignore[no-untyped-def]
    scores = []
    for q in queries:
        expected = {h.id for h in truth.search(q, k, flt)}
        got = {h.id for h in index.search(q, k, flt)}
        scores.append(len(expected & got) / max(len(expected), 1))
    return float(np.mean(scores))


class TestFilters:
    META = {
        "category": "finance",
        "date": "2024-05-01",
        "tags": ["urgent", "q2"],
        "n": 5,
        "title": "Big Report",
    }

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ({"category": "finance"}, True),
            ({"category": "legal"}, False),
            ({"tags": "urgent"}, True),
            ({"category": {"$in": ["legal", "finance"]}}, True),
            ({"category": {"$nin": ["finance"]}}, False),
            ({"category": {"$ne": "legal"}}, True),
            ({"date": {"$gte": "2024-01-01", "$lt": "2025-01-01"}}, True),
            ({"n": {"$gt": 5}}, False),
            ({"n": {"$lte": 5}}, True),
            ({"n": {"$gt": "x"}}, False),
            ({"missing": {"$exists": False}}, True),
            ({"title": {"$exists": True}}, True),
            ({"title": {"$contains": "report"}}, True),
            ({"tags": {"$contains": "URG"}}, True),
            ({"$or": [{"category": "legal"}, {"n": 5}]}, True),
            ({"$and": [{"category": "finance"}, {"n": 6}]}, False),
            ({"$not": {"category": "legal"}}, True),
            ({"missing": {"$gt": 1}}, False),
        ],
    )
    def test_operators(self, spec: dict, expected: bool) -> None:  # type: ignore[type-arg]
        predicate = compile_filter(spec)
        assert predicate is not None
        assert predicate(self.META) is expected

    @pytest.mark.parametrize(
        "spec",
        [
            {"$xor": []},
            {"category": {"$regex": ".*"}},
            {"Bad-Field": 1},
            {"$or": []},
            {"category": {"$in": "finance"}},
            {"n": {"$gt": None}},
            {"n": {"$exists": "yes"}},
            {"category": {"$eq": ["a"]}},
            {"title": {"$contains": 3}},
            "not-an-object",
        ],
    )
    def test_invalid_filters(self, spec: object) -> None:
        with pytest.raises(ValidationError):
            compile_filter(spec)  # type: ignore[arg-type]

    def test_depth_limit_and_empty(self) -> None:
        spec: dict = {"category": "x"}  # type: ignore[type-arg]
        for _ in range(10):
            spec = {"$not": spec}
        with pytest.raises(ValidationError, match="deeply"):
            compile_filter(spec)
        assert compile_filter(None) is None and compile_filter({}) is None


class TestIndexes:
    def test_flat_is_exact_and_filters(self) -> None:
        vectors = clustered()
        index = FlatIndex(DIM)
        fill(index, vectors)
        hits = index.search(vectors[7], 3)
        assert hits[0].id == "v7" and hits[0].score == pytest.approx(1.0, abs=1e-5)
        filtered = index.search(vectors[7], 5, compile_filter({"parity": "even"}))
        assert all(h.metadata["parity"] == "even" for h in filtered)
        assert index.search(vectors[0], 0) == []

    @pytest.mark.parametrize("kind", ["hnsw", "ivf"])
    def test_approximate_recall(self, kind: str) -> None:
        vectors = clustered()
        noise = np.random.default_rng(11).standard_normal((30, DIM)).astype(np.float32)
        queries = l2_normalize(vectors[::13][:30] + 0.3 * noise)
        truth = FlatIndex(DIM)
        fill(truth, vectors)
        index = create_index(
            IndexSettings(kind=kind, ivf_nlist=10, ivf_nprobe=3, ivf_train_threshold=100), DIM
        )
        fill(index, vectors)
        assert recall(index, truth, queries) >= 0.9
        flt = compile_filter({"tags": "t1"})
        assert recall(index, truth, queries, flt=flt) >= 0.9
        assert all("t1" in h.metadata["tags"] for h in index.search(queries[0], 10, flt))

    def test_ivf_trains_lazily_and_retrains_on_growth(self) -> None:
        vectors = clustered()
        index = IVFIndex(DIM, nlist=8, nprobe=2, train_threshold=100)
        fill(index, vectors[:50])
        assert not index.trained
        assert index.search(vectors[3], 1)[0].id == "v3"  # exact fallback
        index.add([f"w{i}" for i in range(60)], vectors[50:110], [{} for _ in range(60)])
        assert index.trained and index.trained_size >= 100
        index.add([f"x{i}" for i in range(200)], vectors[110:310], [{} for _ in range(200)])
        assert index.trained_size == 200  # retrained when the collection doubled
        assert index.stats()["nlist"] == 8

    def test_ivf_highly_selective_filter_falls_back_to_exact(self) -> None:
        vectors = clustered()
        index = IVFIndex(DIM, nlist=10, nprobe=1, train_threshold=100)
        fill(index, vectors)
        hits = index.search(vectors[0], 3, compile_filter({"n": 399}))
        assert [h.id for h in hits] == ["v399"]

    @pytest.mark.parametrize("cls", [FlatIndex, HNSWIndex, IVFIndex])
    def test_remove_upsert_and_rebuild(self, cls: type[VectorIndex]) -> None:
        vectors = clustered(per_cluster=15)
        index = cls(DIM)
        ids = fill(index, vectors)
        assert index.remove(ids[:50]) == 50
        assert index.remove(["unknown"]) == 0
        assert len(index) == len(ids) - 50
        assert all(h.id not in ids[:50] for h in index.search(vectors[0], 20))
        index.add(["v60"], vectors[0:1], [{"replaced": True}])
        assert index.search(vectors[0], 1)[0].id == "v60"
        assert index.stats()["tombstones"] == 51
        index.rebuild()
        assert index.stats()["tombstones"] == 0 and len(index) == len(ids) - 50
        index.update_metadata(["v60"], {"category": "x"})
        assert index.search(vectors[0], 1, compile_filter({"category": "x"}))[0].id == "v60"

    def test_validation(self) -> None:
        index = FlatIndex(DIM)
        with pytest.raises(ValidationError):
            index.add(["a"], np.zeros((2, DIM), dtype=np.float32), [{}])
        with pytest.raises(ValidationError):
            index.add(["a"], np.zeros((1, DIM + 1), dtype=np.float32), [{}])
        with pytest.raises(ValidationError):
            index.search(np.zeros(DIM + 1, dtype=np.float32), 1)
        assert index.search(np.zeros(DIM, dtype=np.float32), 5) == []
        with pytest.raises(ValueError, match="nprobe"):
            IVFIndex(DIM, nlist=2, nprobe=3)

    def test_hnsw_stats_and_levels(self) -> None:
        index = HNSWIndex(DIM, m=8)
        fill(index, clustered(per_cluster=20))
        stats = index.stats()
        assert stats["max_level"] >= 1
        assert 0 < stats["mean_degree_level0"] <= 16
        assert sum(stats["levels_histogram"].values()) == 200


class TestPersistence:
    @pytest.mark.parametrize("kind", ["flat", "ivf", "hnsw"])
    def test_round_trip(self, tmp_path: Path, kind: str) -> None:
        vectors = clustered(per_cluster=15)
        settings = IndexSettings(kind=kind, ivf_nlist=5, ivf_nprobe=2, ivf_train_threshold=20)
        index = create_index(settings, DIM)
        fill(index, vectors)
        index.remove(["v1"])
        index.save(tmp_path, {"provider": "test"})
        index.save(tmp_path, {"provider": "test"})  # second generation replaces the first
        assert len(list(tmp_path.glob("vectors-*.npz"))) == 1
        restored = create_index(settings, DIM)
        assert restored.load(tmp_path, {"provider": "test"})
        assert len(restored) == len(index)
        for q in vectors[:5]:
            assert [h.id for h in restored.search(q, 5)] == [h.id for h in index.search(q, 5)]

    def test_signature_or_kind_mismatch_means_stale(self, tmp_path: Path) -> None:
        index = FlatIndex(DIM)
        fill(index, clustered(per_cluster=2))
        index.save(tmp_path, {"provider": "a"})
        assert not FlatIndex(DIM).load(tmp_path, {"provider": "b"})
        assert not HNSWIndex(DIM).load(tmp_path, {"provider": "a"})
        assert not FlatIndex(DIM).load(tmp_path / "missing", {"provider": "a"})

    def test_corruption_is_detected(self, tmp_path: Path) -> None:
        index = FlatIndex(DIM)
        fill(index, clustered(per_cluster=2))
        index.save(tmp_path, {"p": 1})
        for npz in tmp_path.glob("*.npz"):
            npz.unlink()
        with pytest.raises(IndexCorruptedError):
            FlatIndex(DIM).load(tmp_path, {"p": 1})
        (tmp_path / "vectors.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(IndexCorruptedError):
            FlatIndex(DIM).load(tmp_path, {"p": 1})


def test_spherical_kmeans_separates_clusters() -> None:
    data = clustered(n_clusters=4, per_cluster=30)
    centroids, assignments = spherical_kmeans(data, 4, seed=1)
    assert centroids.shape == (4, DIM)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)
    # Points generated from the same centre end up in the same cluster.
    for block in range(4):
        labels = assignments[block * 30 : (block + 1) * 30]
        assert np.bincount(labels).max() >= 27
    with pytest.raises(ValueError):
        spherical_kmeans(np.zeros((0, DIM), dtype=np.float32), 2)
