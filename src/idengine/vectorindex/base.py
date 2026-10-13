"""Vector index interface and the shared dense vector storage."""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from idengine.errors import IndexCorruptedError, ValidationError
from idengine.fsutil import read_json, write_json_atomic
from idengine.vectorindex.filters import Predicate

Vector = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VectorHit:
    id: str
    score: float
    metadata: Mapping[str, Any]


class VectorStorage:
    """Growable row store: vectors, external ids, metadata and tombstones.

    Rows are never reused while the index is live; removed rows become
    tombstones (``alive[row] = False``) and are dropped on :meth:`compact`.
    """

    def __init__(self, dim: int, capacity: int = 64) -> None:
        self.dim = dim
        self.vectors: npt.NDArray[np.float32] = np.zeros((capacity, dim), dtype=np.float32)
        self.alive: npt.NDArray[np.bool_] = np.zeros(capacity, dtype=bool)
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.row_of: dict[str, int] = {}

    @property
    def size(self) -> int:
        """Number of rows ever allocated (including tombstones)."""
        return len(self.ids)

    @property
    def live_count(self) -> int:
        return len(self.row_of)

    def _grow(self, needed: int) -> None:
        capacity = self.vectors.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(needed, capacity * 2)
        vectors = np.zeros((new_capacity, self.dim), dtype=np.float32)
        vectors[:capacity] = self.vectors
        alive = np.zeros(new_capacity, dtype=bool)
        alive[:capacity] = self.alive
        self.vectors, self.alive = vectors, alive

    def append(self, item_id: str, vector: Vector, metadata: Mapping[str, Any]) -> int:
        if vector.shape != (self.dim,):
            raise ValidationError(f"vector has shape {vector.shape}, expected ({self.dim},)")
        if item_id in self.row_of:
            self.remove(item_id)
        row = self.size
        self._grow(row + 1)
        self.vectors[row] = vector
        self.alive[row] = True
        self.ids.append(item_id)
        self.metadata.append(dict(metadata))
        self.row_of[item_id] = row
        return row

    def remove(self, item_id: str) -> int | None:
        row = self.row_of.pop(item_id, None)
        if row is not None:
            self.alive[row] = False
        return row

    def live_rows(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.alive[: self.size])

    def filtered_rows(self, predicate: Predicate | None) -> npt.NDArray[np.int64]:
        rows = self.live_rows()
        if predicate is None:
            return rows
        return np.asarray([r for r in rows if predicate(self.metadata[r])], dtype=np.int64)

    def hit(self, row: int, score: float) -> VectorHit:
        return VectorHit(self.ids[row], float(score), self.metadata[row])

    def tombstone_ratio(self) -> float:
        return 0.0 if self.size == 0 else 1.0 - self.live_count / self.size

    def compacted(self) -> VectorStorage:
        fresh = VectorStorage(self.dim, capacity=max(64, self.live_count))
        for row in self.live_rows():
            fresh.append(self.ids[row], self.vectors[row], self.metadata[row])
        return fresh

    # -- persistence ---------------------------------------------------------
    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {"vectors": self.vectors[: self.size], "alive": self.alive[: self.size]}

    def manifest(self) -> dict[str, Any]:
        return {"ids": self.ids, "metadata": self.metadata, "dim": self.dim}

    @classmethod
    def restore(cls, manifest: Mapping[str, Any], arrays: Mapping[str, Any]) -> VectorStorage:
        vectors = np.asarray(arrays["vectors"], dtype=np.float32)
        alive = np.asarray(arrays["alive"], dtype=bool)
        ids = list(manifest["ids"])
        metadata = [dict(m) for m in manifest["metadata"]]
        dim = int(manifest["dim"])
        if not (len(ids) == len(metadata) == vectors.shape[0] == alive.shape[0]):
            raise IndexCorruptedError("vector index files are inconsistent")
        if vectors.ndim != 2 or (vectors.size and vectors.shape[1] != dim):
            raise IndexCorruptedError("vector dimensionality does not match manifest")
        storage = cls(dim, capacity=max(64, len(ids)))
        storage.vectors[: len(ids)] = vectors
        storage.alive[: len(ids)] = alive
        storage.ids = ids
        storage.metadata = metadata
        storage.row_of = {item_id: row for row, item_id in enumerate(ids) if alive[row]}
        return storage


def top_k(rows: npt.NDArray[np.int64], scores: npt.NDArray[np.float32], k: int) -> list[int]:
    """Indices into ``rows``/``scores`` of the ``k`` best scores, best first."""
    if scores.size == 0 or k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k] if scores.size > k else np.arange(scores.size)
    return sorted(part.tolist(), key=lambda i: (-float(scores[i]), int(rows[i])))


class VectorIndex(ABC):
    """Common behaviour: locking, storage, persistence via generations."""

    kind: str = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.storage = VectorStorage(dim)
        self.lock = threading.RLock()

    def __len__(self) -> int:
        return self.storage.live_count

    def __contains__(self, item_id: object) -> bool:
        return item_id in self.storage.row_of

    def add(
        self,
        ids: Sequence[str],
        vectors: npt.NDArray[np.float32],
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValidationError("ids, vectors and metadata must have the same length")
        with self.lock:
            for item_id, vector, meta in zip(ids, vectors, metadata, strict=True):
                if item_id in self.storage.row_of:
                    self._on_remove(self.storage.row_of[item_id])
                row = self.storage.append(item_id, np.asarray(vector, dtype=np.float32), meta)
                self._on_add(row)

    def remove(self, ids: Iterable[str]) -> int:
        removed = 0
        with self.lock:
            for item_id in ids:
                row = self.storage.remove(item_id)
                if row is not None:
                    self._on_remove(row)
                    removed += 1
        return removed

    def update_metadata(self, ids: Iterable[str], changes: Mapping[str, Any]) -> None:
        with self.lock:
            for item_id in ids:
                row = self.storage.row_of.get(item_id)
                if row is not None:
                    self.storage.metadata[row].update(changes)

    def search(self, query: Vector, k: int, predicate: Predicate | None = None) -> list[VectorHit]:
        if query.shape != (self.dim,):
            raise ValidationError(f"query vector has shape {query.shape}, expected ({self.dim},)")
        with self.lock:
            if self.storage.live_count == 0 or k <= 0:
                return []
            return self._search(np.asarray(query, dtype=np.float32), k, predicate)

    def exact_search(
        self, query: Vector, k: int, predicate: Predicate | None = None
    ) -> list[VectorHit]:
        """Brute-force cosine search over all live rows (ground truth)."""
        rows = self.storage.filtered_rows(predicate)
        if rows.size == 0:
            return []
        scores = self.storage.vectors[rows] @ query
        return [self.storage.hit(int(rows[i]), scores[i]) for i in top_k(rows, scores, k)]

    @abstractmethod
    def _search(self, query: Vector, k: int, predicate: Predicate | None) -> list[VectorHit]: ...

    def _on_add(self, row: int) -> None:  # noqa: B027 - optional hook
        """Hook for structures that index rows incrementally."""

    def _on_remove(self, row: int) -> None:  # noqa: B027 - optional hook
        """Hook for structures that track removed rows."""

    def rebuild(self) -> None:
        """Compact tombstones and rebuild auxiliary structures."""
        with self.lock:
            self.storage = self.storage.compacted()
            self._rebuild_structure()

    def _rebuild_structure(self) -> None:  # noqa: B027 - optional hook
        """Recreate auxiliary structures from ``self.storage``."""

    def stats(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "vectors": self.storage.live_count,
            "tombstones": self.storage.size - self.storage.live_count,
            "tombstone_ratio": round(self.storage.tombstone_ratio(), 4),
        }

    # -- persistence -----------------------------------------------------------
    def _extra_arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {}

    def _extra_manifest(self) -> dict[str, Any]:
        return {}

    def _restore_extra(self, manifest: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        self._rebuild_structure()

    def save(self, directory: Path, signature: Mapping[str, Any]) -> None:
        """Write a new generation of files, then atomically switch the manifest."""
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "vectors.json"
        with self.lock:
            try:
                previous = read_json(manifest_path)
            except IndexCorruptedError:
                previous = {}  # overwritten below; never trust a corrupt manifest
            existing = [
                int(p.stem.split("-", 1)[1])
                for p in directory.glob("vectors-*.npz")
                if p.stem.split("-", 1)[1].isdigit()
            ]
            generation = max([int(previous.get("generation", 0)), *existing]) + 1
            arrays_name = f"vectors-{generation}.npz"
            np.savez(directory / arrays_name, **self.storage.arrays(), **self._extra_arrays())
            manifest = {
                "format": 1,
                "kind": self.kind,
                "generation": generation,
                "arrays": arrays_name,
                "signature": dict(signature),
                "storage": self.storage.manifest(),
                "structure": self._extra_manifest(),
            }
            write_json_atomic(manifest_path, manifest)
        if previous and previous.get("arrays") and previous["arrays"] != arrays_name:
            # A stale generation left behind (e.g. file locked by a scanner) is harmless.
            with contextlib.suppress(OSError):
                (directory / previous["arrays"]).unlink(missing_ok=True)

    def load(self, directory: Path, signature: Mapping[str, Any]) -> bool:
        """Load persisted state; returns False if absent or built for another vector space."""
        manifest = read_json(directory / "vectors.json")
        if not manifest:
            return False
        if manifest.get("kind") != self.kind or manifest.get("signature") != dict(signature):
            return False
        arrays_path = directory / str(manifest.get("arrays", ""))
        if not arrays_path.is_file():
            raise IndexCorruptedError(f"missing vector data file {arrays_path.name}")
        try:
            with np.load(arrays_path) as arrays:
                loaded = {name: arrays[name] for name in arrays.files}
        except (OSError, ValueError) as exc:
            raise IndexCorruptedError(f"unreadable vector data: {exc}") from exc
        with self.lock:
            self.storage = VectorStorage.restore(manifest["storage"], loaded)
            self._restore_extra(manifest.get("structure", {}), loaded)
        return True
