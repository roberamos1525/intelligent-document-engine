"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float32]


def l2_normalize(matrix: npt.NDArray[np.floating[Any]]) -> Matrix:
    """Row-normalise to unit length; all-zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix / norm if norm > 0 else matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class EmbeddingProvider(ABC):
    """Turns text into dense, L2-normalised vectors.

    Providers that learn from the corpus (``requires_fit``) must be fitted
    before ``embed`` is called and can persist their state with
    ``save``/``load``.
    """

    name: str = "base"
    requires_fit: bool = False

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> Matrix:
        """Embed a batch of texts into an ``(n, dim)`` float32 matrix."""

    def embed_query(self, text: str) -> Matrix:
        return self.embed([text])[0]

    @property
    def is_fitted(self) -> bool:
        return not self.requires_fit

    def fit(self, texts: Sequence[str]) -> None:  # noqa: B027 - optional hook
        """Learn corpus statistics (no-op for stateless providers)."""

    def save(self, directory: Path) -> None:  # noqa: B027 - optional hook
        """Persist learned state (no-op for stateless providers)."""

    def load(self, directory: Path) -> bool:
        """Restore learned state; returns False when nothing was saved."""
        return not self.requires_fit

    def signature(self) -> dict[str, Any]:
        """Identifies the vector space; indexes built with another signature are stale."""
        return {"provider": self.name, "dim": self.dim}
