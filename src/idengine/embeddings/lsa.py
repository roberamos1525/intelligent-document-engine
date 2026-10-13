"""Latent Semantic Analysis embeddings: TF-IDF + randomized truncated SVD.

The TF-IDF matrix is kept in a minimal CSR representation so that corpora with
large vocabularies never need a dense ``documents x vocabulary`` matrix. The
SVD uses the randomized range finder of Halko, Martinsson & Tropp (2011) with
power iterations, which only needs products with the sparse matrix and its
transpose.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from idengine.embeddings.base import EmbeddingProvider, Matrix, l2_normalize
from idengine.errors import ModelNotTrainedError, ValidationError
from idengine.text.tokenize import analyze, term_frequencies

FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class CsrMatrix:
    """Compressed sparse row matrix with just the products LSA needs."""

    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    data: FloatArray
    shape: tuple[int, int]

    def dot(self, dense: FloatArray) -> FloatArray:
        """``self @ dense`` for a dense ``(n_cols, k)`` matrix."""
        rows, _ = self.shape
        out = np.zeros((rows, dense.shape[1]), dtype=np.float64)
        if self.data.size == 0:
            return out
        contributions = self.data[:, None] * dense[self.indices]
        row_ids = np.repeat(np.arange(rows), np.diff(self.indptr))
        np.add.at(out, row_ids, contributions)
        return out

    def tdot(self, dense: FloatArray) -> FloatArray:
        """``self.T @ dense`` for a dense ``(n_rows, k)`` matrix."""
        rows, cols = self.shape
        out = np.zeros((cols, dense.shape[1]), dtype=np.float64)
        if self.data.size == 0:
            return out
        row_ids = np.repeat(np.arange(rows), np.diff(self.indptr))
        np.add.at(out, self.indices, self.data[:, None] * dense[row_ids])
        return out


def randomized_svd(
    matrix: CsrMatrix, rank: int, *, oversample: int = 10, power_iters: int = 4, seed: int = 0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Approximate the top-``rank`` singular triplets ``(U, S, Vt)`` of ``matrix``."""
    rows, cols = matrix.shape
    k = min(rank + oversample, rows, cols)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, k))
    y = matrix.dot(omega)
    q, _ = np.linalg.qr(y)
    for _ in range(power_iters):
        # Re-orthonormalise between products for numerical stability.
        z, _ = np.linalg.qr(matrix.tdot(q))
        q, _ = np.linalg.qr(matrix.dot(z))
    b = matrix.tdot(q).T  # (k, cols) == Q^T A
    u_small, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ u_small
    rank = min(rank, s.size)
    return u[:, :rank], s[:rank], vt[:rank]


class LsaEmbedder(EmbeddingProvider):
    name = "lsa"
    requires_fit = True

    def __init__(
        self,
        dim: int = 128,
        *,
        seed: int = 13,
        min_df: int = 1,
        max_df_ratio: float = 0.95,
        max_features: int = 50_000,
    ) -> None:
        self._dim = dim
        self.seed = seed
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: FloatArray = np.zeros(0)
        self.components: FloatArray = np.zeros((0, 0))  # (dim, vocab)
        self.singular_values: FloatArray = np.zeros(0)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_fitted(self) -> bool:
        return self.components.size > 0

    def _tfidf_rows(self, texts: Sequence[str]) -> CsrMatrix:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for text in texts:
            row: dict[int, float] = {}
            for term, tf in term_frequencies(analyze(text)).items():
                col = self.vocabulary.get(term)
                if col is not None:
                    row[col] = (1.0 + math.log(tf)) * float(self.idf[col])
            norm = math.sqrt(sum(v * v for v in row.values())) or 1.0
            for col in sorted(row):
                indices.append(col)
                data.append(row[col] / norm)
            indptr.append(len(indices))
        return CsrMatrix(
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(data, dtype=np.float64),
            (len(texts), len(self.vocabulary)),
        )

    def fit(self, texts: Sequence[str]) -> None:
        if len(texts) < 2:
            raise ValidationError("LSA needs at least two texts to fit")
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(analyze(text)))
        max_df = max(1, int(self.max_df_ratio * len(texts)))
        candidates = [
            (term, count)
            for term, count in df.items()
            if count >= self.min_df and (count <= max_df or len(texts) < 10)
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        terms = sorted(term for term, _ in candidates[: self.max_features])
        if not terms:
            raise ValidationError("LSA vocabulary is empty after filtering")
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms], dtype=np.float64
        )
        matrix = self._tfidf_rows(texts)
        _, s, vt = randomized_svd(matrix, self._dim, seed=self.seed)
        components = np.zeros((self._dim, len(terms)), dtype=np.float64)
        components[: vt.shape[0]] = vt
        self.components = components
        self.singular_values = np.pad(s, (0, self._dim - s.size))

    def embed(self, texts: Sequence[str]) -> Matrix:
        if not self.is_fitted:
            raise ModelNotTrainedError("LSA embedder must be fitted before use (run reindex)")
        matrix = self._tfidf_rows(texts)
        return l2_normalize(matrix.dot(self.components.T))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.vocabulary, key=self.vocabulary.__getitem__)
        np.savez(
            directory / "lsa.npz",
            idf=self.idf,
            components=self.components,
            singular_values=self.singular_values,
        )
        (directory / "lsa_vocab.json").write_text(json.dumps(terms), encoding="utf-8")

    def load(self, directory: Path) -> bool:
        arrays_path = directory / "lsa.npz"
        vocab_path = directory / "lsa_vocab.json"
        if not arrays_path.exists() or not vocab_path.exists():
            return False
        with np.load(arrays_path) as arrays:
            components = arrays["components"]
            if components.shape[0] != self._dim:
                return False  # saved with another dimensionality: refit needed
            self.idf = arrays["idf"]
            self.components = components
            self.singular_values = arrays["singular_values"]
        terms = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        return True

    def signature(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "dim": self._dim,
            "seed": self.seed,
            "vocabulary": len(self.vocabulary),
        }
