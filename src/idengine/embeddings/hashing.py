"""Stateless hashing-trick embeddings with an implicit sparse random projection.

Every analysed feature (stemmed unigram, optionally bigram) is mapped by a
keyed BLAKE2b hash to ``k`` signed positions of a ``dim``-dimensional vector.
This equals multiplying an (unbounded) hashed bag-of-words vector by a very
sparse random projection matrix (Achlioptas / Li et al.) without ever
materialising the matrix. Weights use sublinear term frequency so repeated
words do not dominate, and vectors are L2-normalised so dot product equals
cosine similarity.

Because there is no learned state, documents can be embedded incrementally
and the vector space never drifts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from idengine.embeddings.base import EmbeddingProvider, Matrix, l2_normalize
from idengine.text.tokenize import analyze, bigrams, term_frequencies

_NONZEROS_PER_FEATURE = 6
_BIGRAM_WEIGHT = 0.5


class HashingEmbedder(EmbeddingProvider):
    name = "hashing"

    def __init__(self, dim: int = 256, *, seed: int = 13, use_bigrams: bool = True) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self._dim = dim
        self.seed = seed
        self.use_bigrams = use_bigrams
        self._key = seed.to_bytes(8, "little", signed=True)
        self._project = lru_cache(maxsize=200_000)(self._project_uncached)

    @property
    def dim(self) -> int:
        return self._dim

    def _project_uncached(self, feature: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=4 * _NONZEROS_PER_FEATURE, key=self._key
        ).digest()
        indices: list[int] = []
        signs: list[float] = []
        for i in range(_NONZEROS_PER_FEATURE):
            word = int.from_bytes(digest[4 * i : 4 * i + 4], "little")
            indices.append((word >> 1) % self._dim)
            signs.append(1.0 if word & 1 else -1.0)
        return tuple(indices), tuple(signs)

    def features(self, text: str) -> dict[str, float]:
        terms = analyze(text)
        weighted: dict[str, float] = {
            term: 1.0 + math.log(tf) for term, tf in term_frequencies(terms).items()
        }
        if self.use_bigrams:
            for gram, tf in term_frequencies(bigrams(terms)).items():
                weighted[gram] = _BIGRAM_WEIGHT * (1.0 + math.log(tf))
        return weighted

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        scale = 1.0 / math.sqrt(_NONZEROS_PER_FEATURE)
        for row, text in enumerate(texts):
            vector = out[row]
            for feature, weight in self.features(text).items():
                indices, signs = self._project(feature)
                for index, sign in zip(indices, signs, strict=True):
                    vector[index] += sign * weight * scale
        return l2_normalize(out)

    def signature(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "dim": self._dim,
            "seed": self.seed,
            "bigrams": self.use_bigrams,
        }
