"""Bag-of-words vectoriser producing raw counts or TF-IDF features."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from idengine.text.tokenize import analyze, bigrams

FloatMatrix = npt.NDArray[np.float64]


class Vectorizer:
    """Vocabulary learned from training texts (unigrams + optional bigrams)."""

    def __init__(
        self, *, max_features: int = 20_000, min_df: int = 1, use_bigrams: bool = True
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.use_bigrams = use_bigrams
        self.vocabulary: dict[str, int] = {}
        self.idf: npt.NDArray[np.float64] = np.zeros(0)

    def _features(self, text: str) -> list[str]:
        terms = analyze(text)
        return terms + bigrams(terms) if self.use_bigrams else terms

    def fit(self, texts: Sequence[str]) -> Vectorizer:
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(self._features(text)))
        kept = [(t, c) for t, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda item: (-item[1], item[0]))
        terms = sorted(t for t, _ in kept[: self.max_features])
        self.vocabulary = {t: i for i, t in enumerate(terms)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms])
        return self

    def counts(self, texts: Sequence[str]) -> FloatMatrix:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                col = self.vocabulary.get(feature)
                if col is not None:
                    matrix[row, col] += 1.0
        return matrix

    def tfidf(self, texts: Sequence[str]) -> FloatMatrix:
        counts = self.counts(texts)
        weighted = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0) * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return weighted / norms

    def state(self) -> dict[str, Any]:
        terms = sorted(self.vocabulary, key=self.vocabulary.__getitem__)
        return {
            "terms": terms,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "use_bigrams": self.use_bigrams,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], idf: npt.NDArray[np.float64]) -> Vectorizer:
        vec = cls(
            max_features=int(state["max_features"]),
            min_df=int(state["min_df"]),
            use_bigrams=bool(state["use_bigrams"]),
        )
        vec.vocabulary = {t: i for i, t in enumerate(state["terms"])}
        vec.idf = np.asarray(idf, dtype=np.float64)
        if vec.idf.shape[0] != len(vec.vocabulary):
            raise ValueError("idf vector does not match vocabulary size")
        return vec
