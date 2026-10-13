"""Okapi BM25 inverted index with metadata filtering and JSON persistence."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idengine.errors import IndexCorruptedError
from idengine.fsutil import read_json, write_json_atomic
from idengine.text.tokenize import term_frequencies
from idengine.vectorindex.filters import Predicate

ANALYZER_VERSION = 1


@dataclass(frozen=True, slots=True)
class LexicalHit:
    id: str
    score: float
    metadata: Mapping[str, Any]
    matched_terms: tuple[str, ...]


class BM25Index:
    """In-memory BM25 (Robertson & Zaragoza) over pre-analysed terms.

    ``doc_terms`` is the source of truth; postings are derived from it so
    persistence only needs to store one structure.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_terms: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.total_len = 0
        self.lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.doc_terms)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self.doc_terms

    @property
    def avg_len(self) -> float:
        return self.total_len / len(self.doc_terms) if self.doc_terms else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        n = len(self.doc_terms)
        df = self.df(term)
        # BM25+ style floor keeps very common terms from going negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def add(self, item_id: str, terms: Sequence[str], metadata: Mapping[str, Any]) -> None:
        with self.lock:
            if item_id in self.doc_terms:
                self._remove(item_id)
            tf = term_frequencies(terms)
            self.doc_terms[item_id] = tf
            self.doc_len[item_id] = len(terms)
            self.metadata[item_id] = dict(metadata)
            self.total_len += len(terms)
            for term, count in tf.items():
                self.postings.setdefault(term, {})[item_id] = count

    def _remove(self, item_id: str) -> bool:
        tf = self.doc_terms.pop(item_id, None)
        if tf is None:
            return False
        for term in tf:
            posting = self.postings.get(term)
            if posting is not None:
                posting.pop(item_id, None)
                if not posting:
                    del self.postings[term]
        self.total_len -= self.doc_len.pop(item_id, 0)
        self.metadata.pop(item_id, None)
        return True

    def remove(self, ids: Iterable[str]) -> int:
        with self.lock:
            return sum(1 for item_id in ids if self._remove(item_id))

    def update_metadata(self, ids: Iterable[str], changes: Mapping[str, Any]) -> None:
        with self.lock:
            for item_id in ids:
                if item_id in self.metadata:
                    self.metadata[item_id].update(changes)

    def search(
        self, query_terms: Sequence[str], k: int, predicate: Predicate | None = None
    ) -> list[LexicalHit]:
        if k <= 0 or not query_terms:
            return []
        with self.lock:
            avg = self.avg_len or 1.0
            query_tf = term_frequencies(query_terms)
            scores: dict[str, float] = {}
            matched: dict[str, list[str]] = {}
            for term, qtf in query_tf.items():
                posting = self.postings.get(term)
                if not posting:
                    continue
                idf = self.idf(term)
                for item_id, tf in posting.items():
                    norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[item_id] / avg)
                    scores[item_id] = scores.get(item_id, 0.0) + qtf * idf * (
                        tf * (self.k1 + 1.0) / (tf + norm)
                    )
                    matched.setdefault(item_id, []).append(term)
            if predicate is not None:
                scores = {i: s for i, s in scores.items() if predicate(self.metadata[i])}
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
            return [
                LexicalHit(item_id, score, self.metadata[item_id], tuple(matched[item_id]))
                for item_id, score in ranked
            ]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.doc_terms),
            "vocabulary": len(self.postings),
            "avg_length": round(self.avg_len, 2),
            "k1": self.k1,
            "b": self.b,
        }

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        with self.lock:
            payload = {
                "format": 1,
                "analyzer": ANALYZER_VERSION,
                "docs": {
                    item_id: {
                        "tf": tf,
                        "len": self.doc_len[item_id],
                        "meta": self.metadata[item_id],
                    }
                    for item_id, tf in self.doc_terms.items()
                },
            }
            write_json_atomic(path, payload)

    def load(self, path: Path) -> bool:
        data = read_json(path)
        if not data:
            return False
        if data.get("analyzer") != ANALYZER_VERSION:
            return False
        docs = data.get("docs")
        if not isinstance(docs, dict):
            raise IndexCorruptedError("BM25 index file has no document table")
        with self.lock:
            self.doc_terms, self.doc_len, self.metadata, self.postings = {}, {}, {}, {}
            self.total_len = 0
            for item_id, entry in docs.items():
                tf = {str(t): int(c) for t, c in entry["tf"].items()}
                self.doc_terms[item_id] = tf
                self.doc_len[item_id] = int(entry["len"])
                self.metadata[item_id] = dict(entry.get("meta", {}))
                self.total_len += self.doc_len[item_id]
                for term, count in tf.items():
                    self.postings.setdefault(term, {})[item_id] = count
        return True
