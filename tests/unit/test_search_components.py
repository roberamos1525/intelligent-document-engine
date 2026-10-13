from __future__ import annotations

from pathlib import Path

import pytest

from idengine.errors import IndexCorruptedError
from idengine.search.bm25 import BM25Index
from idengine.search.fusion import normalize_scores, reciprocal_rank_fusion
from idengine.search.highlight import make_snippet
from idengine.text.tokenize import analyze
from idengine.vectorindex import compile_filter

DOCS = {
    "a": "invoice payment due in thirty days invoice total",
    "b": "remote work policy for employees",
    "c": "late invoice payment fee",
    "d": "deployment rollback runbook",
}


@pytest.fixture
def bm25() -> BM25Index:
    index = BM25Index()
    for key, text in DOCS.items():
        index.add(key, analyze(text), {"group": "finance" if key in "ac" else "other"})
    return index


class TestBM25:
    def test_ranking_and_matched_terms(self, bm25: BM25Index) -> None:
        hits = bm25.search(analyze("invoice payment"), 10)
        # Length normalisation favours the short document "c" over the longer "a".
        assert [h.id for h in hits] == ["c", "a"]
        assert set(hits[0].matched_terms) == {"invoic", "payment"}
        assert hits[0].score > hits[1].score > 0
        # Without length normalisation (b=0) the higher term frequency in "a" wins.
        flat = BM25Index(b=0.0)
        for key, text in DOCS.items():
            flat.add(key, analyze(text), {})
        assert [h.id for h in flat.search(analyze("invoice payment"), 10)] == ["a", "c"]

    def test_idf_and_statistics(self, bm25: BM25Index) -> None:
        assert bm25.df("invoic") == 2
        assert bm25.idf("rollback") > bm25.idf("invoic")
        assert bm25.stats()["documents"] == 4
        assert bm25.avg_len == pytest.approx(sum(len(analyze(t)) for t in DOCS.values()) / 4)

    def test_filter_remove_and_update(self, bm25: BM25Index) -> None:
        only_other = compile_filter({"group": "other"})
        assert bm25.search(analyze("invoice"), 5, only_other) == []
        bm25.update_metadata(["c"], {"group": "other"})
        assert [h.id for h in bm25.search(analyze("invoice"), 5, only_other)] == ["c"]
        assert bm25.remove(["a", "zzz"]) == 1
        assert "a" not in bm25 and bm25.df("thirti") == 0
        bm25.add("c", analyze("completely different"), {})
        assert bm25.search(analyze("invoice"), 5) == []

    def test_degenerate_queries(self, bm25: BM25Index) -> None:
        assert bm25.search([], 5) == []
        assert bm25.search(["unknown"], 5) == []
        assert bm25.search(analyze("invoice"), 0) == []

    def test_persistence(self, bm25: BM25Index, tmp_path: Path) -> None:
        path = tmp_path / "bm25.json"
        bm25.save(path)
        restored = BM25Index()
        assert restored.load(path)
        query = analyze("invoice payment fee")
        assert [(h.id, round(h.score, 6)) for h in restored.search(query, 4)] == [
            (h.id, round(h.score, 6)) for h in bm25.search(query, 4)
        ]
        assert not BM25Index().load(tmp_path / "missing.json")
        path.write_text('{"analyzer": 999, "docs": {}}', encoding="utf-8")
        assert not BM25Index().load(path)
        path.write_text('{"analyzer": 1, "docs": []}', encoding="utf-8")
        with pytest.raises(IndexCorruptedError):
            BM25Index().load(path)


class TestFusion:
    def test_rrf_combines_rankings(self) -> None:
        fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "bm25": ["b", "d"]}, k=60)
        ids = [item for item, _, _ in fused]
        assert ids[0] == "b"  # present in both rankings
        assert fused[0][2] == {"vector": 2, "bm25": 1}
        assert set(ids) == {"a", "b", "c", "d"}

    def test_rrf_weights_and_duplicates(self) -> None:
        fused = reciprocal_rank_fusion(
            {"x": ["a", "a", "b"], "y": ["b"]}, k=1, weights={"x": 1.0, "y": 0.1}
        )
        scores = {item: score for item, score, _ in fused}
        assert scores["a"] == pytest.approx(1 / 2)
        assert scores["b"] == pytest.approx(1 / 4 + 0.1 / 2)

    def test_normalize_scores(self) -> None:
        assert normalize_scores({}) == {}
        assert normalize_scores({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
        assert normalize_scores({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}


class TestHighlight:
    def test_best_window_and_offsets(self) -> None:
        filler = " ".join(f"filler{i}" for i in range(80))
        text = f"{filler} the invoice payment is overdue {filler}"
        snippet = make_snippet(text, set(analyze("invoice payment")), window_words=20)
        assert snippet.text.startswith("… ") and snippet.text.endswith(" …")
        highlighted = [snippet.text[s:e] for s, e in snippet.highlights]
        assert highlighted == ["invoice", "payment"]

    def test_short_text_is_returned_whole(self) -> None:
        snippet = make_snippet("Invoices are due.", {"invoic"})
        assert snippet.text == "Invoices are due."
        assert snippet.highlights == [(0, 8)]
        assert make_snippet("", {"x"}).text == ""
