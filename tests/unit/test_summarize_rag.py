from __future__ import annotations

import numpy as np
import pytest

from idengine.embeddings import HashingEmbedder
from idengine.errors import ValidationError
from idengine.llm.base import LLMAdapter, LLMRequest, LLMResponse
from idengine.rag.context import assemble_context
from idengine.rag.rerank import RankedChunk, Reranker, proximity_score
from idengine.search.service import SearchHit
from idengine.summarize import MapReduceSummarizer, Summarizer, SummaryMethod, textrank_summary
from idengine.summarize.textrank import pagerank, rank_sentences, similarity_matrix


def hit(chunk_id: str, text: str, score: float = 0.5, **kw: object) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=kw.get("document_id", "doc_" + chunk_id),  # type: ignore[arg-type]
        title=kw.get("title", "Doc " + chunk_id),  # type: ignore[arg-type]
        filename=None,
        text=text,
        snippet=text[:50],
        highlights=[],
        score=score,
        heading_path=list(kw.get("heading_path", [])),  # type: ignore[call-overload]
        page_start=kw.get("page_start"),  # type: ignore[arg-type]
        page_end=kw.get("page_end"),  # type: ignore[arg-type]
    )


class RecordingLLM(LLMAdapter):
    name = "recording"
    model = "test"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        joined = " ".join(p.text for p in request.passages)
        return LLMResponse(f"S({len(request.passages)}):{joined[:40]}", self.name, self.model)


class TestTextRank:
    def test_pagerank_properties(self) -> None:
        weights = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        scores = pagerank(weights)
        assert scores.sum() == pytest.approx(1.0)
        assert scores[0] > scores[1] == pytest.approx(scores[2])
        assert pagerank(np.zeros((0, 0))).size == 0
        isolated = pagerank(np.zeros((3, 3)))
        assert np.allclose(isolated, 1 / 3)

    def test_similarity_matrix_is_symmetric_with_zero_diagonal(self) -> None:
        sim = similarity_matrix([["a", "b"], ["b", "c"], ["d"]])
        assert np.allclose(sim, sim.T) and np.allclose(np.diag(sim), 0)
        assert sim[0, 2] == 0 and sim[0, 1] > 0

    def test_central_sentences_win_and_order_is_preserved(self) -> None:
        text = (
            "The invoice payment process requires approval from finance. "
            "Cats sleep a lot during the day. "
            "Finance approval of each invoice payment is logged in the payment system. "
            "Every invoice payment over a limit needs a second finance approval."
        )
        summary = textrank_summary(text, max_sentences=2)
        assert [s.index for s in summary] == sorted(s.index for s in summary)
        assert all("Cats" not in s.text for s in summary)
        assert textrank_summary("", max_sentences=2) == []
        short = rank_sentences(["Hi.", "Ok."])
        assert all(r.score == 0 for r in short)


class TestMapReduce:
    def test_map_then_reduce(self) -> None:
        llm = RecordingLLM()
        text, stages = MapReduceSummarizer(llm).summarize(["one.", "two.", " "], max_sentences=3)
        assert [s["stage"] for s in stages] == ["map", "reduce"]
        assert len(llm.requests) == 3  # two maps + one reduce
        assert llm.requests[-1].max_sentences == 3 and text.startswith("S(2)")

    def test_collapse_when_partials_exceed_budget(self) -> None:
        llm = RecordingLLM()
        summarizer = MapReduceSummarizer(llm, reduce_token_budget=12)
        _, stages = summarizer.summarize([f"chunk {i} text." for i in range(8)], max_sentences=2)
        assert any(s["stage"].startswith("collapse") for s in stages)
        assert summarizer.summarize([], max_sentences=2) == ("", [])

    def test_summarizer_facade(self) -> None:
        summarizer = Summarizer(RecordingLLM())
        textrank = summarizer.summarize(
            ["A sentence about invoices and payments here."], max_sentences=1
        )
        assert textrank.method is SummaryMethod.TEXTRANK and textrank.provider == "textrank"
        mapreduce = summarizer.summarize(["x y z."], method="mapreduce", max_sentences=1)
        assert mapreduce.provider == "recording" and mapreduce.stages
        with pytest.raises(ValidationError):
            summarizer.summarize(["x"], max_sentences=0)


class TestRerank:
    def test_proximity(self) -> None:
        assert proximity_score(["a", "b", "x", "c"], {"a", "b"}) == 1.0
        assert proximity_score(["a", "x", "x", "x", "b"], {"a", "b"}) == pytest.approx(2 / 5)
        assert proximity_score(["x"], {"a"}) == 0.0
        assert proximity_score(["a"], {"a", "b"}) == 1.0

    def test_scores_prefer_coverage(self) -> None:
        reranker = Reranker(HashingEmbedder(dim=64))
        ranked = reranker.score(
            "invoice payment deadline",
            [
                hit("a", "Unrelated text about cats.", 0.9),
                hit("b", "The invoice payment deadline is May.", 0.1),
            ],
        )
        best = max(ranked, key=lambda r: r.score)
        assert best.hit.chunk_id == "b"
        assert set(best.features) == {"retrieval", "coverage", "proximity", "semantic"}

    def test_mmr_diversifies(self) -> None:
        text = "Invoice payment is due within thirty days of receipt."
        hits = [
            hit("a", text, 0.9),
            hit("b", text, 0.85),
            hit("c", "Invoice payment by bank transfer only.", 0.5),
        ]
        selected = Reranker(HashingEmbedder(dim=64), mmr_lambda=0.5).select(
            "invoice payment", hits, 2
        )
        assert [r.hit.chunk_id for r in selected] == ["a", "c"]
        jaccard = Reranker(None, mmr_lambda=0.5).select("invoice payment", hits, 2)
        assert [r.hit.chunk_id for r in jaccard] == ["a", "c"]
        assert Reranker(None).select("q", [], 3) == []


class TestContext:
    def test_numbering_budget_and_truncation(self) -> None:
        long_text = " ".join(f"Sentence {i} has several words in it." for i in range(30))
        chunks = [
            RankedChunk(
                hit(
                    "a",
                    "Short first passage.",
                    page_start=2,
                    page_end=3,
                    heading_path=["Doc a", "Terms"],
                ),
                0.9,
            ),
            RankedChunk(hit("b", long_text), 0.8),
            RankedChunk(hit("c", "Never included."), 0.1),
        ]
        context = assemble_context(chunks, token_budget=60)
        assert [p.number for p in context.passages] == [1, 2]
        assert context.truncated and context.tokens <= 60
        assert context.passages[1].text.endswith(".")
        assert context.citations[0].source == "Doc a, pp. 2-3, Terms"
        assert context.citations[1].chunk_id == "b"

    def test_stops_when_remaining_budget_is_small(self) -> None:
        chunks = [
            RankedChunk(hit("a", "word " * 45), 0.9),
            RankedChunk(hit("b", "Another sentence here. And more."), 0.5),
        ]
        context = assemble_context(chunks, token_budget=50)
        assert len(context.passages) == 1 and context.truncated
