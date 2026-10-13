"""Retrieval-augmented generation: retrieve -> rerank -> assemble -> generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from idengine.config import LLMSettings, RagSettings
from idengine.errors import LLMError, ValidationError
from idengine.llm.base import LLMAdapter, LLMRequest, Task
from idengine.llm.prompts import NOT_FOUND_ANSWER
from idengine.observability.logging import get_logger
from idengine.observability.metrics import EngineMetrics
from idengine.rag.context import Citation, assemble_context
from idengine.rag.rerank import Reranker
from idengine.search.service import SearchMode, SearchService

log = get_logger("rag")


@dataclass(slots=True)
class Answer:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    short_answer: str | None
    provider: str
    model: str
    took_ms: float
    retrieval: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class RagPipeline:
    def __init__(
        self,
        settings: RagSettings,
        llm_settings: LLMSettings,
        search: SearchService,
        reranker: Reranker,
        llm: LLMAdapter,
        metrics: EngineMetrics,
    ) -> None:
        self.settings = settings
        self.llm_settings = llm_settings
        self.search = search
        self.reranker = reranker
        self.llm = llm
        self.metrics = metrics

    def ask(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        mode: SearchMode | str = SearchMode.HYBRID,
    ) -> Answer:
        started = time.perf_counter()
        retrieved = self.search.search(
            question, mode=mode, top_k=self.settings.retrieve_k, filters=filters
        )
        if top_k is not None and not 1 <= top_k <= 50:
            raise ValidationError("top_k must be between 1 and 50")
        k = self.settings.rerank_k if top_k is None else top_k
        selected = self.reranker.select(question, retrieved.hits, k)
        relevant = [
            r
            for r in selected
            if r.features.get("coverage", 0) >= self.settings.min_relevance
            or r.features.get("semantic", 0) >= 0.3
        ]
        context = assemble_context(relevant, token_budget=self.settings.context_token_budget)
        retrieval_info = {
            "mode": retrieved.mode.value,
            "retrieved": len(retrieved.hits),
            "reranked": len(selected),
            "context_passages": len(context.passages),
            "context_tokens": context.tokens,
            "context_truncated": context.truncated,
            "warnings": retrieved.warnings,
        }

        if not context.passages:
            answer = Answer(
                question,
                NOT_FOUND_ANSWER,
                False,
                [],
                None,
                self.llm.name,
                self.llm.model,
                self._elapsed(started),
                retrieval_info,
            )
            self.metrics.rag_latency.observe(time.perf_counter() - started)
            return answer

        request = LLMRequest(
            task=Task.ANSWER,
            instruction=question,
            passages=context.passages,
            max_output_tokens=self.llm_settings.max_output_tokens,
            temperature=self.llm_settings.temperature,
            max_sentences=self.llm_settings.answer_sentences,
        )
        try:
            response = self.llm.generate(request)
            self.metrics.llm_calls.inc(labels={"provider": self.llm.name, "outcome": "ok"})
        except LLMError:
            self.metrics.llm_calls.inc(labels={"provider": self.llm.name, "outcome": "error"})
            raise
        cited = set(response.citations)
        citations = [c for c in context.citations if c.number in cited] if cited else []
        if response.answerable and not citations:
            # Model answered without citing: expose the full context for verification.
            citations = context.citations
        elapsed = time.perf_counter() - started
        self.metrics.rag_latency.observe(elapsed)
        log.info(
            "question answered",
            extra={
                "provider": response.provider,
                "passages": len(context.passages),
                "answerable": response.answerable,
            },
        )
        return Answer(
            question=question,
            answer=response.text,
            answerable=response.answerable,
            citations=citations if response.answerable else [],
            short_answer=response.short_answer,
            provider=response.provider,
            model=response.model,
            took_ms=round(elapsed * 1000, 2),
            retrieval=retrieval_info,
            usage=response.usage,
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
