"""Document summarisation: TextRank (local) and map-reduce via the LLM adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from idengine.errors import ValidationError
from idengine.llm.base import LLMAdapter, LLMRequest, Passage, Task
from idengine.summarize.textrank import textrank_summary
from idengine.text.tokenize import count_tokens


class SummaryMethod(StrEnum):
    TEXTRANK = "textrank"
    MAP_REDUCE = "mapreduce"


@dataclass(slots=True)
class Summary:
    method: SummaryMethod
    text: str
    sentences: list[str]
    provider: str
    stages: list[dict[str, Any]] = field(default_factory=list)


class MapReduceSummarizer:
    """Summarise each chunk ("map"), then summarise the summaries ("reduce").

    If the combined partial summaries exceed ``reduce_token_budget`` they are
    reduced in groups recursively ("collapse"), so arbitrarily long documents
    are handled with bounded prompt sizes.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        map_sentences: int = 2,
        reduce_token_budget: int = 1200,
        max_depth: int = 4,
    ) -> None:
        self.llm = llm
        self.map_sentences = map_sentences
        self.reduce_token_budget = reduce_token_budget
        self.max_depth = max_depth

    def _call(self, texts: Sequence[str], max_sentences: int, instruction: str) -> str:
        passages = [Passage(i + 1, t, f"part {i + 1}") for i, t in enumerate(texts)]
        request = LLMRequest(Task.SUMMARIZE, instruction, passages, max_sentences=max_sentences)
        return self.llm.generate(request).text.strip()

    def _groups(self, texts: Sequence[str]) -> list[list[str]]:
        groups: list[list[str]] = [[]]
        used = 0
        for text in texts:
            tokens = count_tokens(text)
            if groups[-1] and used + tokens > self.reduce_token_budget:
                groups.append([])
                used = 0
            groups[-1].append(text)
            used += tokens
        return groups

    def summarize(
        self, chunks: Sequence[str], *, max_sentences: int
    ) -> tuple[str, list[dict[str, Any]]]:
        if not chunks:
            return "", []
        stages: list[dict[str, Any]] = []
        partials = [
            self._call([chunk], self.map_sentences, "Summarize this section.")
            for chunk in chunks
            if chunk.strip()
        ]
        partials = [p for p in partials if p]
        stages.append({"stage": "map", "inputs": len(chunks), "outputs": len(partials)})
        depth = 0
        while (
            sum(count_tokens(p) for p in partials) > self.reduce_token_budget
            and depth < self.max_depth
        ):
            groups = self._groups(partials)
            if len(groups) <= 1:
                break
            partials = [
                self._call(group, self.map_sentences + 1, "Combine these partial summaries.")
                for group in groups
            ]
            depth += 1
            stages.append({"stage": f"collapse-{depth}", "outputs": len(partials)})
        final = self._call(partials, max_sentences, "Write a summary of the whole document.")
        stages.append({"stage": "reduce", "inputs": len(partials)})
        return final, stages


class Summarizer:
    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm
        self.map_reduce = MapReduceSummarizer(llm)

    def summarize(
        self,
        chunks: Sequence[str],
        *,
        method: SummaryMethod | str = SummaryMethod.TEXTRANK,
        max_sentences: int = 5,
    ) -> Summary:
        if not 1 <= max_sentences <= 50:
            raise ValidationError("max_sentences must be between 1 and 50")
        method = SummaryMethod(method)
        if method is SummaryMethod.TEXTRANK:
            ranked = textrank_summary("\n\n".join(chunks), max_sentences=max_sentences)
            sentences = [r.text for r in ranked]
            return Summary(method, " ".join(sentences), sentences, "textrank")
        text, stages = self.map_reduce.summarize(chunks, max_sentences=max_sentences)
        return Summary(method, text, [text] if text else [], self.llm.name, stages)
