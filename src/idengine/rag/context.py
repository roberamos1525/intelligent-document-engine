"""Context assembly: numbered passages within a token budget, plus citation records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from idengine.llm.base import Passage
from idengine.rag.rerank import RankedChunk
from idengine.text.tokenize import count_tokens, split_sentences


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: str
    document_id: str
    title: str | None
    source: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    snippet: str
    score: float


@dataclass(slots=True)
class AssembledContext:
    passages: list[Passage]
    citations: list[Citation]
    tokens: int
    truncated: bool


def _truncate_to_budget(text: str, budget: int) -> str:
    """Keep whole leading sentences that fit in ``budget`` tokens."""
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        tokens = count_tokens(sentence.text)
        if used + tokens > budget:
            break
        kept.append(sentence.text)
        used += tokens
    return " ".join(kept)


def assemble_context(chunks: Sequence[RankedChunk], *, token_budget: int) -> AssembledContext:
    """Number passages 1..n in rank order until the token budget is exhausted.

    A passage that does not fit entirely is cut at a sentence boundary when
    at least a quarter of the remaining budget can be used; otherwise
    assembly stops.
    """
    passages: list[Passage] = []
    citations: list[Citation] = []
    used = 0
    truncated = False
    for ranked in chunks:
        hit = ranked.hit
        remaining = token_budget - used
        if remaining <= 0:
            truncated = True
            break
        text = hit.text
        tokens = count_tokens(text)
        if tokens > remaining:
            truncated = True
            if remaining < token_budget // 4:
                break
            text = _truncate_to_budget(text, remaining)
            if not text:
                break
            tokens = count_tokens(text)
        number = len(passages) + 1
        passages.append(Passage(number, text, hit.citation_label, ranked.score))
        citations.append(
            Citation(
                number=number,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                title=hit.title,
                source=hit.citation_label,
                page_start=hit.page_start,
                page_end=hit.page_end,
                heading_path=hit.heading_path,
                snippet=hit.snippet,
                score=ranked.score,
            )
        )
        used += tokens
    return AssembledContext(passages, citations, used, truncated)
