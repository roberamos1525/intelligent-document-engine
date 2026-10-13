"""Token-aware, heading-aware chunking with sentence-level overlap.

The chunker works on sentence-sized *units*. Units are packed greedily into
chunks up to ``max_tokens``; when a chunk is emitted the next one starts with
the trailing units of the previous chunk (up to ``overlap_tokens``) so that
facts spanning a boundary remain retrievable. With ``respect_headings`` a
chunk never spans two sections and records its heading path, which is later
used for citations and prepended to the text that gets embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from idengine.config import ChunkingSettings
from idengine.ingest.model import Block, BlockKind, ExtractedDocument, Section
from idengine.text.tokenize import count_tokens, split_sentences

_TOKEN_SPAN_RE = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    text: str
    token_count: int
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None

    @property
    def section(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None

    def index_text(self) -> str:
        """Text used for embeddings/BM25: heading context plus body."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    tokens: int
    block_id: int
    page: int | None


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Hard-split text into windows of at most ``max_tokens`` approximate tokens."""
    spans = [m.span() for m in _TOKEN_SPAN_RE.finditer(text)]
    if len(spans) <= max_tokens:
        return [text.strip()] if text.strip() else []
    pieces: list[str] = []
    for i in range(0, len(spans), max_tokens):
        window = spans[i : i + max_tokens]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append(piece)
    return pieces


class Chunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    # -- unit preparation ------------------------------------------------------
    def _units_for_block(self, block: Block, block_id: int) -> list[_Unit]:
        limit = self.settings.max_tokens
        if block.kind is BlockKind.TABLE:
            parts = block.text.split("\n")
        elif block.kind is BlockKind.CODE:
            parts = [block.text] if count_tokens(block.text) <= limit else block.text.split("\n")
        else:
            total = count_tokens(block.text)
            parts = (
                [block.text]
                if total <= limit and block.kind is BlockKind.LIST_ITEM
                else [s.text for s in split_sentences(block.text)]
            )
        units: list[_Unit] = []
        for part in parts:
            tokens = count_tokens(part)
            if tokens == 0:
                continue
            pieces = [part] if tokens <= limit else split_by_tokens(part, limit)
            for piece in pieces:
                units.append(_Unit(piece, count_tokens(piece), block_id, block.page))
        return units

    def _sections(self, document: ExtractedDocument) -> list[Section]:
        if self.settings.respect_headings:
            return document.sections()
        # Headings become ordinary text so nothing is lost when flattening.
        blocks = [
            Block(BlockKind.PARAGRAPH, b.text, page=b.page) if b.is_heading else b
            for b in document.blocks
        ]
        return [Section(heading_path=(), blocks=blocks)]

    # -- packing -------------------------------------------------------------
    @staticmethod
    def _join(units: list[_Unit]) -> str:
        parts: list[str] = []
        previous: int | None = None
        for unit in units:
            if previous is not None:
                parts.append(" " if unit.block_id == previous else "\n\n")
            parts.append(unit.text)
            previous = unit.block_id
        return "".join(parts)

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        budget = self.settings.overlap_tokens
        tail: list[_Unit] = []
        used = 0
        for unit in reversed(units):
            if used + unit.tokens > budget:
                break
            tail.insert(0, unit)
            used += unit.tokens
        # Never carry over the whole previous chunk, or packing would not advance.
        return tail if len(tail) < len(units) else []

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        limit = self.settings.max_tokens
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        fresh = 0  # units in `current` that were not carried over as overlap
        for unit in units:
            if current and fresh and current_tokens + unit.tokens > limit:
                groups.append(current)
                current = self._overlap_tail(current)
                current_tokens = sum(u.tokens for u in current)
                fresh = 0
                while current and current_tokens + unit.tokens > limit:
                    current_tokens -= current.pop(0).tokens
            current.append(unit)
            current_tokens += unit.tokens
            fresh += 1
        if current and fresh:
            groups.append(current)
        # Fold a tiny trailing chunk into its predecessor when it fits.
        if len(groups) >= 2:
            last_tokens = sum(u.tokens for u in groups[-1])
            if last_tokens < self.settings.min_tokens:
                prev = groups[-2]
                carried = {id(u) for u in prev}
                merged = prev + [u for u in groups[-1] if id(u) not in carried]
                if sum(u.tokens for u in merged) <= limit:
                    groups[-2:] = [merged]
        return groups

    def chunk(self, document: ExtractedDocument) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        block_id = 0
        for section in self._sections(document):
            units: list[_Unit] = []
            for block in section.blocks:
                units.extend(self._units_for_block(block, block_id))
                block_id += 1
            for group in self._pack(units):
                text = self._join(group)
                pages = [u.page for u in group if u.page is not None]
                drafts.append(
                    ChunkDraft(
                        ordinal=len(drafts),
                        text=text,
                        token_count=count_tokens(text),
                        heading_path=section.heading_path,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                    )
                )
        return drafts
