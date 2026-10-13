"""Heuristic structure recovery for unstructured text (plain text and PDF pages).

Plain text has no markup, so headings are inferred from typographic
conventions: setext underlines, numbered section titles, short ALL-CAPS
lines and short Title Case lines standing on their own.
"""

from __future__ import annotations

import re

from idengine.ingest.model import Block, BlockKind
from idengine.text.normalize import clean_text

_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z].{1,90})$"
)
_LIST_ITEM_RE = re.compile(r"^(?:[-*+•]\s+|\(?\d{1,3}[.)]\s+|\(?[a-z][.)]\s+)(?P<body>.+)$")
_SETEXT_RE = re.compile(r"^(=+|-+)\s*$")
_TERMINAL_PUNCT = (".", ":", ";", ",", "?", "!")
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs"}
)


def _title_case_ratio(line: str) -> float:
    tokens = [t for t in re.findall(r"[A-Za-z][\w'-]*", line) if t.lower() not in _SMALL_WORDS]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t[0].isupper()) / len(tokens)


def heading_level(line: str, *, isolated: bool) -> int | None:
    """Return a heading level for ``line`` or ``None`` if it reads like body text.

    ``isolated`` means the line is followed by a blank line or another block
    boundary, which makes Title Case detection much more reliable.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 100 or stripped.endswith(_TERMINAL_PUNCT):
        return None
    numbered = _NUMBERED_HEADING_RE.match(stripped)
    if numbered and len(stripped.split()) <= 12:
        return min(numbered.group("num").count(".") + 1, 6)
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) >= 3 and all(ch.isupper() for ch in letters) and len(stripped.split()) <= 8:
        return 1
    words = stripped.split()
    if isolated and 1 <= len(words) <= 10 and _title_case_ratio(stripped) >= 0.75:
        return 2
    return None


def blocks_from_plain_text(text: str, *, page: int | None = None) -> list[Block]:
    """Convert raw text into structural blocks."""
    text = clean_text(text)
    lines = text.split("\n")
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(BlockKind.PARAGRAPH, " ".join(paragraph), page=page))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not line:
            flush()
            i += 1
            continue
        setext = _SETEXT_RE.match(nxt) if nxt else None
        if setext and len(line) <= 100 and not paragraph:
            blocks.append(
                Block(BlockKind.HEADING, line, level=1 if nxt[0] == "=" else 2, page=page)
            )
            i += 2
            continue
        list_match = _LIST_ITEM_RE.match(line)
        if list_match and not _NUMBERED_HEADING_RE.match(line):
            flush()
            blocks.append(Block(BlockKind.LIST_ITEM, list_match.group("body").strip(), page=page))
            i += 1
            continue
        # Only the first line of a block can be a heading; a line that starts a
        # block is "isolated" from preceding text by construction.
        level = heading_level(line, isolated=True) if not paragraph else None
        if level is not None:
            blocks.append(Block(BlockKind.HEADING, line, level=level, page=page))
            i += 1
            continue
        paragraph.append(line)
        if line.endswith((".", "?", "!")) and nxt and heading_level(nxt, isolated=True):
            flush()
        i += 1
    flush()
    return blocks
