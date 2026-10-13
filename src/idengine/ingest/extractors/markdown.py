"""Markdown extractor (CommonMark subset) with YAML-style front matter."""

from __future__ import annotations

import re

from idengine.errors import ExtractionError
from idengine.ingest.detect import decode_text
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument
from idengine.text.normalize import clean_text

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^(=+|-+)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HR_RE = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*(?:-\s*){3,}$|^\s*(?:_\s*){3,}$")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def strip_inline(text: str) -> str:
    """Remove inline Markdown syntax, keeping the visible text."""
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _CODE_SPAN_RE.sub(r"\1", text)
    for _ in range(2):  # nested emphasis such as ***bold italic***
        text = _EMPHASIS_RE.sub(r"\2", text)
    text = _HTML_TAG_RE.sub("", text)
    return text.strip()


def parse_front_matter(text: str) -> tuple[dict[str, str | list[str]], str]:
    """Parse a leading ``---`` block of ``key: value`` pairs.

    Supports scalar values, inline lists (``[a, b]``) and indented ``- item``
    lists. Anything more complex is kept as a raw string.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() in ("---", "..."))
    except StopIteration:
        return {}, text
    meta: dict[str, str | list[str]] = {}
    current_key: str | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s+-\s+(.*)$", line)
        if item and current_key is not None:
            existing = meta.get(current_key)
            values = existing if isinstance(existing, list) else []
            values.append(item.group(1).strip().strip("\"'"))
            meta[current_key] = values
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        current_key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[current_key] = [
                v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()
            ]
        else:
            meta[current_key] = value.strip("\"'")
    return meta, "\n".join(lines[end + 1 :])


def _blocks(body: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    lines = body.split("\n")

    def flush() -> None:
        if paragraph:
            text = strip_inline(" ".join(paragraph))
            if text:
                blocks.append(Block(BlockKind.PARAGRAPH, text))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if _FENCE_RE.match(stripped):
            flush()
            fence = stripped[:3]
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            if any(c.strip() for c in code):
                blocks.append(Block(BlockKind.CODE, "\n".join(code).strip("\n")))
            i += 1
            continue
        if not stripped or _HR_RE.match(stripped):
            flush()
            i += 1
            continue
        atx = _ATX_RE.match(stripped)
        if atx:
            flush()
            title = strip_inline(atx.group(2))
            if title:
                blocks.append(Block(BlockKind.HEADING, title, level=len(atx.group(1))))
            i += 1
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if paragraph == [] and nxt and _SETEXT_RE.match(nxt) and not _LIST_RE.match(line):
            blocks.append(
                Block(BlockKind.HEADING, strip_inline(stripped), level=1 if nxt[0] == "=" else 2)
            )
            i += 2
            continue
        if stripped.startswith("|") and nxt and _TABLE_SEP_RE.match(nxt):
            flush()
            rows: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not _TABLE_SEP_RE.match(row):
                    cells = [strip_inline(c) for c in row.strip("|").split("|")]
                    rows.append(" | ".join(cells))
                i += 1
            blocks.append(Block(BlockKind.TABLE, "\n".join(rows)))
            continue
        list_item = _LIST_RE.match(line)
        if list_item:
            flush()
            blocks.append(Block(BlockKind.LIST_ITEM, strip_inline(list_item.group(1))))
            i += 1
            continue
        paragraph.append(stripped.lstrip("> ").strip() if stripped.startswith(">") else stripped)
        i += 1
    flush()
    return blocks


def extract_markdown(data: bytes) -> ExtractedDocument:
    raw, encoding = decode_text(data)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    meta, body = parse_front_matter(raw)
    blocks = []
    for block in _blocks(body):
        text = block.text if block.kind is BlockKind.CODE else clean_text(block.text)
        if text:
            block.text = text
            blocks.append(block)
    if not blocks:
        raise ExtractionError("document contains no text")

    title_value = meta.get("title")
    title = title_value if isinstance(title_value, str) and title_value else None
    if title is None:
        title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
    authors_value = meta.get("authors", meta.get("author"))
    if isinstance(authors_value, list):
        authors = authors_value
    elif isinstance(authors_value, str) and authors_value:
        authors = [a.strip() for a in re.split(r",|;| and ", authors_value) if a.strip()]
    else:
        authors = []
    created = meta.get("date")
    lang = meta.get("lang", meta.get("language"))
    return ExtractedDocument(
        doc_type=DocType.MARKDOWN,
        blocks=blocks,
        title=title,
        authors=authors,
        created=created if isinstance(created, str) else None,
        language_hint=lang if isinstance(lang, str) else None,
        properties={"encoding": encoding, "front_matter": meta},
    )
