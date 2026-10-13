"""Text cleaning and normalisation applied after extraction."""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": " - ",
        "…": "...",
        " ": " ",
        "•": "-",
        "­": "",
    }
)

# Control characters except tab and newline.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿]")
# "infor-\nmation" -> "information" (only when both sides are lowercase letters).
_HYPHEN_BREAK_RE = re.compile(r"([a-z])-\n([a-z])")
_SPACES_RE = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LIGATURES)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _CONTROL_RE.sub(" ", text)


def dehyphenate(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines within paragraphs while keeping paragraph breaks.

    A line break is kept when the next line looks like a list item, a heading
    candidate (short line without terminal punctuation followed by a blank
    line is handled by the extractor) or when the current line ends a
    sentence and the next starts with an uppercase letter after a short line.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        merged: list[str] = [lines[0]]
        for line in lines[1:]:
            if _starts_list_item(line) or _starts_list_item(merged[-1]):
                merged.append(line)
            else:
                merged[-1] = f"{merged[-1]} {line}"
        out.append("\n".join(merged))
    return "\n\n".join(out)


def _starts_list_item(line: str) -> bool:
    return bool(re.match(r"^(?:[-*+]\s|\d{1,3}[.)]\s|[a-z][.)]\s)", line))


def clean_text(text: str, *, unwrap: bool = False) -> str:
    """Full cleaning pipeline used for every extracted block of text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_unicode(text)
    text = dehyphenate(text)
    if unwrap:
        text = unwrap_lines(text)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def fingerprint_text(text: str) -> str:
    """Canonical form used for the normalised-content hash (format independent)."""
    text = normalize_unicode(text).lower()
    return " ".join(re.findall(r"[^\W_]+", text))
