"""Tokenisation, sentence splitting and the search analyzer."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise

from idengine.text.stemmer import stem

STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "may",
        "must",
        "shall",
        "upon",
        "us",
        "via",
        "per",
        "within",
        "without",
        "whether",
        "either",
        "neither",
        "yet",
        "etc",
    ]
)

# Words: letters/digits joined by internal apostrophes, hyphens or dots (e.g. "U.S", "e-mail").
_WORD_RE = re.compile(r"[^\W_]+(?:['’.\-][^\W_]+)*", re.UNICODE)
# Model-agnostic "token" approximation used for chunk budgeting: words and punctuation.
_TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
        "corp", "e.g", "i.e", "no", "fig", "approx", "dept", "est", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "p", "pp", "vol",
    }
)  # fmt: skip

_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'’”)\]]*(?=\s+|$)")


@dataclass(frozen=True, slots=True)
class Span:
    """A substring of a source text with its character offsets."""

    text: str
    start: int
    end: int


def words(text: str) -> list[str]:
    """Lowercased word tokens (no stopword removal, no stemming)."""
    return [m.group(0).lower().replace("’", "'") for m in _WORD_RE.finditer(text)]


def word_spans(text: str) -> Iterator[Span]:
    for m in _WORD_RE.finditer(text):
        yield Span(m.group(0), m.start(), m.end())


def count_tokens(text: str) -> int:
    """Approximate model token count (words plus punctuation marks)."""
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def analyze(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Search analyzer: lowercase, tokenise, drop stopwords, Porter-stem.

    Tokens containing digits are kept verbatim (no stemming) so invoice
    numbers and years still match exactly.
    """
    terms: list[str] = []
    for token in words(text):
        token = token.strip(".'-")
        if not token:
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token):
            terms.append(token)
        elif len(token) > 1:
            terms.append(stem(token.replace("'", "")))
    return terms


def bigrams(terms: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in pairwise(terms)]


def _is_abbreviation(text: str, end: int) -> bool:
    """Whether the period ending at ``end`` (exclusive) belongs to an abbreviation."""
    i = end - 1
    while i >= 0 and text[i] == ".":
        i -= 1
    j = i
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        j -= 1
    candidate = text[j + 1 : i + 1].lower()
    if candidate in _ABBREVIATIONS:
        return True
    # Single capital initials such as "J. Smith".
    return len(candidate) == 1 and text[i].isupper()


def split_sentences(text: str) -> list[Span]:
    """Split text into sentences with character offsets.

    Rule based: a sentence ends at ``.``, ``!`` or ``?`` followed by
    whitespace, unless the period ends a known abbreviation or an initial, or
    the next character is lowercase. Blank lines always end a sentence.
    """
    spans: list[Span] = []
    for block_start, block in _paragraph_blocks(text):
        start = 0
        for m in _SENTENCE_END_RE.finditer(block):
            end = m.end()
            if block[m.start()] == "." and _is_abbreviation(block, m.start() + 1):
                continue
            nxt = block[end:].lstrip()
            if nxt and nxt[0].islower():
                continue
            _append_span(spans, block, block_start, start, end)
            start = end
        _append_span(spans, block, block_start, start, len(block))
    return spans


def _paragraph_blocks(text: str) -> Iterator[tuple[int, str]]:
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        yield pos, text[pos : m.start()]
        pos = m.end()
    yield pos, text[pos:]


def _append_span(spans: list[Span], block: str, offset: int, start: int, end: int) -> None:
    raw = block[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    lead = len(raw) - len(raw.lstrip())
    s = offset + start + lead
    spans.append(Span(" ".join(stripped.split()), s, s + len(stripped)))


def sentences(text: str) -> list[str]:
    return [s.text for s in split_sentences(text)]


def term_frequencies(terms: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    return counts
