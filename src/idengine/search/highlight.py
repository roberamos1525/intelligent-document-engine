"""Query-focused snippets with highlight offsets."""

from __future__ import annotations

from dataclasses import dataclass

from idengine.text.stemmer import stem
from idengine.text.tokenize import STOPWORDS, word_spans


@dataclass(frozen=True, slots=True)
class Snippet:
    text: str
    highlights: list[tuple[int, int]]


def _normalise(token: str) -> str:
    token = token.lower().strip(".'-")
    return token if any(ch.isdigit() for ch in token) else stem(token)


def make_snippet(text: str, query_terms: set[str], *, window_words: int = 40) -> Snippet:
    """Pick the window of ``window_words`` words with the most query-term hits.

    ``query_terms`` are analysed (stemmed) terms. Highlight offsets refer to
    the returned snippet text.
    """
    spans = list(word_spans(text))
    if not spans:
        return Snippet(text[:300], [])
    hits = [
        i
        for i, span in enumerate(spans)
        if span.text.lower() not in STOPWORDS and _normalise(span.text) in query_terms
    ]
    if len(spans) <= window_words:
        start_word, end_word = 0, len(spans)
    else:
        best_start, best_count = 0, -1
        left = 0
        for right in range(len(hits)):
            while hits[right] - hits[left] >= window_words:
                left += 1
            count = right - left + 1
            if count > best_count:
                best_count, best_start = count, hits[left]
        # Centre the window slightly before the first hit for context.
        start_word = max(0, min(best_start - window_words // 5, len(spans) - window_words))
        end_word = start_word + window_words
    char_start = 0 if start_word == 0 else spans[start_word].start
    char_end = len(text) if end_word >= len(spans) else spans[end_word - 1].end
    body = text[char_start:char_end].replace("\n", " ")
    prefix = "" if char_start == 0 else "… "
    suffix = "" if char_end >= len(text) else " …"
    snippet = prefix + body + suffix
    highlights = [
        (spans[i].start - char_start + len(prefix), spans[i].end - char_start + len(prefix))
        for i in hits
        if start_word <= i < end_word
    ]
    return Snippet(snippet, highlights)
