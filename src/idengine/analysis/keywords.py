"""Keyword extraction: RAKE phrases and corpus-aware TF-IDF terms."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from idengine.text.stemmer import stem
from idengine.text.tokenize import STOPWORDS, analyze, words

_PHRASE_SPLIT_RE = re.compile(r"[.,;:!?()\[\]{}\"“”|\n\t/\\]+")
_MAX_PHRASE_WORDS = 4


@dataclass(frozen=True, slots=True)
class Keyword:
    phrase: str
    score: float
    method: str


def _content_runs(text: str) -> list[list[str]]:
    """Maximal runs of content words (no stopwords, digits or punctuation breaks)."""
    runs: list[list[str]] = []
    for fragment in _PHRASE_SPLIT_RE.split(text.lower()):
        current: list[str] = []
        for word in words(fragment):
            word = word.strip(".'-")
            if not word or word in STOPWORDS or word.isdigit() or len(word) < 2:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(word)
        if current:
            runs.append(current)
    return runs


def repeated_phrases(text: str, *, min_count: int = 2, top_k: int = 10) -> list[Keyword]:
    """Multi-word collocations (2-3 content words) that recur within the document.

    Complements RAKE, which discards long candidate runs: "machine learning"
    inside "machine learning models require training" is still found here.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for run in _content_runs(text):
        for n in (2, 3):
            for i in range(len(run) - n + 1):
                counts[tuple(run[i : i + n])] += 1
    scored: dict[str, float] = {}
    for gram, count in counts.items():
        if count < min_count:
            continue
        # Prefer the longer phrase when it is as frequent as its sub-phrase.
        if len(gram) == 2 and any(
            counts[longer] >= count
            for longer in counts
            if len(longer) == 3 and _contains(longer, gram)
        ):
            continue
        scored[" ".join(gram)] = count * len(gram)
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [Keyword(phrase, float(score), "ngram") for phrase, score in ranked[:top_k]]


def _contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    return longer[:2] == shorter or longer[1:] == shorter


def rake(text: str, *, top_k: int = 10) -> list[Keyword]:
    """Rapid Automatic Keyword Extraction (Rose et al., 2010).

    Candidate phrases are maximal runs of non-stopwords; each word scores
    ``degree / frequency`` and a phrase scores the sum of its words.
    """
    phrases = [p for p in _content_runs(text) if len(p) <= _MAX_PHRASE_WORDS]
    if not phrases:
        return []

    frequency: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in phrases:
        for word in phrase:
            frequency[word] += 1
            degree[word] += len(phrase)
    word_score = {w: degree[w] / frequency[w] for w in frequency}

    phrase_scores: dict[str, float] = {}
    phrase_counts: Counter[str] = Counter(" ".join(p) for p in phrases)
    for phrase in phrases:
        key = " ".join(phrase)
        if key not in phrase_scores:
            # Repetition matters for a document's topic; reward it mildly.
            phrase_scores[key] = sum(word_score[w] for w in phrase) * (
                1 + math.log(phrase_counts[key])
            )
    ranked = sorted(phrase_scores.items(), key=lambda item: (-item[1], item[0]))
    return [Keyword(phrase, round(score, 4), "rake") for phrase, score in ranked[:top_k]]


def tfidf_keywords(
    text: str,
    *,
    document_frequency: Callable[[str], int] | None = None,
    total_documents: int = 0,
    top_k: int = 10,
) -> list[Keyword]:
    """Score stemmed terms by TF-IDF against corpus statistics.

    Without corpus statistics the IDF term is constant and this reduces to
    sublinear term frequency. Each stem is reported using its most frequent
    surface form in the document.
    """
    surface: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for word in words(text):
        word = word.strip(".'-")
        if word and word not in STOPWORDS and len(word) > 2 and not word.isdigit():
            surface[stem(word.replace("'", ""))][word] += 1
    counts = Counter(t for t in analyze(text) if t in surface)
    if not counts:
        return []
    scores: dict[str, float] = {}
    for term, tf in counts.items():
        idf = 1.0
        if document_frequency is not None and total_documents > 0:
            df = document_frequency(term)
            idf = math.log((1 + total_documents) / (1 + df)) + 1.0
        scores[term] = (1 + math.log(tf)) * idf
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        Keyword(surface[term].most_common(1)[0][0], round(score, 4), "tfidf")
        for term, score in ranked[:top_k]
    ]


def extract_keywords(
    text: str,
    *,
    top_k: int = 10,
    document_frequency: Callable[[str], int] | None = None,
    total_documents: int = 0,
) -> list[str]:
    """Blend recurring phrases, RAKE phrases and TF-IDF terms without overlaps."""
    phrases = rake(text, top_k=top_k)
    terms = tfidf_keywords(
        text,
        document_frequency=document_frequency,
        total_documents=total_documents,
        top_k=top_k,
    )
    selected: list[str] = []
    seen_stems: set[str] = set()
    for keyword in repeated_phrases(text, top_k=max(1, top_k // 2)):
        _add(keyword.phrase, selected, seen_stems)
    # Interleave so both multi-word phrases and salient single terms appear.
    for pair in zip(phrases, terms, strict=False):
        for keyword in pair:
            _add(keyword.phrase, selected, seen_stems)
    for keyword in phrases[len(terms) :] + terms[len(phrases) :]:
        _add(keyword.phrase, selected, seen_stems)
    return selected[:top_k]


def _add(phrase: str, selected: list[str], seen_stems: set[str]) -> None:
    stems = {stem(w) for w in phrase.split()}
    if stems <= seen_stems:
        return
    selected.append(phrase)
    seen_stems.update(stems)
