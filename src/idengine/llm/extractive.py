"""Deterministic, offline "LLM": extractive answering and summarisation.

Answers are built from the context sentences that best match the question
(IDF-weighted term overlap, phrase matches, the retrieval rank of the source
passage, and whether the sentence contains the kind of entity the question
asks for). Every sentence is cited. When nothing matches well enough the
adapter declines to answer instead of guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from idengine.analysis.entities import EntityExtractor
from idengine.llm.base import LLMAdapter, LLMRequest, LLMResponse, Task
from idengine.llm.prompts import NOT_FOUND_ANSWER
from idengine.summarize.textrank import textrank_summary
from idengine.text.tokenize import analyze, bigrams, split_sentences

_ANSWER_TYPES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(e-?mail)\b", re.I), ("EMAIL",)),
    (re.compile(r"\b(phone|telephone)\b", re.I), ("PHONE",)),
    (re.compile(r"\b(website|url|web address|link)\b", re.I), ("URL",)),
    (re.compile(r"\b(percent|percentage)\b", re.I), ("PERCENT",)),
    # Interrogatives outrank topic nouns: "how many weeks of paid leave" is a count and
    # "who approved the budget" asks for a person, not an amount.
    (re.compile(r"\b(how (many|long)|number of|status code|error code)\b", re.I), ("NUMBER",)),
    (re.compile(r"\bwho\b", re.I), ("PERSON", "ORG")),
    (re.compile(r"\b(when|what date|which date)\b", re.I), ("DATE", "DURATION")),
    (
        re.compile(
            r"\b(where|which (state|country|city)|jurisdiction|which law|governing law)\b", re.I
        ),
        ("LOCATION",),
    ),
    (
        re.compile(
            r"\b(how much|amount|cost|price|fee|fees|total|budget|salary|paid|pay|value|revenue"
            r"|rent|deposit|charge|penalty|stipend|allowance|cap)\b",
            re.I,
        ),
        ("MONEY",),
    ),
    (re.compile(r"\b(deadline|due date|effective date|date)\b", re.I), ("DATE", "DURATION")),
)
_UNITS = r"(?:business\s+|calendar\s+|working\s+)?(?:days?|weeks?|months?|years?|hours?|minutes?)"
_NUMBER = r"\d+(?:[.,]\d+)?"
_NUMBER_RE = re.compile(
    rf"\b(?:between\s+{_NUMBER}\s+and\s+{_NUMBER}|from\s+{_NUMBER}\s+to\s+{_NUMBER}"
    rf"|{_NUMBER}\s*(?:-|to)\s*{_NUMBER}|{_NUMBER})(?:\s+{_UNITS})?\b",
    re.I,
)
_DURATION_RE = re.compile(
    rf"\b(?:within|after|before|in|for|up to|at least|no later than)\s+(?:{_NUMBER}|one|two|three|"
    rf"four|five|six|seven|ten|thirty|sixty|ninety)\s+{_UNITS}\b",
    re.I,
)
_MIN_OVERLAP = 0.2
_FOLLOW_UP_RATIO = 0.6


@dataclass(slots=True)
class _Candidate:
    passage: int
    position: int
    text: str
    score: float
    overlap: float
    typed_values: list[str]


def expected_answer_types(question: str) -> tuple[str, ...]:
    for pattern, labels in _ANSWER_TYPES:
        if pattern.search(question):
            return labels
    return ()


class ExtractiveLLM(LLMAdapter):
    name = "extractive"
    model = "extractive-v1"

    def __init__(self, *, entity_extractor: EntityExtractor | None = None) -> None:
        self.entities = entity_extractor or EntityExtractor()

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.task is Task.SUMMARIZE:
            return self._summarize(request)
        return self._answer(request)

    # -- answering ---------------------------------------------------------------
    def _typed_values(self, sentence: str, types: tuple[str, ...]) -> list[str]:
        if not types:
            return []
        values: list[str] = []
        if "NUMBER" in types:
            values.extend(m.group(0) for m in _NUMBER_RE.finditer(sentence))
        entity_types = set(types) - {"NUMBER", "DURATION"}
        if entity_types:
            values.extend(
                e.text for e in self.entities.extract(sentence) if e.label in entity_types
            )
        if "DURATION" in types and not values:
            # Relative deadlines ("within 30 days") answer "when" if no date exists.
            values.extend(m.group(0) for m in _DURATION_RE.finditer(sentence))
        return values

    def _candidates(self, request: LLMRequest) -> list[_Candidate]:
        question_terms = set(analyze(request.instruction))
        question_bigrams = set(bigrams(analyze(request.instruction)))
        types = expected_answer_types(request.instruction)
        sentences: list[tuple[int, int, str, list[str]]] = []
        for passage in request.passages:
            for position, span in enumerate(split_sentences(passage.text)):
                sentences.append((passage.number, position, span.text, analyze(span.text)))
        if not sentences or not question_terms:
            return []
        df = Counter(term for *_, terms in sentences for term in set(terms))
        n = len(sentences)
        idf = {t: math.log(1 + n / (1 + df.get(t, 0))) + 0.5 for t in question_terms}
        total_weight = sum(idf.values())
        best_rank = {p.number: rank for rank, p in enumerate(request.passages)}

        candidates: list[_Candidate] = []
        for number, position, text, terms in sentences:
            term_set = set(terms)
            matched = question_terms & term_set
            overlap = sum(idf[t] for t in matched) / total_weight
            phrase_bonus = 0.1 * len(question_bigrams & set(bigrams(terms)))
            typed = self._typed_values(text, types)
            type_bonus = 0.25 if typed and matched else 0.0
            rank_bonus = 0.08 / (1 + best_rank.get(number, 0))
            # Very long sentences match more terms by chance; dampen slightly.
            length_penalty = 0.05 * max(0.0, math.log(max(len(terms), 1) / 25))
            score = overlap + phrase_bonus + type_bonus + rank_bonus - length_penalty
            candidates.append(_Candidate(number, position, text, score, overlap, typed))
        return candidates

    def _short_answer(self, best: _Candidate, question: str) -> str | None:
        if not best.typed_values:
            return None
        if len(best.typed_values) == 1:
            return best.typed_values[0]
        # Prefer the value closest to a question term in the sentence.
        question_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        lowered = best.text.lower()
        anchors = [m.start() for w in question_words for m in re.finditer(re.escape(w), lowered)]
        if not anchors:
            return best.typed_values[0]

        def distance(value: str) -> int:
            pos = best.text.find(value)
            return min(abs(pos - a) for a in anchors) if pos >= 0 else 10**9

        return min(best.typed_values, key=distance)

    def _answer(self, request: LLMRequest) -> LLMResponse:
        candidates = self._candidates(request)
        if not candidates:
            return LLMResponse(NOT_FOUND_ANSWER, self.name, self.model, answerable=False)
        ranked = sorted(candidates, key=lambda c: (-c.score, c.passage, c.position))
        best = ranked[0]
        if best.overlap < _MIN_OVERLAP:
            return LLMResponse(NOT_FOUND_ANSWER, self.name, self.model, answerable=False)
        chosen = [best]
        for candidate in ranked[1:]:
            if len(chosen) >= request.max_sentences:
                break
            if candidate.overlap <= 0 or candidate.score < _FOLLOW_UP_RATIO * best.score:
                break
            if any(candidate.text == c.text for c in chosen):
                continue
            chosen.append(candidate)
        chosen.sort(key=lambda c: (c.passage, c.position))
        text = " ".join(f"{c.text} [{c.passage}]" for c in chosen)
        citations = list(dict.fromkeys(c.passage for c in chosen))
        return LLMResponse(
            text,
            self.name,
            self.model,
            citations=citations,
            short_answer=self._short_answer(best, request.instruction),
            extra={"confidence": round(min(1.0, best.score), 4)},
        )

    # -- summarisation -------------------------------------------------------------
    def _summarize(self, request: LLMRequest) -> LLMResponse:
        sentences: list[tuple[int, str]] = []
        for passage in request.passages:
            sentences.extend((passage.number, s.text) for s in split_sentences(passage.text))
        if not sentences:
            return LLMResponse("", self.name, self.model, answerable=False)
        ranked = textrank_summary([s for _, s in sentences], max_sentences=request.max_sentences)
        chosen = [sentences[r.index] for r in ranked]
        return LLMResponse(
            " ".join(text for _, text in chosen),
            self.name,
            self.model,
            citations=list(dict.fromkeys(number for number, _ in chosen)),
        )
