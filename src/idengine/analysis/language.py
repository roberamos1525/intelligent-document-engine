"""Language identification with character n-gram profiles (Cavnar & Trenkle, 1994).

Each language profile is the ranked list of the most frequent 1-4 character
n-grams of a sample text shipped in ``analysis/profiles``. A document is
assigned the language whose profile has the smallest "out-of-place" distance
to the document's own profile.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from importlib import resources

PROFILE_SIZE = 400
MAX_NGRAM = 4
MIN_LETTERS = 20
UNDETERMINED = "und"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

_NON_LETTER_RE = re.compile(r"[^\w']+|[\d_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    code: str
    confidence: float
    scores: dict[str, float]

    @property
    def name(self) -> str:
        return LANGUAGE_NAMES.get(self.code, "Undetermined")


def ngram_profile(text: str, size: int = PROFILE_SIZE) -> list[str]:
    """Ranked list of the ``size`` most frequent n-grams (1..MAX_NGRAM) in ``text``."""
    counts: Counter[str] = Counter()
    for token in _NON_LETTER_RE.split(text.lower()):
        if not token:
            continue
        padded = f"_{token}_"
        for n in range(1, MAX_NGRAM + 1):
            for i in range(len(padded) - n + 1):
                gram = padded[i : i + n]
                if gram != "_":
                    counts[gram] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gram for gram, _ in ranked[:size]]


def out_of_place(document: list[str], language: dict[str, int]) -> int:
    """Sum of rank displacements; unknown n-grams get the maximum penalty."""
    penalty = len(language)
    return sum(
        abs(rank - language[gram]) if gram in language else penalty
        for rank, gram in enumerate(document)
    )


@cache
def _profiles() -> dict[str, dict[str, int]]:
    profiles: dict[str, dict[str, int]] = {}
    package = resources.files("idengine.analysis") / "profiles"
    for code in LANGUAGE_NAMES:
        sample = (package / f"{code}.txt").read_text(encoding="utf-8")
        profiles[code] = {gram: rank for rank, gram in enumerate(ngram_profile(sample))}
    return profiles


def detect_language(text: str, *, hint: str | None = None) -> LanguageGuess:
    """Detect the language of ``text``.

    ``hint`` (for example an HTML ``lang`` attribute) is accepted when the
    text is too short to decide, or when the detector's top two candidates
    are close and the hint is one of them.
    """
    sample = text[:20_000]
    letters = sum(1 for ch in sample if ch.isalpha())
    hint = hint.lower() if hint and hint.lower() in LANGUAGE_NAMES else None
    if letters < MIN_LETTERS:
        if hint:
            return LanguageGuess(hint, 0.5, {})
        return LanguageGuess(UNDETERMINED, 0.0, {})

    doc_profile = ngram_profile(sample, PROFILE_SIZE)
    distances = {code: out_of_place(doc_profile, prof) for code, prof in _profiles().items()}
    worst = len(doc_profile) * PROFILE_SIZE or 1
    scores = {code: round(1.0 - d / worst, 4) for code, d in distances.items()}
    ranked = sorted(distances.items(), key=lambda item: item[1])
    best_code, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else worst
    margin = (second - best) / max(second, 1)
    confidence = round(min(1.0, margin * 10), 4)
    if hint and hint != best_code and hint == ranked[1][0] and margin < 0.02:
        return LanguageGuess(hint, confidence, scores)
    return LanguageGuess(best_code, confidence, scores)
