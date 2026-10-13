"""Porter stemming algorithm (M.F. Porter, 1980), implemented from the paper.

Used by the search analyzer so that "invoices", "invoiced" and "invoicing"
share the stem "invoic". Results are memoised because the same tokens repeat
heavily across a corpus.
"""

from __future__ import annotations

from functools import lru_cache

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Number of VC sequences in ``stem`` (Porter's *m*)."""
    m = 0
    i = 0
    n = len(stem)
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_consonant(stem, i):
            i += 1
        m += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """True if word ends consonant-vowel-consonant where the last is not w, x or y."""
    if len(word) < 3:
        return False
    n = len(word)
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _replace(word: str, suffix: str, replacement: str, min_measure: int) -> str | None:
    """Replace ``suffix`` if present and the remaining stem has measure > min_measure.

    Returns the new word, the unchanged word if the suffix matched but the
    condition failed, or ``None`` if the suffix did not match at all.
    """
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_measure:
        return stem + replacement
    return word


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        stem = word[:-3]
        return word[:-1] if _measure(stem) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _has_vowel(stem):
                return word
            word = stem
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)

_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _apply_table(word: str, table: tuple[tuple[str, str], ...]) -> str:
    for suffix, replacement in table:
        result = _replace(word, suffix, replacement, 0)
        if result is not None:
            return result
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        if suffix == "ion" and not stem.endswith(("s", "t")):
            return word
        return stem if _measure(stem) > 1 else word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """Return the Porter stem of a lowercase word."""
    if len(word) <= 2 or not word.isalpha():
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _apply_table(word, _STEP2)
    word = _apply_table(word, _STEP3)
    word = _step4(word)
    return _step5(word)
