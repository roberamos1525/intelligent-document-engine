"""Question-answering overlap metrics in the style of SQuAD."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, gold: str) -> float:
    """Share of gold tokens present in the prediction (lenient for long answers)."""
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 0.0
    pred = Counter(normalize_answer(prediction).split())
    common = Counter(gold_tokens) & pred
    return sum(common.values()) / len(gold_tokens)


def best_over_golds(prediction: str, golds: Sequence[str]) -> dict[str, float]:
    """Max of each metric over alternative gold answers."""
    if not golds:
        return {"exact_match": 0.0, "f1": 0.0, "answer_recall": 0.0}
    return {
        "exact_match": max(exact_match(prediction, g) for g in golds),
        "f1": max(token_f1(prediction, g) for g in golds),
        "answer_recall": max(answer_recall(prediction, g) for g in golds),
    }
