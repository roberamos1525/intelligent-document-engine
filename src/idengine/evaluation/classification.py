"""Classification metrics: accuracy, per-class P/R/F1, macro/micro/weighted F1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    per_class: list[ClassMetrics]
    labels: list[str]
    confusion: list[list[int]]  # rows: true label, columns: predicted label
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], *, labels: Sequence[str] | None = None
) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    for truth, pred in zip(y_true, y_pred, strict=True):
        if truth in index and pred in index:
            confusion[index[truth]][index[pred]] += 1

    per_class: list[ClassMetrics] = []
    total_tp = 0
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        predicted = sum(row[i] for row in confusion)
        actual = sum(confusion[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_class.append(
            ClassMetrics(
                label,
                round(precision, 4),
                round(recall, 4),
                round(_f1(precision, recall), 4),
                actual,
            )
        )
        total_tp += tp

    n = len(y_true)
    supported = [m for m in per_class if m.support > 0]
    macro = sum(m.f1 for m in supported) / len(supported) if supported else 0.0
    weighted = sum(m.f1 * m.support for m in per_class) / n if n else 0.0
    accuracy = total_tp / n if n else 0.0
    # For single-label classification micro-F1 equals accuracy over known labels.
    return ClassificationReport(
        accuracy=round(accuracy, 4),
        macro_f1=round(macro, 4),
        micro_f1=round(accuracy, 4),
        weighted_f1=round(weighted, 4),
        per_class=per_class,
        labels=list(labels),
        confusion=confusion,
        samples=n,
    )
