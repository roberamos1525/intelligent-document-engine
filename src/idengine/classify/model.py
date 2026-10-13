"""Trainable text classifier: vectoriser + algorithm + label set, with persistence."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from idengine.classify.algorithms import Model, MultinomialNaiveBayes, SoftmaxRegression
from idengine.classify.vectorizer import Vectorizer
from idengine.config import ClassifierSettings
from idengine.errors import IndexCorruptedError, ValidationError
from idengine.evaluation.classification import ClassificationReport, classification_report
from idengine.fsutil import read_json, write_json_atomic

MODEL_FORMAT = 1


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class TextClassifier:
    def __init__(self, settings: ClassifierSettings) -> None:
        self.settings = settings
        self.vectorizer = Vectorizer(max_features=settings.max_features)
        self.model: Model = self._new_model()
        self.labels: list[str] = []
        self.loss_history: list[float] = []

    def _new_model(self) -> Model:
        if self.settings.algorithm == "logistic":
            return SoftmaxRegression(
                l2=self.settings.l2,
                epochs=self.settings.epochs,
                learning_rate=self.settings.learning_rate,
            )
        return MultinomialNaiveBayes(alpha=self.settings.alpha)

    @property
    def algorithm(self) -> str:
        return self.model.name

    def _features(self, texts: Sequence[str]) -> np.ndarray:
        if self.model.name == "naive_bayes":
            return self.vectorizer.counts(texts)
        return self.vectorizer.tfidf(texts)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TextClassifier:
        if len(texts) != len(labels):
            raise ValidationError("texts and labels must have the same length")
        distinct = sorted(set(labels))
        if len(distinct) < 2:
            raise ValidationError("training data needs at least two distinct labels")
        if len(texts) < 4:
            raise ValidationError("training data needs at least four examples")
        self.labels = distinct
        index = {label: i for i, label in enumerate(distinct)}
        self.vectorizer.fit(texts)
        if not self.vectorizer.vocabulary:
            raise ValidationError("training texts produced an empty vocabulary")
        y = np.array([index[label] for label in labels], dtype=np.int64)
        self.model = self._new_model()
        self.loss_history = self.model.fit(self._features(texts), y, len(distinct))
        return self

    @property
    def is_trained(self) -> bool:
        return bool(self.labels)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not self.is_trained:
            raise ValidationError("classifier is not trained")
        probs = self.model.predict_proba(self._features(texts))
        predictions: list[Prediction] = []
        for row in probs:
            best = int(np.argmax(row))
            predictions.append(
                Prediction(
                    label=self.labels[best],
                    confidence=round(float(row[best]), 4),
                    probabilities={
                        label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)
                    },
                )
            )
        return predictions

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> ClassificationReport:
        predicted = [p.label for p in self.predict(texts)]
        return classification_report(
            labels, predicted, labels=sorted(set(self.labels) | set(labels))
        )

    # -- persistence -----------------------------------------------------------
    def save(self, directory: Path, info: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "classifier.npz", idf=self.vectorizer.idf, **self.model.arrays())
        write_json_atomic(
            directory / "classifier.json",
            {
                "format": MODEL_FORMAT,
                "algorithm": self.model.name,
                "labels": self.labels,
                "vectorizer": self.vectorizer.state(),
                "settings": self.settings.model_dump(),
                "info": info,
            },
        )

    @classmethod
    def load(cls, directory: Path) -> tuple[TextClassifier, dict[str, Any]] | None:
        manifest = read_json(directory / "classifier.json")
        if not manifest:
            return None
        if manifest.get("format") != MODEL_FORMAT:
            raise IndexCorruptedError("unsupported classifier model format")
        try:
            settings = ClassifierSettings.model_validate(manifest["settings"])
            with np.load(directory / "classifier.npz") as data:
                arrays = {name: data[name] for name in data.files}
            classifier = cls(settings)
            classifier.vectorizer = Vectorizer.from_state(manifest["vectorizer"], arrays["idf"])
            if manifest["algorithm"] == "logistic":
                classifier.model = SoftmaxRegression.from_arrays(arrays)
            else:
                classifier.model = MultinomialNaiveBayes.from_arrays(arrays, settings.alpha)
            classifier.labels = list(manifest["labels"])
        except (KeyError, ValueError, OSError) as exc:
            raise IndexCorruptedError(f"classifier model is unreadable: {exc}") from exc
        return classifier, dict(manifest.get("info", {}))


def stratified_split(
    texts: Sequence[str], labels: Sequence[str], *, test_ratio: float, seed: int = 0
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split keeping each label's proportion; every label keeps >=1 training example."""
    if not 0.0 < test_ratio < 1.0:
        raise ValidationError("test_ratio must be between 0 and 1")
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        n_test = int(len(indices) * test_ratio + 0.5)  # round half up, not to even
        n_test = min(n_test, len(indices) - 1)
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    train_idx.sort()
    test_idx.sort()
    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
    )


def cross_validate(
    settings: ClassifierSettings,
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified k-fold cross-validation; returns mean/stdev of accuracy and macro-F1."""
    if folds < 2:
        raise ValidationError("folds must be at least 2")
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    assignment = [0] * len(labels)
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        for position, i in enumerate(indices):
            assignment[i] = position % folds
    accuracies: list[float] = []
    macro: list[float] = []
    for fold in range(folds):
        train = [i for i in range(len(labels)) if assignment[i] != fold]
        test = [i for i in range(len(labels)) if assignment[i] == fold]
        if not test or len({labels[i] for i in train}) < 2:
            continue
        model = TextClassifier(settings).fit([texts[i] for i in train], [labels[i] for i in train])
        report = model.evaluate([texts[i] for i in test], [labels[i] for i in test])
        accuracies.append(report.accuracy)
        macro.append(report.macro_f1)
    if not accuracies:
        raise ValidationError("not enough data for cross-validation")
    return {
        "folds": len(accuracies),
        "accuracy_mean": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "macro_f1_mean": round(float(np.mean(macro)), 4),
        "macro_f1_std": round(float(np.std(macro)), 4),
    }
