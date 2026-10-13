"""Classifier lifecycle: training, evaluation, persistence and prediction."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idengine.classify.model import Prediction, TextClassifier, cross_validate, stratified_split
from idengine.config import ClassifierSettings
from idengine.errors import ModelNotTrainedError, ValidationError
from idengine.observability.logging import get_logger
from idengine.storage.db import utcnow

log = get_logger("classify")

MAX_TRAINING_EXAMPLES = 50_000


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_id: str
    algorithm: str
    labels: list[str]
    examples: int
    holdout: dict[str, Any] | None
    cross_validation: dict[str, Any] | None
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "labels": self.labels,
            "examples": self.examples,
            "holdout": self.holdout,
            "cross_validation": self.cross_validation,
            "seconds": self.seconds,
        }


class ClassifierService:
    def __init__(self, settings: ClassifierSettings, model_dir: Path) -> None:
        self.settings = settings
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._classifier: TextClassifier | None = None
        self._info: dict[str, Any] = {}

    def load(self) -> bool:
        loaded = TextClassifier.load(self.model_dir)
        if loaded is None:
            return False
        with self._lock:
            self._classifier, self._info = loaded
        log.info("classifier loaded", extra={"model_id": self._info.get("model_id")})
        return True

    @property
    def is_trained(self) -> bool:
        return self._classifier is not None

    def status(self) -> dict[str, Any]:
        if self._classifier is None:
            return {"trained": False}
        return {"trained": True, **self._info}

    def train(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        algorithm: str | None = None,
        test_ratio: float = 0.2,
        folds: int = 0,
        seed: int = 0,
    ) -> TrainingResult:
        if len(texts) > MAX_TRAINING_EXAMPLES:
            raise ValidationError(f"at most {MAX_TRAINING_EXAMPLES} training examples are allowed")
        cleaned = [(t, lbl.strip()) for t, lbl in zip(texts, labels, strict=True) if t.strip()]
        if any(not lbl or len(lbl) > 64 for _, lbl in cleaned):
            raise ValidationError("labels must be 1-64 characters")
        texts = [t for t, _ in cleaned]
        labels = [lbl for _, lbl in cleaned]
        settings = self.settings
        if algorithm is not None:
            if algorithm not in ("naive_bayes", "logistic"):
                raise ValidationError(f"unknown algorithm '{algorithm}'")
            settings = settings.model_copy(update={"algorithm": algorithm})

        started = time.perf_counter()
        holdout: dict[str, Any] | None = None
        counts = Counter(labels)
        if test_ratio > 0 and min(counts.values(), default=0) >= 2 and len(texts) >= 8:
            train_x, train_y, test_x, test_y = stratified_split(
                texts, labels, test_ratio=test_ratio, seed=seed
            )
            if test_x:
                probe = TextClassifier(settings).fit(train_x, train_y)
                holdout = probe.evaluate(test_x, test_y).to_dict()
        cv = cross_validate(settings, texts, labels, folds=folds, seed=seed) if folds >= 2 else None

        # The deployed model is trained on all examples.
        classifier = TextClassifier(settings).fit(texts, labels)
        result = TrainingResult(
            model_id=f"clf_{uuid.uuid4().hex[:12]}",
            algorithm=classifier.algorithm,
            labels=classifier.labels,
            examples=len(texts),
            holdout=holdout,
            cross_validation=cv,
            seconds=round(time.perf_counter() - started, 3),
        )
        info = {
            "model_id": result.model_id,
            "algorithm": result.algorithm,
            "labels": result.labels,
            "examples": result.examples,
            "label_counts": dict(sorted(counts.items())),
            "trained_at": utcnow(),
            "holdout": holdout,
            "cross_validation": cv,
        }
        classifier.save(self.model_dir, info)
        with self._lock:
            self._classifier, self._info = classifier, info
        log.info("classifier trained", extra={"model_id": result.model_id, "examples": len(texts)})
        return result

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        classifier = self._classifier
        if classifier is None:
            raise ModelNotTrainedError("no classifier has been trained yet")
        return classifier.predict(texts)

    def predict_one(self, text: str) -> Prediction | None:
        """Prediction for auto-classification, or None if untrained/below threshold."""
        if self._classifier is None or not text.strip():
            return None
        prediction = self._classifier.predict([text])[0]
        if prediction.confidence < self.settings.min_confidence:
            return None
        return prediction
