from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from idengine.classify import ClassifierService, TextClassifier
from idengine.classify.algorithms import MultinomialNaiveBayes, SoftmaxRegression, softmax
from idengine.classify.model import cross_validate, stratified_split
from idengine.classify.vectorizer import Vectorizer
from idengine.config import ClassifierSettings
from idengine.errors import IndexCorruptedError, ModelNotTrainedError, ValidationError
from idengine.evaluation.classification import classification_report

GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "golden" / "dataset.json").read_text(encoding="utf-8")
)["classification"]
TRAIN_TEXTS = [e["text"] for e in GOLDEN["train"]]
TRAIN_LABELS = [e["label"] for e in GOLDEN["train"]]
TEST_TEXTS = [e["text"] for e in GOLDEN["test"]]
TEST_LABELS = [e["label"] for e in GOLDEN["test"]]


class TestAlgorithms:
    def test_softmax_rows_sum_to_one(self) -> None:
        probs = softmax(np.array([[1000.0, 1000.0], [0.0, np.log(3.0)]]))
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.allclose(probs[1], [0.25, 0.75])

    def test_naive_bayes_on_counts(self) -> None:
        x = np.array([[3, 0], [2, 0], [0, 4], [0, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(x, y, 3)  # class 2 never seen: must not produce -inf
        probs = model.predict_proba(np.array([[5, 0], [0, 5]], dtype=float))
        assert probs.argmax(axis=1).tolist() == [0, 1]
        assert np.isfinite(model.class_log_prior).all()
        with pytest.raises(ValueError):
            MultinomialNaiveBayes(alpha=0)

    def test_softmax_regression_converges(self) -> None:
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        model = SoftmaxRegression(epochs=400, learning_rate=1.0)
        history = model.fit(x, y, 2)
        assert history[-1] < history[0]
        assert (model.predict_proba(x).argmax(axis=1) == y).mean() == 1.0


class TestVectorizer:
    def test_vocabulary_counts_and_tfidf(self) -> None:
        vec = Vectorizer(max_features=50).fit(["red apple", "green apple", "red car"])
        assert "appl" in vec.vocabulary and "red_appl" in vec.vocabulary
        counts = vec.counts(["red red apple"])
        assert counts[0, vec.vocabulary["red"]] == 2
        tfidf = vec.tfidf(["red apple", "unknown words"])
        assert np.isclose(np.linalg.norm(tfidf[0]), 1.0) and not tfidf[1].any()
        restored = Vectorizer.from_state(vec.state(), vec.idf)
        assert restored.vocabulary == vec.vocabulary
        with pytest.raises(ValueError):
            Vectorizer.from_state(vec.state(), vec.idf[:-1])

    def test_feature_cap(self) -> None:
        vec = Vectorizer(max_features=3, use_bigrams=False).fit(["a1 b2 c3 d4", "a1 b2", "a1"])
        assert sorted(vec.vocabulary) == ["a1", "b2", "c3"]


class TestTextClassifier:
    @pytest.mark.parametrize("algorithm", ["naive_bayes", "logistic"])
    def test_accuracy_on_golden_split(self, algorithm: str) -> None:
        settings = ClassifierSettings(algorithm=algorithm)  # type: ignore[arg-type]
        model = TextClassifier(settings).fit(TRAIN_TEXTS, TRAIN_LABELS)
        report = model.evaluate(TEST_TEXTS, TEST_LABELS)
        assert report.accuracy >= 0.85
        prediction = model.predict(["The invoice is overdue and late fees apply."])[0]
        assert prediction.label == "finance"
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        model = TextClassifier(ClassifierSettings(algorithm="logistic")).fit(
            TRAIN_TEXTS, TRAIN_LABELS
        )
        model.save(tmp_path, {"model_id": "m1"})
        loaded = TextClassifier.load(tmp_path)
        assert loaded is not None
        restored, info = loaded
        assert info == {"model_id": "m1"}
        assert [p.label for p in restored.predict(TEST_TEXTS)] == [
            p.label for p in model.predict(TEST_TEXTS)
        ]
        assert TextClassifier.load(tmp_path / "none") is None
        (tmp_path / "classifier.json").write_text('{"format": 99}', encoding="utf-8")
        with pytest.raises(IndexCorruptedError):
            TextClassifier.load(tmp_path)

    def test_training_validation(self) -> None:
        model = TextClassifier(ClassifierSettings())
        with pytest.raises(ValidationError):
            model.predict(["x"])
        with pytest.raises(ValidationError, match="two distinct"):
            model.fit(["a b", "c d", "e f", "g h"], ["x"] * 4)
        with pytest.raises(ValidationError, match="four"):
            model.fit(["a", "b"], ["x", "y"])
        with pytest.raises(ValidationError, match="same length"):
            model.fit(["a"], ["x", "y"])
        with pytest.raises(ValidationError, match="vocabulary"):
            model.fit(["the", "and", "of", "a"], ["x", "y", "x", "y"])


class TestSplitsAndReports:
    def test_stratified_split_keeps_every_label(self) -> None:
        train_x, train_y, test_x, test_y = stratified_split(
            TRAIN_TEXTS, TRAIN_LABELS, test_ratio=0.25, seed=3
        )
        assert len(test_x) == 12 and len(train_x) == 28
        assert set(train_y) == set(test_y) == set(TRAIN_LABELS)
        with pytest.raises(ValidationError):
            stratified_split(TRAIN_TEXTS, TRAIN_LABELS, test_ratio=1.5)

    def test_cross_validation(self) -> None:
        result = cross_validate(ClassifierSettings(), TRAIN_TEXTS, TRAIN_LABELS, folds=5)
        assert result["folds"] == 5 and result["accuracy_mean"] >= 0.5  # 4 classes: chance is 0.25
        with pytest.raises(ValidationError):
            cross_validate(ClassifierSettings(), TRAIN_TEXTS, TRAIN_LABELS, folds=1)

    def test_classification_report_values(self) -> None:
        report = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        per = {m.label: m for m in report.per_class}
        assert report.accuracy == 0.75
        assert per["a"].precision == 1.0 and per["a"].recall == 0.5
        assert per["b"].precision == pytest.approx(0.6667, abs=1e-4)
        assert report.confusion == [[1, 1], [0, 2]]
        assert report.macro_f1 == pytest.approx((0.6667 + 0.8) / 2, abs=1e-3)
        with pytest.raises(ValueError):
            classification_report(["a"], [])


class TestService:
    def test_train_predict_and_reload(self, tmp_path: Path) -> None:
        service = ClassifierService(ClassifierSettings(), tmp_path)
        assert not service.is_trained and service.status() == {"trained": False}
        assert service.predict_one("anything") is None
        with pytest.raises(ModelNotTrainedError):
            service.predict(["x"])
        result = service.train(TRAIN_TEXTS, TRAIN_LABELS, test_ratio=0.25, folds=3)
        assert result.holdout is not None and result.holdout["accuracy"] >= 0.5
        assert result.cross_validation is not None
        assert service.status()["examples"] == 40
        reloaded = ClassifierService(ClassifierSettings(), tmp_path)
        assert reloaded.load() and reloaded.status()["model_id"] == result.model_id

    def test_min_confidence_and_validation(self, tmp_path: Path) -> None:
        service = ClassifierService(ClassifierSettings(min_confidence=0.999), tmp_path)
        service.train(TRAIN_TEXTS, TRAIN_LABELS, test_ratio=0.0)
        assert service.predict_one("vague text about nothing in particular") is None
        with pytest.raises(ValidationError):
            service.train(TRAIN_TEXTS, ["x" * 65] * 40)
        with pytest.raises(ValidationError):
            service.train(TRAIN_TEXTS, TRAIN_LABELS, algorithm="svm")
