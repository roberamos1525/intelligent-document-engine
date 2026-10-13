from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from idengine.errors import ConfigurationError
from idengine.evaluation.dataset import load_dataset
from idengine.evaluation.qa import (
    answer_recall,
    best_over_golds,
    exact_match,
    normalize_answer,
    token_f1,
)
from idengine.evaluation.retrieval import (
    dcg_at_k,
    dedupe,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from idengine.evaluation.runner import compare_to_baseline


class TestRetrievalMetrics:
    RANKED = ["d3", "d1", "d7", "d2"]

    def test_recall_precision_rr(self) -> None:
        relevant = {"d1", "d2"}
        assert recall_at_k(self.RANKED, relevant, 1) == 0.0
        assert recall_at_k(self.RANKED, relevant, 2) == 0.5
        assert recall_at_k(self.RANKED, relevant, 4) == 1.0
        assert recall_at_k(self.RANKED, set(), 4) == 0.0
        assert precision_at_k(self.RANKED, relevant, 2) == 0.5
        assert precision_at_k(self.RANKED, relevant, 0) == 0.0
        assert reciprocal_rank(self.RANKED, relevant) == 0.5
        assert reciprocal_rank(self.RANKED, {"zz"}) == 0.0

    def test_ndcg(self) -> None:
        gains = {"d1": 2.0, "d2": 1.0}
        expected_dcg = 3 / math.log2(3) + 1 / math.log2(5)
        assert dcg_at_k(self.RANKED, gains, 4) == pytest.approx(expected_dcg)
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(self.RANKED, gains, 4) == pytest.approx(expected_dcg / ideal)
        assert ndcg_at_k(["d1", "d2"], gains, 10) == pytest.approx(1.0)
        assert ndcg_at_k(self.RANKED, {}, 4) == 0.0

    def test_helpers(self) -> None:
        assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
        assert mean([]) == 0.0 and mean([1.0, 2.0]) == 1.5


class TestQaMetrics:
    def test_normalisation_and_exact_match(self) -> None:
        assert normalize_answer("The  Answer, is: 42!") == "answer is 42"
        assert exact_match("the $500", "$500.") == 1.0
        assert exact_match("500 dollars", "$500") == 0.0

    def test_token_f1_and_recall(self) -> None:
        assert token_f1("within 45 days", "45 days") == pytest.approx(0.8)
        assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0
        assert token_f1("cats", "dogs") == 0.0
        assert answer_recall("payment due within 45 days of invoice", "45 days") == 1.0
        assert answer_recall("anything", "") == 0.0

    def test_best_over_golds(self) -> None:
        scores = best_over_golds("Delaware", ["State of Delaware", "Delaware"])
        assert scores == {"exact_match": 1.0, "f1": 1.0, "answer_recall": 1.0}
        assert best_over_golds("x", [])["f1"] == 0.0


class TestDatasetAndBaseline:
    def test_golden_dataset_loads(self) -> None:
        dataset, root = load_dataset(Path(__file__).parents[1] / "fixtures" / "golden")
        assert len(dataset.documents) == 12
        assert dataset.classification is not None
        assert (root / dataset.documents["f_invoice"].file).is_file()

    def test_dataset_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_dataset(tmp_path)
        (tmp_path / "dataset.json").write_text(
            json.dumps(
                {
                    "name": "bad",
                    "documents": {"a": {"file": "a.txt"}},
                    "retrieval": [{"query": "q", "relevant": ["missing"]}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="unknown"):
            load_dataset(tmp_path)
        (tmp_path / "dataset.json").write_text(
            json.dumps({"name": "x", "documents": {"a": {"file": "a.txt"}}}), encoding="utf-8"
        )
        with pytest.raises(ConfigurationError, match="corpus file"):
            load_dataset(tmp_path)

    def test_compare_to_baseline(self) -> None:
        baseline = {"m1": 0.9, "m2": 0.5, "m3": 0.7}
        current = {"m1": 0.89, "m2": 0.4}
        regressions = compare_to_baseline(current, baseline, tolerance=0.02)
        assert [r.metric for r in regressions] == ["m2", "m3"]
        assert "m2: 0.4000 < baseline 0.5000" in regressions[0].describe()
        assert compare_to_baseline({"m": 1.0}, {"m": 0.5}, tolerance=0.0) == []
