"""Quality regression gate: the golden evaluation must not drop below the committed baseline.

If an intentional change moves the metrics, review the new numbers and refresh
the baseline with ``idengine eval --update-baseline``.
"""

from __future__ import annotations

from pathlib import Path

from conftest import GOLDEN_DIR, make_settings
from idengine.evaluation.runner import Evaluator, compare_to_baseline, load_baseline

TOLERANCE = 0.02


def test_golden_metrics_do_not_regress(tmp_path: Path) -> None:
    report = Evaluator(make_settings(tmp_path / "unused")).run(GOLDEN_DIR)
    baseline = load_baseline(GOLDEN_DIR / "baseline.json")
    assert baseline, "baseline.json is missing or empty"
    regressions = compare_to_baseline(report.metrics, baseline, tolerance=TOLERANCE)
    assert not regressions, "\n".join(r.describe() for r in regressions)


def test_quality_floors(tmp_path: Path) -> None:
    """Absolute floors that hold regardless of the baseline file."""
    metrics = Evaluator(make_settings(tmp_path / "unused")).run(GOLDEN_DIR).metrics
    assert metrics["retrieval.hybrid.recall@5"] >= 0.9
    assert metrics["retrieval.hybrid.mrr"] >= metrics["retrieval.semantic.mrr"] - 0.05
    assert metrics["qa.citation_accuracy"] >= 0.9
    assert metrics["qa.abstention_accuracy"] == 1.0
    assert metrics["classification.accuracy"] >= 0.8
