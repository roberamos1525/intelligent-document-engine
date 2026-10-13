"""End-to-end evaluation over a golden dataset, plus baseline regression checks."""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idengine.config import Settings
from idengine.engine import Engine, LabeledExample
from idengine.evaluation.classification import classification_report
from idengine.evaluation.dataset import GoldenDataset, load_dataset
from idengine.evaluation.qa import best_over_golds
from idengine.evaluation.retrieval import (
    dedupe,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from idengine.fsutil import read_json, write_json_atomic
from idengine.search.service import SearchMode

RETRIEVAL_KS = (1, 3, 5, 10)
_CITATION_MARK_RE = re.compile(r"\s*\[\d+(?:,\s*\d+)*\]")


@dataclass(slots=True)
class EvaluationReport:
    dataset: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "metrics": self.metrics,
            "details": self.details,
            "seconds": self.seconds,
        }


@dataclass(frozen=True, slots=True)
class Regression:
    metric: str
    baseline: float
    current: float
    tolerance: float

    def describe(self) -> str:
        return (
            f"{self.metric}: {self.current:.4f} < baseline {self.baseline:.4f} "
            f"- {self.tolerance:.4f}"
        )


def _strip_citations(text: str) -> str:
    return _CITATION_MARK_RE.sub("", text).strip()


class Evaluator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, directory: Path) -> EvaluationReport:
        dataset, root = load_dataset(directory)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="idengine-eval-") as tmp:
            settings = self.settings.model_copy(deep=True)
            settings.storage.data_dir = Path(tmp)
            settings.jobs.workers = 0
            engine = Engine(settings)
            try:
                doc_ids = self._ingest(engine, dataset, root)
                metrics: dict[str, float] = {}
                details: dict[str, Any] = {"documents": len(doc_ids)}
                if dataset.retrieval:
                    retrieval_metrics, retrieval_details = self._retrieval(engine, dataset, doc_ids)
                    metrics.update(retrieval_metrics)
                    details["retrieval"] = retrieval_details
                if dataset.qa:
                    qa_metrics, qa_details = self._qa(engine, dataset, doc_ids)
                    metrics.update(qa_metrics)
                    details["qa"] = qa_details
                if dataset.classification is not None:
                    cls_metrics, cls_details = self._classification(engine, dataset)
                    metrics.update(cls_metrics)
                    details["classification"] = cls_details
            finally:
                engine.close()
        return EvaluationReport(
            dataset=dataset.name,
            metrics={k: round(v, 4) for k, v in sorted(metrics.items())},
            details=details,
            seconds=round(time.perf_counter() - started, 3),
        )

    @staticmethod
    def _ingest(engine: Engine, dataset: GoldenDataset, root: Path) -> dict[str, str]:
        ids: dict[str, str] = {}
        for key, doc in dataset.documents.items():
            outcome, _ = engine.ingest_path(root / doc.file)
            ids[key] = outcome.document.id
        return ids

    @staticmethod
    def _retrieval(
        engine: Engine, dataset: GoldenDataset, doc_ids: dict[str, str]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        key_of = {v: k for k, v in doc_ids.items()}
        metrics: dict[str, float] = {}
        per_query: list[dict[str, Any]] = []
        for mode in SearchMode:
            recalls: dict[int, list[float]] = {k: [] for k in RETRIEVAL_KS}
            precision3: list[float] = []
            rr: list[float] = []
            ndcg: list[float] = []
            for case in dataset.retrieval:
                response = engine.search(
                    case.query, mode=mode, top_k=max(RETRIEVAL_KS), group_by_document=True
                )
                ranked = dedupe([key_of.get(h.document_id, h.document_id) for h in response.hits])
                relevant = set(case.relevant)
                gains = case.graded or dict.fromkeys(case.relevant, 1.0)
                for k in RETRIEVAL_KS:
                    recalls[k].append(recall_at_k(ranked, relevant, k))
                precision3.append(precision_at_k(ranked, relevant, 3))
                rr.append(reciprocal_rank(ranked, relevant))
                ndcg.append(ndcg_at_k(ranked, gains, 10))
                if mode is SearchMode.HYBRID:
                    per_query.append(
                        {
                            "query": case.query,
                            "top3": ranked[:3],
                            "relevant": case.relevant,
                            "rr": round(rr[-1], 4),
                        }
                    )
            for k in RETRIEVAL_KS:
                metrics[f"retrieval.{mode.value}.recall@{k}"] = mean(recalls[k])
            metrics[f"retrieval.{mode.value}.mrr"] = mean(rr)
            metrics[f"retrieval.{mode.value}.precision@3"] = mean(precision3)
            metrics[f"retrieval.{mode.value}.ndcg@10"] = mean(ndcg)
        return metrics, {"queries": per_query}

    @staticmethod
    def _qa(
        engine: Engine, dataset: GoldenDataset, doc_ids: dict[str, str]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        em: list[float] = []
        f1: list[float] = []
        recall: list[float] = []
        citation_hits: list[float] = []
        abstain_correct: list[float] = []
        rows: list[dict[str, Any]] = []
        for case in dataset.qa:
            answer = engine.ask(case.question)
            if not case.answerable:
                abstain_correct.append(float(not answer.answerable))
                rows.append({"question": case.question, "abstained": not answer.answerable})
                continue
            prediction = answer.short_answer or _strip_citations(answer.answer)
            scores = best_over_golds(prediction, case.answers)
            full_scores = best_over_golds(_strip_citations(answer.answer), case.answers)
            em.append(scores["exact_match"])
            f1.append(max(scores["f1"], full_scores["f1"]))
            recall.append(full_scores["answer_recall"])
            if case.document is not None:
                expected = doc_ids[case.document]
                citation_hits.append(
                    float(any(c.document_id == expected for c in answer.citations))
                )
            rows.append(
                {
                    "question": case.question,
                    "prediction": prediction,
                    "f1": round(f1[-1], 4),
                    "answerable": answer.answerable,
                }
            )
        metrics = {
            "qa.exact_match": mean(em),
            "qa.f1": mean(f1),
            "qa.answer_recall": mean(recall),
            "qa.citation_accuracy": mean(citation_hits),
        }
        if abstain_correct:
            metrics["qa.abstention_accuracy"] = mean(abstain_correct)
        return metrics, {"cases": rows}

    @staticmethod
    def _classification(
        engine: Engine, dataset: GoldenDataset
    ) -> tuple[dict[str, float], dict[str, Any]]:
        split = dataset.classification
        assert split is not None
        examples = [LabeledExample(label=e.label, text=e.text) for e in split.train]
        result = engine.train_classifier(examples, test_ratio=0.0)
        predictions = engine.classifier.predict([e.text for e in split.test])
        report = classification_report(
            [e.label for e in split.test], [p.label for p in predictions]
        )
        return (
            {
                "classification.accuracy": report.accuracy,
                "classification.macro_f1": report.macro_f1,
            },
            {"model": result.to_dict(), "report": report.to_dict()},
        )


def compare_to_baseline(
    metrics: dict[str, float], baseline: dict[str, float], *, tolerance: float
) -> list[Regression]:
    """Metrics (all higher-is-better) that dropped more than ``tolerance`` below baseline."""
    regressions: list[Regression] = []
    for name, expected in sorted(baseline.items()):
        current = metrics.get(name)
        if current is None or current < expected - tolerance:
            regressions.append(Regression(name, expected, current or 0.0, tolerance))
    return regressions


def load_baseline(path: Path) -> dict[str, float]:
    data = read_json(path)
    metrics = data.get("metrics", {})
    return {str(k): float(v) for k, v in metrics.items()}


def write_baseline(path: Path, report: EvaluationReport) -> None:
    write_json_atomic(path, {"dataset": report.dataset, "metrics": report.metrics}, indent=2)
