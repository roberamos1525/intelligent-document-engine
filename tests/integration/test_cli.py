from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from builders import INVOICE_TEXT, POLICY_MARKDOWN, build_docx
from conftest import GOLDEN_DIR
from idengine.cli import build_parser, main


def run(data_dir: Path, *args: str, json_output: bool = True) -> tuple[int, Any]:
    out = io.StringIO()
    argv = ["--data-dir", str(data_dir), *(["--json"] if json_output else []), *args]
    code = main(argv, out=out)
    text = out.getvalue()
    if json_output and text.strip():
        return code, json.loads(text)
    return code, text


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "nested").mkdir(parents=True)
    (folder / "invoice.txt").write_text(INVOICE_TEXT, encoding="utf-8")
    (folder / "policy.md").write_text(POLICY_MARKDOWN, encoding="utf-8")
    (folder / "nested" / "contract.docx").write_bytes(
        build_docx([("h1", "Delivery"), ("p", "The supplier delivers within 10 business days.")])
    )
    (folder / "ignored.bin").write_bytes(b"\x00\x01")
    return folder


def test_ingest_search_ask_and_docs(tmp_path: Path, corpus: Path) -> None:
    data = tmp_path / "data"
    code, results = run(data, "ingest", str(corpus), "--recursive", "--tags", "demo")
    assert code == 0 and len(results) == 3
    assert all(r["status"] == "ready" for r in results)

    code, again = run(data, "ingest", str(corpus / "invoice.txt"))
    assert again[0]["duplicate"] is True

    code, search = run(data, "search", "invoice late fee", "--top-k", "2", "--group")
    assert code == 0 and search["hits"][0]["title"] == "ACME WIDGETS INVOICE"

    code, filtered = run(data, "search", "delivery", "--filter", '{"doc_type": "docx"}')
    assert all("contract" in h["source"].lower() or h["title"] for h in filtered["hits"])

    code, answer = run(data, "ask", "What is the total amount due?")
    assert code == 0 and answer["short_answer"] == "$12,450.00"
    code, _ = run(data, "ask", "Who won the 1998 chess olympiad?")
    assert code == 2  # not answerable

    code, listing = run(data, "docs", "list")
    assert len(listing) == 3
    doc_id = next(d["id"] for d in listing if d["type"] == "markdown")
    code, shown = run(data, "docs", "show", doc_id)
    assert shown["authors"] == ["Jane Doe", "John Roe"]
    code, summary = run(data, "summarize", doc_id, "--sentences", "1")
    assert summary["summary"]
    code, deleted = run(data, "docs", "delete", doc_id)
    assert deleted == {"deleted": doc_id}


def test_human_readable_output(tmp_path: Path, corpus: Path) -> None:
    data = tmp_path / "data"
    code, text = run(data, "ingest", str(corpus / "invoice.txt"), json_output=False)
    assert code == 0 and "1 ingested, 0 failed" in text
    code, text = run(data, "search", "invoice", json_output=False)
    assert "hits in" in text and "1." in text
    code, text = run(data, "ask", "What is the total amount due?", json_output=False)
    assert "Short answer: $12,450.00" in text and "[1]" in text
    code, text = run(data, "docs", "list", json_output=False)
    assert "1 of 1 documents" in text


def test_background_ingest_and_jobs(tmp_path: Path, corpus: Path) -> None:
    data = tmp_path / "data"
    code, results = run(data, "ingest", str(corpus / "policy.md"), "--background")
    assert code == 0 and results[0]["job_id"]
    code, jobs = run(data, "jobs", "list", "--status", "succeeded")
    assert len(jobs) == 1 and jobs[0]["type"] == "ingest"
    code, ran = run(data, "jobs", "run")
    assert ran == {"ran": 0}
    code, kept = run(data, "jobs", "purge")  # default: older than 30 days
    assert kept == {"purged": 0}
    code, purged = run(data, "jobs", "purge", "--older-than-days", "-1")
    assert purged == {"purged": 1}


def test_classifier_commands(tmp_path: Path, corpus: Path) -> None:
    data = tmp_path / "data"
    dataset = json.loads((GOLDEN_DIR / "dataset.json").read_text(encoding="utf-8"))
    training = tmp_path / "train.jsonl"
    training.write_text(
        "\n".join(json.dumps(e) for e in dataset["classification"]["train"]) + "\n\n",
        encoding="utf-8",
    )
    code, result = run(data, "classify", "train", str(training), "--folds", "3")
    assert code == 0 and result["examples"] == 40 and result["cross_validation"]["folds"] == 3
    code, prediction = run(data, "classify", "predict", "Pay the invoice before the due date.")
    assert prediction["label"] == "finance"
    code, results = run(data, "ingest", str(corpus / "policy.md"))
    doc_id = results[0]["document_id"]
    code, applied = run(data, "classify", "document", doc_id, "--dry-run")
    assert applied["label"] == "hr"
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    code, error = run(data, "classify", "train", str(bad))
    assert code == 1 and error["error"]["code"] == "validation_error"


def test_eval_with_baseline_and_regression_detection(tmp_path: Path) -> None:
    dataset = tmp_path / "golden"
    shutil.copytree(GOLDEN_DIR, dataset)
    data = tmp_path / "data"
    report_path = tmp_path / "report.json"
    code, report = run(data, "eval", "--dataset", str(dataset), "--output", str(report_path))
    assert code == 0 and report["regressions"] == []
    assert json.loads(report_path.read_text(encoding="utf-8"))["details"]["documents"] == 12

    baseline = dataset / "baseline.json"
    inflated = json.loads(baseline.read_text(encoding="utf-8"))
    inflated["metrics"]["qa.f1"] = 1.5  # impossible target -> regression
    baseline.write_text(json.dumps(inflated), encoding="utf-8")
    code, report = run(data, "eval", "--dataset", str(dataset))
    assert code == 1 and report["regressions"][0].startswith("qa.f1")

    code, report = run(data, "eval", "--dataset", str(dataset), "--update-baseline")
    assert code == 0
    assert json.loads(baseline.read_text(encoding="utf-8"))["metrics"]["qa.f1"] <= 1.0


def test_stats_config_reindex_and_errors(tmp_path: Path) -> None:
    data = tmp_path / "data"
    code, stats = run(data, "stats")
    assert code == 0 and stats["documents"] == 0
    code, config = run(data, "config")
    assert config["storage"]["data_dir"] == str(data)
    code, rebuilt = run(data, "reindex", "--no-refit")
    assert rebuilt["chunks"] == 0
    code, error = run(data, "docs", "show", "doc_missing")
    assert code == 1 and error["error"]["code"] == "not_found"
    code, error = run(data, "search", "x", "--filter", "[1, 2]")
    assert code == 1 and "JSON object" in error["error"]["message"]
    code, error = run(data, "summarize")
    assert code == 1
    missing_config = main(["--config", str(tmp_path / "nope.toml"), "stats"], out=io.StringIO())
    assert missing_config == 3


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
