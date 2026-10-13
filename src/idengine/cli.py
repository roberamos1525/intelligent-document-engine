"""Command line interface: ``idengine <command> ...``.

Every command runs against the same data directory as the API (``IDE_DATA_DIR``
or ``--data-dir``). Output is human readable by default and JSON with ``--json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

from idengine import __version__
from idengine.config import Settings, load_settings
from idengine.engine import Engine, LabeledExample
from idengine.errors import ConfigurationError, EngineError, ValidationError
from idengine.observability.logging import configure_logging

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".text"}
DEFAULT_DATASET = Path("tests/fixtures/golden")


class CliContext:
    def __init__(self, args: argparse.Namespace, out: TextIO) -> None:
        self.args = args
        self.out = out
        overrides: dict[str, Any] = {}
        if args.data_dir:
            overrides["storage"] = {"data_dir": args.data_dir}
        self.settings: Settings = load_settings(args.config, overrides=overrides)
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = Engine(self.settings)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()

    def emit(self, payload: Any, human: Callable[[], Iterator[str]] | None = None) -> None:
        if self.args.json or human is None:
            self.out.write(json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n")
            return
        for line in human():
            self.out.write(line + "\n")


def _parse_filter(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"--filter must be JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("--filter must be a JSON object")
    return value


def _iter_inputs(paths: Sequence[str], recursive: bool) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(path.glob(pattern)):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield child
        else:
            yield path


# -- commands -------------------------------------------------------------------------
def cmd_ingest(ctx: CliContext) -> int:
    args = ctx.args
    results: list[dict[str, Any]] = []
    failures = 0
    for path in _iter_inputs(args.paths, args.recursive):
        try:
            outcome, job = ctx.engine.ingest_path(
                path, tags=args.tags or [], category=args.category, background=args.background
            )
        except EngineError as exc:
            failures += 1
            results.append({"file": str(path), "error": exc.code, "message": exc.message})
            continue
        doc = outcome.document
        results.append(
            {
                "file": str(path),
                "document_id": doc.id,
                "status": doc.status.value,
                "duplicate": outcome.duplicate,
                "chunks": doc.chunk_count,
                "title": doc.title,
                "job_id": job.id if job else None,
            }
        )
    if args.background:
        ctx.engine.jobs.drain(timeout=args.timeout)

    def human() -> Iterator[str]:
        for r in results:
            if "error" in r:
                yield f"FAILED  {r['file']}: {r['message']}"
            else:
                flag = " (duplicate)" if r["duplicate"] else ""
                yield (
                    f"{r['status']:<10} {r['document_id']}  {r['chunks']:>3} chunks  "
                    f"{r['file']}{flag}"
                )
        yield f"{len(results) - failures} ingested, {failures} failed"

    ctx.emit(results, human)
    return 1 if failures else 0


def cmd_search(ctx: CliContext) -> int:
    args = ctx.args
    response = ctx.engine.search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        filters=_parse_filter(args.filter),
        group_by_document=args.group,
    )
    payload = {
        "query": response.query,
        "mode": response.mode.value,
        "took_ms": response.took_ms,
        "warnings": response.warnings,
        "hits": [
            {
                "rank": i,
                "score": h.score,
                "document_id": h.document_id,
                "chunk_id": h.chunk_id,
                "title": h.title,
                "source": h.citation_label,
                "snippet": h.snippet,
            }
            for i, h in enumerate(response.hits, start=1)
        ],
    }

    def human() -> Iterator[str]:
        for w in response.warnings:
            yield f"warning: {w}"
        for hit in payload["hits"]:
            yield f"{hit['rank']:>2}. [{hit['score']:.4f}] {hit['source']}"
            yield f"    {hit['snippet']}"
        yield f"{len(response.hits)} hits in {response.took_ms} ms ({response.mode.value})"

    ctx.emit(payload, human)
    return 0


def cmd_ask(ctx: CliContext) -> int:
    answer = ctx.engine.ask(
        ctx.args.question, filters=_parse_filter(ctx.args.filter), top_k=ctx.args.top_k
    )
    payload = {
        "question": answer.question,
        "answer": answer.answer,
        "short_answer": answer.short_answer,
        "answerable": answer.answerable,
        "provider": answer.provider,
        "citations": [
            {"number": c.number, "source": c.source, "document_id": c.document_id}
            for c in answer.citations
        ],
        "took_ms": answer.took_ms,
    }

    def human() -> Iterator[str]:
        yield answer.answer
        if answer.short_answer:
            yield f"Short answer: {answer.short_answer}"
        for c in answer.citations:
            yield f"  [{c.number}] {c.source}"

    ctx.emit(payload, human)
    return 0 if answer.answerable else 2


def cmd_summarize(ctx: CliContext) -> int:
    args = ctx.args
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        summary = ctx.engine.summarize_text(text, method=args.method, max_sentences=args.sentences)
    else:
        if not args.document_id:
            raise ValidationError("give a document id or --file")
        summary = ctx.engine.summarize_document(
            args.document_id, method=args.method, max_sentences=args.sentences
        )
    payload = {
        "method": summary.method.value,
        "provider": summary.provider,
        "summary": summary.text,
    }
    ctx.emit(payload, lambda: iter([summary.text]))
    return 0


def cmd_docs(ctx: CliContext) -> int:
    args = ctx.args
    engine = ctx.engine
    if args.docs_command == "list":
        from idengine.storage import DocumentQuery

        records, total = engine.documents.list(
            DocumentQuery(status=args.status, category=args.category, limit=args.limit)
        )
        payload = [
            {
                "id": r.id,
                "status": r.status.value,
                "type": r.doc_type,
                "title": r.title,
                "category": r.category,
                "chunks": r.chunk_count,
            }
            for r in records
        ]

        def human() -> Iterator[str]:
            for r in payload:
                yield (
                    f"{r['id']}  {r['status']:<10} {r['type']:<8} "
                    f"{(r['category'] or '-'):<12} {r['title']}"
                )
            yield f"{len(payload)} of {total} documents"

        ctx.emit(payload, human)
    elif args.docs_command == "show":
        record = engine.documents.get(args.document_id)
        ctx.emit(
            {
                "id": record.id,
                "filename": record.filename,
                "status": record.status.value,
                "title": record.title,
                "authors": record.authors,
                "document_date": record.document_date,
                "language": record.language,
                "category": record.category,
                "tags": record.tags,
                "near_duplicate_of": record.near_duplicate_of,
                "chunks": record.chunk_count,
                "metadata": record.metadata,
            }
        )
    else:
        engine.ingestion.delete(args.document_id)
        ctx.emit({"deleted": args.document_id}, lambda: iter([f"deleted {args.document_id}"]))
    return 0


def _read_examples(path: Path) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            examples.append(
                LabeledExample(
                    label=str(item["label"]),
                    text=item.get("text"),
                    document_id=item.get("document_id"),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValidationError(f"{path}:{number}: invalid example ({exc})") from exc
    return examples


def cmd_classify(ctx: CliContext) -> int:
    args = ctx.args
    engine = ctx.engine
    if args.classify_command == "train":
        result = engine.train_classifier(
            _read_examples(Path(args.data)),
            algorithm=args.algorithm,
            test_ratio=args.test_ratio,
            folds=args.folds,
        )
        ctx.emit(result.to_dict())
    elif args.classify_command == "predict":
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        if not text:
            raise ValidationError("give text or --file")
        prediction = engine.classify_text(text)
        ctx.emit(
            {
                "label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            },
            lambda: iter([f"{prediction.label} ({prediction.confidence:.2%})"]),
        )
    else:
        prediction = engine.classify_document(args.document_id, apply=not args.dry_run)
        ctx.emit(
            {
                "document_id": args.document_id,
                "label": prediction.label,
                "confidence": prediction.confidence,
            },
            lambda: iter([f"{args.document_id}: {prediction.label} ({prediction.confidence:.2%})"]),
        )
    return 0


def cmd_eval(ctx: CliContext) -> int:
    from idengine.evaluation.runner import (
        Evaluator,
        compare_to_baseline,
        load_baseline,
        write_baseline,
    )

    args = ctx.args
    report = Evaluator(ctx.settings).run(Path(args.dataset))
    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    regressions = []
    baseline_path = Path(args.baseline) if args.baseline else Path(args.dataset) / "baseline.json"
    if args.update_baseline:
        write_baseline(baseline_path, report)
    elif baseline_path.is_file():
        regressions = compare_to_baseline(
            report.metrics, load_baseline(baseline_path), tolerance=args.tolerance
        )
    payload = {
        "dataset": report.dataset,
        "seconds": report.seconds,
        "metrics": report.metrics,
        "regressions": [r.describe() for r in regressions],
    }

    def human() -> Iterator[str]:
        width = max(len(k) for k in report.metrics) if report.metrics else 10
        for name, value in report.metrics.items():
            yield f"{name:<{width}}  {value:.4f}"
        if args.update_baseline:
            yield f"baseline written to {baseline_path}"
        for r in regressions:
            yield f"REGRESSION {r.describe()}"
        yield f"evaluated '{report.dataset}' in {report.seconds}s"

    ctx.emit(payload, human)
    return 1 if regressions else 0


def cmd_jobs(ctx: CliContext) -> int:
    engine = ctx.engine
    if ctx.args.jobs_command == "purge":
        removed = engine.jobs.repository.purge_finished(ctx.args.older_than_days * 86400)
        ctx.emit({"purged": removed}, lambda: iter([f"purged {removed} finished job(s)"]))
        return 0
    if ctx.args.jobs_command == "run":
        ran = engine.jobs.drain(timeout=ctx.args.timeout, include_delayed=ctx.args.wait_retries)
        ctx.emit({"ran": ran}, lambda: iter([f"ran {ran} job(s)"]))
        return 0
    jobs, total = engine.jobs.repository.list(status=ctx.args.status, limit=ctx.args.limit)
    payload = [
        {
            "id": j.id,
            "type": j.type,
            "status": j.status.value,
            "attempts": j.attempts,
            "error": j.error,
        }
        for j in jobs
    ]

    def human() -> Iterator[str]:
        for j in payload:
            yield f"{j['id']}  {j['type']:<16} {j['status']:<10} attempts={j['attempts']}"
        yield f"{len(payload)} of {total} jobs"

    ctx.emit(payload, human)
    return 0


def cmd_reindex(ctx: CliContext) -> int:
    ctx.emit(ctx.engine.reindex(refit=not ctx.args.no_refit))
    return 0


def cmd_stats(ctx: CliContext) -> int:
    ctx.emit(ctx.engine.stats())
    return 0


def cmd_config(ctx: CliContext) -> int:
    ctx.emit(ctx.settings.redacted())
    return 0


def cmd_serve(ctx: CliContext) -> int:  # pragma: no cover - blocking server loop
    import uvicorn

    from idengine.api.app import create_app

    app = create_app(ctx.settings)
    uvicorn.run(app, host=ctx.args.host, port=ctx.args.port, log_config=None)
    return 0


# -- parser -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idengine", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="TOML config file (default: IDE_CONFIG_FILE)")
    parser.add_argument("--data-dir", help="data directory (default: IDE_DATA_DIR or ./var)")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="log to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest files or directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--tags", nargs="*")
    p.add_argument("--category")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--background", action="store_true", help="use the job queue")
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="search indexed chunks")
    p.add_argument("query")
    p.add_argument("--mode", choices=["semantic", "lexical", "hybrid"], default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--filter", help='metadata filter JSON, e.g. {"category": "finance"}')
    p.add_argument("--group", action="store_true", help="best chunk per document")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("--filter")
    p.add_argument("--top-k", type=int)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("summarize", help="summarise a document or a text file")
    p.add_argument("document_id", nargs="?")
    p.add_argument("--file")
    p.add_argument("--method", choices=["textrank", "mapreduce"], default="textrank")
    p.add_argument("--sentences", type=int, default=5)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("docs", help="list, show or delete documents")
    docs = p.add_subparsers(dest="docs_command", required=True)
    d = docs.add_parser("list")
    d.add_argument("--status", choices=["processing", "ready", "failed"])
    d.add_argument("--category")
    d.add_argument("--limit", type=int, default=50)
    d = docs.add_parser("show")
    d.add_argument("document_id")
    d = docs.add_parser("delete")
    d.add_argument("document_id")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("classify", help="train or apply the document classifier")
    cls = p.add_subparsers(dest="classify_command", required=True)
    c = cls.add_parser("train", help="train from a JSONL file of {label, text|document_id}")
    c.add_argument("data")
    c.add_argument("--algorithm", choices=["naive_bayes", "logistic"])
    c.add_argument("--test-ratio", type=float, default=0.2)
    c.add_argument("--folds", type=int, default=0)
    c = cls.add_parser("predict")
    c.add_argument("text", nargs="?")
    c.add_argument("--file")
    c = cls.add_parser("document")
    c.add_argument("document_id")
    c.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("eval", help="evaluate against a golden dataset")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--baseline", help="baseline JSON (default: <dataset>/baseline.json)")
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--output", help="write the full report as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("jobs", help="inspect or run background jobs")
    jobs = p.add_subparsers(dest="jobs_command", required=True)
    j = jobs.add_parser("list")
    j.add_argument("--status", choices=["queued", "running", "succeeded", "failed", "cancelled"])
    j.add_argument("--limit", type=int, default=50)
    j = jobs.add_parser("run", help="run queued jobs inline until the queue is empty")
    j.add_argument("--timeout", type=float, default=600.0)
    j.add_argument("--wait-retries", action="store_true")
    j = jobs.add_parser("purge", help="delete finished jobs older than N days")
    j.add_argument("--older-than-days", type=float, default=30.0)
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("reindex", help="rebuild indexes from the database")
    p.add_argument("--no-refit", action="store_true")
    p.set_defaults(func=cmd_reindex)

    sub.add_parser("stats", help="show engine statistics").set_defaults(func=cmd_stats)
    sub.add_parser("config", help="print effective configuration").set_defaults(func=cmd_config)

    p = sub.add_parser("serve", help="run the HTTP API with uvicorn")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = out or sys.stdout
    try:
        ctx = CliContext(args, stream)
    except ConfigurationError as exc:
        print(f"configuration error: {exc.message}", file=sys.stderr)
        return 3
    if args.verbose or args.command == "serve":
        configure_logging(ctx.settings.logging.level, json_format=ctx.settings.logging.json_format)
    else:
        configure_logging("ERROR", json_format=False)
    try:
        code: int = args.func(ctx)
        return code
    except EngineError as exc:
        if args.json:
            stream.write(json.dumps({"error": exc.to_dict()}) + "\n")
        else:
            print(f"error ({exc.code}): {exc.message}", file=sys.stderr)
        return 1
    finally:
        ctx.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
