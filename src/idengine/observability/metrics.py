"""A small thread-safe metrics registry with Prometheus text exposition.

Supports counters, gauges and histograms with label sets. This is intentionally
dependency-free; the exposition format follows the Prometheus text format
version 0.0.4 so the ``/metrics`` endpoint can be scraped directly.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _label_key(labels: dict[str, str] | None, allowed: Sequence[str]) -> LabelKey:
    labels = labels or {}
    unknown = set(labels) - set(allowed)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    return tuple((name, str(labels.get(name, ""))) for name in allowed)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(key: LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    pairs = [(k, v) for k, v in key if v != ""] + list(extra)
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs) + "}"


def _format_value(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help_text
        self.label_names = tuple(labels)
        self._lock = threading.Lock()

    def render(self) -> list[str]:
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labels)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        key = _label_key(labels, self.label_names)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(_label_key(labels, self.label_names), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return [f"{self.name}{_format_labels(k)} {_format_value(v)}" for k, v in items]


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labels)
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels, self.label_names)
        with self._lock:
            self._values[key] = float(value)

    def add(self, amount: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels, self.label_names)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(_label_key(labels, self.label_names), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return [f"{self.name}{_format_labels(k)} {_format_value(v)}" for k, v in items]


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, labels)
        if list(buckets) != sorted(buckets):
            raise ValueError("histogram buckets must be sorted")
        self.buckets = tuple(buckets)
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels, self.label_names)
        index = bisect.bisect_left(self.buckets, value)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            counts[index] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    @contextmanager
    def time(self, labels: dict[str, str] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start, labels)

    def count(self, labels: dict[str, str] | None = None) -> int:
        return sum(self._counts.get(_label_key(labels, self.label_names), []))

    def render(self) -> list[str]:
        lines: list[str] = []
        with self._lock:
            snapshot = {k: (list(v), self._sums[k]) for k, v in self._counts.items()}
        for key, (counts, total) in sorted(snapshot.items()):
            cumulative = 0
            for bound, count in zip((*self.buckets, math.inf), counts, strict=True):
                cumulative += count
                le = (("le", _format_value(bound)),)
                lines.append(f"{self.name}_bucket{_format_labels(key, le)} {cumulative}")
            lines.append(f"{self.name}_sum{_format_labels(key)} {_format_value(total)}")
            lines.append(f"{self.name}_count{_format_labels(key)} {cumulative}")
        return lines


class MetricsRegistry:
    """Owns all metrics for one engine instance."""

    def __init__(self, namespace: str = "idengine") -> None:
        self.namespace = namespace
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _register(self, metric: _Metric) -> _Metric:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if type(existing) is not type(metric):
                    raise ValueError(f"metric {metric.name} already registered as {existing.kind}")
                return existing
            self._metrics[metric.name] = metric
            return metric

    def _full(self, name: str) -> str:
        return f"{self.namespace}_{name}" if self.namespace else name

    def counter(self, name: str, help_text: str, labels: Sequence[str] = ()) -> Counter:
        metric = self._register(Counter(self._full(name), help_text, labels))
        assert isinstance(metric, Counter)
        return metric

    def gauge(self, name: str, help_text: str, labels: Sequence[str] = ()) -> Gauge:
        metric = self._register(Gauge(self._full(name), help_text, labels))
        assert isinstance(metric, Gauge)
        return metric

    def histogram(
        self,
        name: str,
        help_text: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        metric = self._register(Histogram(self._full(name), help_text, labels, buckets))
        assert isinstance(metric, Histogram)
        return metric

    def render(self) -> str:
        with self._lock:
            metrics = sorted(self._metrics.values(), key=lambda m: m.name)
        lines: list[str] = []
        for metric in metrics:
            lines.append(f"# HELP {metric.name} {metric.help}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


class EngineMetrics:
    """The concrete metric set used across the engine."""

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()
        r = self.registry
        self.http_requests = r.counter(
            "http_requests_total",
            "HTTP requests by route and status",
            ("method", "route", "status"),
        )
        self.http_latency = r.histogram(
            "http_request_seconds", "HTTP request latency", ("method", "route")
        )
        self.documents_ingested = r.counter(
            "documents_ingested_total",
            "Documents ingested by type and outcome",
            ("type", "outcome"),
        )
        self.chunks_indexed = r.counter("chunks_indexed_total", "Chunks added to the indexes")
        self.ingest_latency = r.histogram("ingest_seconds", "End-to-end ingestion latency")
        self.search_latency = r.histogram("search_seconds", "Search latency by mode", ("mode",))
        self.rag_latency = r.histogram("ask_seconds", "RAG question answering latency")
        self.llm_calls = r.counter("llm_calls_total", "LLM adapter calls", ("provider", "outcome"))
        self.jobs = r.counter("jobs_total", "Job state transitions", ("type", "outcome"))
        self.job_latency = r.histogram("job_seconds", "Job execution time", ("type",))
        self.documents_gauge = r.gauge("documents", "Documents currently stored")
        self.chunks_gauge = r.gauge("chunks", "Chunks currently indexed")
        self.queue_depth = r.gauge("job_queue_depth", "Jobs waiting to run")

    def render(self) -> str:
        return self.registry.render()
