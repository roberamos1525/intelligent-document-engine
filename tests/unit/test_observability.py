from __future__ import annotations

import json
import logging

import pytest

from idengine.api.security import TokenBucket
from idengine.observability.logging import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    request_id_var,
)
from idengine.observability.metrics import EngineMetrics, MetricsRegistry


class TestMetrics:
    def test_counter_gauge_histogram_exposition(self) -> None:
        registry = MetricsRegistry(namespace="t")
        counter = registry.counter("hits_total", "Hits", ("route",))
        gauge = registry.gauge("depth", "Depth")
        histogram = registry.histogram("latency_seconds", "Latency", buckets=(0.1, 1.0))
        counter.inc(labels={"route": "/a"})
        counter.inc(2, labels={"route": '/b"x'})
        gauge.set(5)
        gauge.add(-2)
        histogram.observe(0.05)
        histogram.observe(0.5)
        histogram.observe(5)
        text = registry.render()
        assert "# TYPE t_hits_total counter" in text
        assert 't_hits_total{route="/a"} 1' in text
        assert 't_hits_total{route="/b\\"x"} 2' in text
        assert "t_depth 3" in text
        assert 't_latency_seconds_bucket{le="0.1"} 1' in text
        assert 't_latency_seconds_bucket{le="1"} 2' in text
        assert 't_latency_seconds_bucket{le="+Inf"} 3' in text
        assert "t_latency_seconds_count 3" in text
        assert "t_latency_seconds_sum 5.55" in text
        assert histogram.count() == 3 and gauge.value() == 3

    def test_validation(self) -> None:
        registry = MetricsRegistry()
        counter = registry.counter("c", "c", ("a",))
        with pytest.raises(ValueError):
            counter.inc(-1)
        with pytest.raises(ValueError):
            counter.inc(labels={"zzz": "1"})
        assert registry.counter("c", "c", ("a",)) is counter
        with pytest.raises(ValueError):
            registry.gauge("c", "c")
        with pytest.raises(ValueError):
            registry.histogram("h", "h", buckets=(2.0, 1.0))

    def test_histogram_timer_and_engine_metrics(self) -> None:
        metrics = EngineMetrics()
        with metrics.search_latency.time({"mode": "hybrid"}):
            pass
        assert metrics.search_latency.count({"mode": "hybrid"}) == 1
        assert "idengine_search_seconds_bucket" in metrics.render()


class TestLogging:
    def _record(self, **extra: object) -> logging.LogRecord:
        record = logging.LogRecord(
            "idengine.test", logging.INFO, __file__, 1, "hello %s", ("x",), None
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_json_formatter_includes_context_and_extras(self) -> None:
        token = request_id_var.set("req-1")
        try:
            payload = json.loads(JsonFormatter().format(self._record(document_id="doc_1")))
        finally:
            request_id_var.reset(token)
        assert payload["msg"] == "hello x"
        assert payload["request_id"] == "req-1"
        assert payload["document_id"] == "doc_1"
        assert payload["level"] == "INFO" and payload["ts"].endswith("Z")

    def test_text_formatter_and_configuration(self) -> None:
        token = request_id_var.set("abc")
        try:
            assert TextFormatter().format(self._record()).endswith("[req=abc]")
        finally:
            request_id_var.reset(token)
        configure_logging("DEBUG", json_format=False)
        configure_logging("WARNING", json_format=True)
        logger = logging.getLogger("idengine")
        assert logger.level == logging.WARNING
        assert sum(1 for h in logger.handlers if getattr(h, "_idengine", False)) == 1


def test_token_bucket_refills_over_time() -> None:
    now = [0.0]
    bucket = TokenBucket(rate=2.0, capacity=2, clock=lambda: now[0])
    assert bucket.take("k") == 0.0
    assert bucket.take("k") == 0.0
    wait = bucket.take("k")
    assert wait == pytest.approx(0.5)
    now[0] += 0.5
    assert bucket.take("k") == 0.0
    assert bucket.take("other") == 0.0  # independent buckets per key
    assert TokenBucket(rate=0.0, capacity=1, clock=lambda: 0.0).take("x") == 0.0
