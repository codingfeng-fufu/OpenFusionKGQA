"""Tests for the API runtime benchmark script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_benchmark():
    module_path = REPO_ROOT / "scripts" / "benchmark_api_runtime.py"
    spec = importlib.util.spec_from_file_location("benchmark_api_runtime", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_latency_percentiles_without_leaking_token():
    benchmark = _load_benchmark()
    calls = []
    clock_values = iter([0.00, 0.01, 0.01, 0.03, 0.03, 0.06, 0.06, 0.10])

    def fake_clock():
        return next(clock_values)

    def fake_request(url, *, method, headers, timeout, json_body):
        calls.append((url, method, dict(headers), timeout, json_body))
        return benchmark.HttpResponse(200, '{"status":"healthy"}', "application/json")

    report = benchmark.run_benchmark(
        "http://127.0.0.1:8000",
        path="/healthz",
        request_count=4,
        concurrency=1,
        token="secret-token",
        timeout=1.5,
        max_p95_ms=100.0,
        min_success_rate=1.0,
        http_request=fake_request,
        clock=fake_clock,
    )

    assert report["status"] == "passed"
    assert report["summary"]["attempted"] == 4
    assert report["summary"]["succeeded"] == 4
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["latency_ms"]["p50"] == 20.0
    assert report["summary"]["latency_ms"]["p95"] == 40.0
    assert report["summary"]["latency_ms"]["p99"] == 40.0
    assert all(call[2]["Authorization"] == "Bearer secret-token" for call in calls)
    assert "secret-token" not in json.dumps(report, ensure_ascii=False)


def test_benchmark_fails_when_latency_threshold_is_exceeded():
    benchmark = _load_benchmark()
    clock_values = iter([0.00, 0.20, 0.20, 0.45])

    def fake_clock():
        return next(clock_values)

    def fake_request(url, *, method, headers, timeout, json_body):
        return benchmark.HttpResponse(200, "ok", "text/plain")

    report = benchmark.run_benchmark(
        "http://127.0.0.1:8000",
        path="/healthz",
        request_count=2,
        concurrency=1,
        max_p95_ms=100.0,
        min_success_rate=1.0,
        http_request=fake_request,
        clock=fake_clock,
    )

    assert report["status"] == "failed"
    threshold_results = {item["name"]: item for item in report["threshold_results"]}
    assert threshold_results["max_p95_ms"]["status"] == "failed"
    assert threshold_results["min_success_rate"]["status"] == "passed"


def test_benchmark_records_http_and_exception_failures():
    benchmark = _load_benchmark()
    clock_values = iter([0.00, 0.01, 0.01, 0.02, 0.02, 0.03])
    responses = iter(
        [
            benchmark.HttpResponse(200, "ok", "text/plain"),
            benchmark.HttpResponse(503, "not ready", "text/plain"),
            OSError("connection refused"),
        ]
    )

    def fake_clock():
        return next(clock_values)

    def fake_request(url, *, method, headers, timeout, json_body):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    report = benchmark.run_benchmark(
        "http://127.0.0.1:8000",
        path="/readyz",
        request_count=3,
        concurrency=1,
        min_success_rate=0.9,
        http_request=fake_request,
        clock=fake_clock,
    )

    assert report["status"] == "failed"
    assert report["summary"]["succeeded"] == 1
    assert report["summary"]["failed"] == 2
    assert report["summary"]["error_count"] == 2
    assert report["errors"]["http_status_503"] == 1
    assert report["errors"]["OSError"] == 1


def test_benchmark_main_supports_ask_payload_and_json_exit_code(capsys, monkeypatch):
    benchmark = _load_benchmark()

    def fake_run_benchmark(**kwargs):
        assert kwargs["base_url"] == "http://127.0.0.1:8000"
        assert kwargs["path"] == "/ask"
        assert kwargs["method"] == "POST"
        assert kwargs["json_body"] == {"question": "GraphRAG 是什么？"}
        assert kwargs["token"] == "env-token"
        return {
            "status": "failed",
            "auth": "set",
            "summary": {"attempted": 1},
        }

    monkeypatch.setenv("KGQA_API_AUTH_TOKEN", "env-token")
    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    exit_code = benchmark.main(
        [
            "--base-url",
            "http://127.0.0.1:8000",
            "--ask-question",
            "GraphRAG 是什么？",
            "--requests",
            "1",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "env-token" not in json.dumps(payload, ensure_ascii=False)
