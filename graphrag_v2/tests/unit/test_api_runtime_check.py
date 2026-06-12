"""Tests for the API runtime live-check script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_check():
    module_path = REPO_ROOT / "scripts" / "check_api_runtime.py"
    spec = importlib.util.spec_from_file_location("check_api_runtime", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_api_runtime_check_passes_ready_runtime_without_leaking_token():
    check = _load_check()
    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, dict(headers), timeout))
        if url.endswith("/healthz"):
            return check.HttpResponse(200, '{"status":"healthy"}', "application/json")
        if url.endswith("/readyz"):
            return check.HttpResponse(200, '{"status":"ready","checks":{}}', "application/json")
        if url.endswith("/metrics"):
            return check.HttpResponse(
                200,
                "\n".join(
                    [
                        "# TYPE kgqa_api_requests_total counter",
                        "kgqa_api_requests_total 2",
                        "# TYPE kgqa_api_request_latency_ms_sum counter",
                        "kgqa_api_request_latency_ms_sum 12.5",
                        "# TYPE kgqa_api_errors_total counter",
                    ]
                ),
                "text/plain",
            )
        raise AssertionError(url)

    report = check.run_checks(
        "http://127.0.0.1:8000",
        token="secret-token",
        timeout=2.5,
        http_get=fake_get,
    )

    assert report["status"] == "passed"
    assert [item["name"] for item in report["checks"]] == ["healthz", "readyz", "metrics"]
    assert {item["status"] for item in report["checks"]} == {"passed"}
    assert all(call[1]["Authorization"] == "Bearer secret-token" for call in calls)
    assert all(call[2] == 2.5 for call in calls)
    assert "secret-token" not in json.dumps(report, ensure_ascii=False)
    assert report["auth"] == "set"


def test_api_runtime_check_reports_readiness_failure():
    check = _load_check()

    def fake_get(url, *, headers, timeout):
        if url.endswith("/healthz"):
            return check.HttpResponse(200, '{"status":"healthy"}', "application/json")
        if url.endswith("/readyz"):
            return check.HttpResponse(
                503,
                '{"status":"not_ready","checks":{"index_path":{"status":"failed"}}}',
                "application/json",
            )
        if url.endswith("/metrics"):
            return check.HttpResponse(200, "kgqa_api_requests_total 2", "text/plain")
        raise AssertionError(url)

    report = check.run_checks("http://127.0.0.1:8000", http_get=fake_get)

    assert report["status"] == "failed"
    readyz = {item["name"]: item for item in report["checks"]}["readyz"]
    assert readyz["status"] == "failed"
    assert readyz["http_status"] == 503
    assert readyz["body"]["status"] == "not_ready"


def test_api_runtime_check_reports_connection_failure():
    check = _load_check()

    def fake_get(url, *, headers, timeout):
        raise OSError("connection refused")

    report = check.run_checks("http://127.0.0.1:8000", http_get=fake_get)

    assert report["status"] == "failed"
    assert report["checks"][0]["status"] == "failed"
    assert report["checks"][0]["error_type"] == "OSError"
    assert "connection refused" in report["checks"][0]["error"]


def test_api_runtime_check_main_prints_json_and_exit_code(capsys, monkeypatch):
    check = _load_check()

    def fake_run_checks(base_url, *, token, timeout, http_get=None):
        assert base_url == "http://127.0.0.1:8000"
        assert token == "env-token"
        assert timeout == 1.25
        return {
            "status": "passed",
            "base_url": base_url,
            "auth": "set",
            "checks": [],
        }

    monkeypatch.setenv("KGQA_API_AUTH_TOKEN", "env-token")
    monkeypatch.setattr(check, "run_checks", fake_run_checks)

    exit_code = check.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "1.25"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert "env-token" not in json.dumps(payload, ensure_ascii=False)
