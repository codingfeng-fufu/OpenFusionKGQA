#!/usr/bin/env python3
"""Live-check a running OpenFusionKGQA API runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping
from urllib import error, request
from urllib.parse import urljoin


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: str
    content_type: str = ""


HttpGet = Callable[[str], HttpResponse]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_checks(
    base_url: str,
    *,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    http_get: Callable[[str], HttpResponse] | None = None,
) -> dict:
    normalized_base_url = _normalize_base_url(base_url)
    headers = _auth_headers(token)
    getter = http_get or (
        lambda url, *, headers, timeout: _http_get(url, headers=headers, timeout=timeout)
    )

    checks = [
        _check_json_endpoint(
            "healthz",
            urljoin(normalized_base_url + "/", "healthz"),
            expected_status_code=200,
            expected_status_field="healthy",
            headers=headers,
            timeout=timeout,
            http_get=getter,
        ),
        _check_json_endpoint(
            "readyz",
            urljoin(normalized_base_url + "/", "readyz"),
            expected_status_code=200,
            expected_status_field="ready",
            headers=headers,
            timeout=timeout,
            http_get=getter,
        ),
        _check_metrics_endpoint(
            urljoin(normalized_base_url + "/", "metrics"),
            headers=headers,
            timeout=timeout,
            http_get=getter,
        ),
    ]
    return {
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "checked_at": utc_now(),
        "base_url": normalized_base_url,
        "auth": "set" if token else "unset",
        "checks": checks,
    }


def _check_json_endpoint(
    name: str,
    url: str,
    *,
    expected_status_code: int,
    expected_status_field: str,
    headers: Mapping[str, str],
    timeout: float,
    http_get: Callable[[str], HttpResponse],
) -> dict:
    response_or_error = _safe_get(url, headers=headers, timeout=timeout, http_get=http_get)
    if isinstance(response_or_error, dict):
        return {"name": name, **response_or_error}
    response = response_or_error
    body = _parse_json(response.body)
    passed = (
        response.status_code == expected_status_code
        and isinstance(body, dict)
        and body.get("status") == expected_status_field
    )
    result = {
        "name": name,
        "status": "passed" if passed else "failed",
        "http_status": response.status_code,
        "content_type": response.content_type,
    }
    if isinstance(body, dict):
        result["body"] = body
    else:
        result["error"] = "response body is not valid JSON"
        result["body_tail"] = response.body[-500:]
    return result


def _check_metrics_endpoint(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
    http_get: Callable[[str], HttpResponse],
) -> dict:
    response_or_error = _safe_get(url, headers=headers, timeout=timeout, http_get=http_get)
    if isinstance(response_or_error, dict):
        return {"name": "metrics", **response_or_error}
    response = response_or_error
    required_markers = (
        "kgqa_api_requests_total",
        "kgqa_api_request_latency_ms_sum",
        "kgqa_api_errors_total",
    )
    missing = [marker for marker in required_markers if marker not in response.body]
    passed = response.status_code == 200 and not missing
    result = {
        "name": "metrics",
        "status": "passed" if passed else "failed",
        "http_status": response.status_code,
        "content_type": response.content_type,
    }
    if missing:
        result["missing_metrics"] = missing
        result["body_tail"] = response.body[-500:]
    return result


def _safe_get(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
    http_get: Callable[[str], HttpResponse],
) -> HttpResponse | dict:
    try:
        return http_get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        return {
            "status": "failed",
            "http_status": None,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def _http_get(url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    req = request.Request(url, headers=dict(headers), method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return HttpResponse(
                status_code=int(response.status),
                body=body,
                content_type=response.headers.get("content-type", ""),
            )
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return HttpResponse(
            status_code=int(exc.code),
            body=body,
            content_type=exc.headers.get("content-type", ""),
        )


def _parse_json(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _auth_headers(token: str | None) -> dict[str, str]:
    normalized = (token or "").strip()
    return {"Authorization": f"Bearer {normalized}"} if normalized else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live-check a running OpenFusionKGQA API runtime.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=None, help="Bearer token. Defaults to KGQA_API_AUTH_TOKEN.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = args.token if args.token is not None else os.getenv("KGQA_API_AUTH_TOKEN")
    report = run_checks(args.base_url, token=token, timeout=args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
