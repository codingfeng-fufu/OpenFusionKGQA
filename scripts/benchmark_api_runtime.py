#!/usr/bin/env python3
"""Benchmark a running OpenFusionKGQA API runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping
from urllib import error, request
from urllib.parse import urljoin


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_REQUESTS = 20
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_P95_MS = 500.0
DEFAULT_MIN_SUCCESS_RATE = 1.0


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: str
    content_type: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_benchmark(
    base_url: str,
    *,
    path: str = "/healthz",
    method: str = "GET",
    json_body: dict | None = None,
    request_count: int = DEFAULT_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_p95_ms: float = DEFAULT_MAX_P95_MS,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    http_request: Callable[..., HttpResponse] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    normalized_base_url = _normalize_base_url(base_url)
    normalized_path = _normalize_path(path)
    url = urljoin(normalized_base_url + "/", normalized_path.lstrip("/"))
    headers = _auth_headers(token)
    requester = http_request or _http_request
    monotonic = clock or time.perf_counter

    samples = []
    worker_count = max(1, min(concurrency, request_count))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _run_one,
                url,
                method=method,
                headers=headers,
                timeout=timeout,
                json_body=json_body,
                http_request=requester,
                clock=monotonic,
            )
            for _ in range(request_count)
        ]
        for future in as_completed(futures):
            samples.append(future.result())

    summary = _summarize(samples)
    threshold_results = _evaluate_thresholds(
        summary,
        max_p95_ms=max_p95_ms,
        min_success_rate=min_success_rate,
    )
    return {
        "status": (
            "passed"
            if summary["failed"] == 0
            and all(item["status"] == "passed" for item in threshold_results)
            else "failed"
        ),
        "checked_at": utc_now(),
        "base_url": normalized_base_url,
        "path": normalized_path,
        "method": method,
        "auth": "set" if token else "unset",
        "requests": request_count,
        "concurrency": worker_count,
        "timeout_seconds": timeout,
        "thresholds": {
            "max_p95_ms": max_p95_ms,
            "min_success_rate": min_success_rate,
        },
        "summary": summary,
        "errors": dict(sorted(_error_counts(samples).items())),
        "threshold_results": threshold_results,
    }


def _run_one(
    url: str,
    *,
    method: str,
    headers: Mapping[str, str],
    timeout: float,
    json_body: dict | None,
    http_request: Callable[..., HttpResponse],
    clock: Callable[[], float],
) -> dict:
    start = clock()
    try:
        response = http_request(
            url,
            method=method,
            headers=headers,
            timeout=timeout,
            json_body=json_body,
        )
        elapsed_ms = round((clock() - start) * 1000, 3)
        success = 200 <= response.status_code < 300
        return {
            "success": success,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "error_type": None if success else f"http_status_{response.status_code}",
        }
    except Exception as exc:
        elapsed_ms = round((clock() - start) * 1000, 3)
        return {
            "success": False,
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def _summarize(samples: list[dict]) -> dict:
    attempted = len(samples)
    succeeded = sum(1 for sample in samples if sample["success"])
    failed = attempted - succeeded
    latencies = sorted(sample["elapsed_ms"] for sample in samples)
    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": round(succeeded / attempted, 6) if attempted else 0.0,
        "error_count": failed,
        "latency_ms": {
            "min": _round_or_zero(latencies[0] if latencies else 0.0),
            "max": _round_or_zero(latencies[-1] if latencies else 0.0),
            "avg": _round_or_zero(sum(latencies) / len(latencies) if latencies else 0.0),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        },
    }


def _evaluate_thresholds(
    summary: Mapping,
    *,
    max_p95_ms: float,
    min_success_rate: float,
) -> list[dict]:
    success_rate = float(summary["success_rate"])
    p95 = float(summary["latency_ms"]["p95"])
    return [
        {
            "name": "min_success_rate",
            "status": "passed" if success_rate >= min_success_rate else "failed",
            "actual": success_rate,
            "threshold": min_success_rate,
        },
        {
            "name": "max_p95_ms",
            "status": "passed" if p95 <= max_p95_ms else "failed",
            "actual": p95,
            "threshold": max_p95_ms,
        },
    ]


def _error_counts(samples: list[dict]) -> Counter:
    return Counter(
        sample["error_type"]
        for sample in samples
        if sample.get("error_type")
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil((percentile / 100) * len(values)) - 1))
    return _round_or_zero(values[index])


def _round_or_zero(value: float) -> float:
    return round(float(value), 3)


def _http_request(
    url: str,
    *,
    method: str,
    headers: Mapping[str, str],
    timeout: float,
    json_body: dict | None,
) -> HttpResponse:
    body_bytes = None
    request_headers = dict(headers)
    if json_body is not None:
        body_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = request.Request(url, data=body_bytes, headers=request_headers, method=method)
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


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _normalize_path(path: str) -> str:
    normalized = path.strip() or "/healthz"
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _auth_headers(token: str | None) -> dict[str, str]:
    normalized = (token or "").strip()
    return {"Authorization": f"Bearer {normalized}"} if normalized else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a running OpenFusionKGQA API runtime.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--path", default="/healthz")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-p95-ms", type=float, default=DEFAULT_MAX_P95_MS)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--token", default=None, help="Bearer token. Defaults to KGQA_API_AUTH_TOKEN.")
    parser.add_argument("--ask-question", default=None, help="Benchmark POST /ask with this question.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = args.token if args.token is not None else os.getenv("KGQA_API_AUTH_TOKEN")
    path = "/ask" if args.ask_question is not None else args.path
    method = "POST" if args.ask_question is not None else "GET"
    json_body = {"question": args.ask_question} if args.ask_question is not None else None
    report = run_benchmark(
        base_url=args.base_url,
        path=path,
        method=method,
        json_body=json_body,
        request_count=args.requests,
        concurrency=args.concurrency,
        token=token,
        timeout=args.timeout,
        max_p95_ms=args.max_p95_ms,
        min_success_rate=args.min_success_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
