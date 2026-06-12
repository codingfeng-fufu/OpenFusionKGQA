"""In-memory API runtime metrics."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock


class ApiMetrics:
    """Small process-local metrics collector for the FastAPI runtime."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_ms_sums: dict[tuple[str, str, int], float] = defaultdict(float)
        self._error_counts: dict[tuple[str, str, int, str], int] = defaultdict(int)

    def record(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        elapsed_ms: float,
        error_type: str | None = None,
    ) -> None:
        request_key = (method, path, status_code)
        with self._lock:
            self._request_counts[request_key] += 1
            self._latency_ms_sums[request_key] += elapsed_ms
            if error_type:
                self._error_counts[(method, path, status_code, error_type)] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            request_counts = dict(self._request_counts)
            latency_ms_sums = dict(self._latency_ms_sums)
            error_counts = dict(self._error_counts)

        lines = [
            "# HELP kgqa_api_requests_total Total API requests by method, path, and status code.",
            "# TYPE kgqa_api_requests_total counter",
        ]
        for (method, path, status_code), count in sorted(request_counts.items()):
            lines.append(
                f'kgqa_api_requests_total{{method="{_escape_label(method)}",path="{_escape_label(path)}",status_code="{status_code}"}} {count}'
            )

        lines.extend(
            [
                "# HELP kgqa_api_request_latency_ms_sum Total API request latency in milliseconds.",
                "# TYPE kgqa_api_request_latency_ms_sum counter",
            ]
        )
        for (method, path, status_code), elapsed_ms in sorted(latency_ms_sums.items()):
            lines.append(
                f'kgqa_api_request_latency_ms_sum{{method="{_escape_label(method)}",path="{_escape_label(path)}",status_code="{status_code}"}} {elapsed_ms:.3f}'
            )

        lines.extend(
            [
                "# HELP kgqa_api_errors_total Total API errors by error type.",
                "# TYPE kgqa_api_errors_total counter",
            ]
        )
        for (method, path, status_code, error_type), count in sorted(error_counts.items()):
            lines.append(
                f'kgqa_api_errors_total{{error_type="{_escape_label(error_type)}",method="{_escape_label(method)}",path="{_escape_label(path)}",status_code="{status_code}"}} {count}'
            )
        return "\n".join(lines) + "\n"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
