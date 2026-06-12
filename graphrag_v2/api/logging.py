"""Structured JSON logging helpers for the API runtime."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from graphrag_v2.artifacts.run_observability import redact_secrets


LOGGER_NAME = "graphrag_v2.api"
logger = logging.getLogger(LOGGER_NAME)


class JsonRequestLogMiddleware(BaseHTTPMiddleware):
    """Emit one redacted JSON log record per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        status_code = 500
        error_type = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            error_type = exc.__class__.__name__
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            error_type = getattr(request.state, "error_type", None) or error_type
            path = _route_path(request)
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record(
                    method=request.method,
                    path=path,
                    status_code=status_code,
                    elapsed_ms=elapsed_ms,
                    error_type=error_type,
                )
            payload: dict[str, Any] = {
                "event": "api_request",
                "request_id": request_id,
                "method": request.method,
                "path": str(redact_secrets(path)),
                "status_code": status_code,
                "elapsed_ms": elapsed_ms,
            }
            if request.url.query:
                payload["query"] = str(redact_secrets(str(request.url.query)))
            if error_type:
                payload["error_type"] = error_type
            log_json(payload)
        response.headers["X-Request-ID"] = request_id
        return response


def log_json(payload: Mapping[str, Any]) -> None:
    logger.info(json.dumps(redact_secrets(dict(payload)), ensure_ascii=False, sort_keys=True))


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)
