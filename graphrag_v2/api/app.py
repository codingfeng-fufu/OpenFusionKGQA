"""FastAPI application for the production runtime foundation."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from graphrag_v2.api.logging import JsonRequestLogMiddleware
from graphrag_v2.api.metrics import ApiMetrics
from graphrag_v2.api.settings import ApiRuntimeSettings
from graphrag_v2.artifacts.run_observability import redact_secrets
from graphrag_v2.cli.main import _create_answerer
from graphrag_v2.config import GraphRagConfig, load_config
from graphrag_v2.graph_store import create_graph_store
from graphrag_v2.qa import GraphGroundedQA


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    confidence: float
    metadata: dict[str, Any]


def create_app(settings: ApiRuntimeSettings | None = None) -> FastAPI:
    runtime_settings = settings or ApiRuntimeSettings.from_env()
    app = FastAPI(title="OpenFusionKGQA API", version="0.2.0-beta.1")
    app.state.settings = runtime_settings
    app.state.metrics = ApiMetrics()
    app.add_middleware(JsonRequestLogMiddleware)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            error_type="ValidationError",
            error=_validation_error_message(exc),
            details=exc.errors(),
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metrics")
    def metrics(request: Request):
        auth_error = _require_bearer_token(request, runtime_settings)
        if auth_error is not None:
            return auth_error
        return Response(
            content=request.app.state.metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/readyz")
    def readyz(request: Request):
        auth_error = _require_bearer_token(request, runtime_settings)
        if auth_error is not None:
            return auth_error
        report = readiness_report(runtime_settings)
        if report["status"] != "ready":
            return JSONResponse(status_code=503, content=report)
        return report

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
        auth_error = _require_bearer_token(request, runtime_settings)
        if auth_error is not None:
            return auth_error
        boundary_error = _validate_question_bounds(request, payload, runtime_settings)
        if boundary_error is not None:
            return boundary_error
        report = readiness_report(runtime_settings)
        if report["status"] != "ready":
            return _error_response(
                request,
                status_code=503,
                error_type="ReadinessError",
                error="runtime is not ready",
                details=report,
            )
        try:
            config = _load_runtime_config(runtime_settings)
            qa = GraphGroundedQA.from_index(
                runtime_settings.index_path,
                graph_store_config=config.graph_store,
                answerer=_create_answerer(runtime_settings.answerer, config),
                allow_neo4j_fallback=not runtime_settings.strict_neo4j,
            )
            result = qa.ask(payload.question)
        except Exception as exc:
            return _error_response(
                request,
                status_code=500,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
        metadata = dict(result.metadata)
        metadata["request_id"] = getattr(request.state, "request_id", "")
        return {
            "question": result.question,
            "answer": result.answer,
            "route": result.route,
            "citations": result.citations,
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "confidence": result.confidence,
            "metadata": metadata,
        }

    return app


def readiness_report(settings: ApiRuntimeSettings) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    index_path = Path(settings.index_path)
    if not index_path.exists():
        checks["index_path"] = {
            "status": "failed",
            "path": str(index_path),
            "error": "index path does not exist",
        }
        return {"status": "not_ready", "checks": checks}
    checks["index_path"] = {"status": "passed", "path": str(index_path)}

    try:
        config = _load_runtime_config(settings)
        checks["config"] = {"status": "passed"}
    except Exception as exc:
        checks["config"] = {
            "status": "failed",
            "error": str(redact_secrets(str(exc))),
        }
        return {"status": "not_ready", "checks": checks}

    try:
        graph_store = create_graph_store(
            provider="neo4j" if settings.strict_neo4j else "json",
            index_path=index_path,
            config=config.graph_store,
        )
        stats = graph_store.get_stats()
        checks["graph_store"] = {
            "status": "passed",
            "provider": stats.provider,
            "health_status": stats.health_status,
            "index_id": stats.index_id,
        }
    except Exception as exc:
        checks["graph_store"] = {
            "status": "failed",
            "error": str(redact_secrets(str(exc))),
        }
        return {"status": "not_ready", "checks": checks}

    if settings.answerer == "llm":
        try:
            answerer = _create_answerer(settings.answerer, config)
            llm_client = getattr(answerer, "llm_client", None)
            checks["llm"] = {
                "status": "passed",
                "provider": getattr(llm_client, "provider_name", None),
                "model": getattr(llm_client, "model", None)
                or getattr(llm_client, "model_name", None),
                "mock_mode": bool(getattr(llm_client, "mock_mode", False)),
            }
        except Exception as exc:
            checks["llm"] = {
                "status": "failed",
                "error": str(redact_secrets(str(exc))),
            }
            return {"status": "not_ready", "checks": checks}

    return {"status": "ready", "checks": checks}


def _load_runtime_config(settings: ApiRuntimeSettings) -> GraphRagConfig:
    return load_config(settings.config_path) if settings.config_path else GraphRagConfig()


def _require_bearer_token(
    request: Request,
    settings: ApiRuntimeSettings,
) -> JSONResponse | None:
    if not settings.auth_token:
        return None
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return _error_response(
            request,
            status_code=401,
            error_type="Unauthorized",
            error="missing bearer token",
        )
    if not hmac.compare_digest(token.strip(), settings.auth_token):
        return _error_response(
            request,
            status_code=401,
            error_type="Unauthorized",
            error="invalid bearer token",
        )
    return None


def _validate_question_bounds(
    request: Request,
    payload: AskRequest,
    settings: ApiRuntimeSettings,
) -> JSONResponse | None:
    if len(payload.question) <= settings.max_question_chars:
        return None
    return _error_response(
        request,
        status_code=422,
        error_type="ValidationError",
        error=f"question exceeds max length of {settings.max_question_chars} characters",
    )


def _error_response(
    request: Request,
    *,
    status_code: int,
    error_type: str,
    error: str,
    details: Any | None = None,
) -> JSONResponse:
    request.state.error_type = error_type
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "error": str(redact_secrets(error)),
        "request_id": _request_id(request),
    }
    if details is not None:
        payload["details"] = redact_secrets(details)
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


def _request_id(request: Request) -> str:
    return str(
        getattr(request.state, "request_id", None)
        or request.headers.get("X-Request-ID")
        or ""
    )


def _validation_error_message(exc: RequestValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = str(error.get("msg") or "invalid value")
        messages.append(f"{loc}: {message}" if loc else message)
    return "; ".join(messages) if messages else "request validation failed"


app = create_app()
