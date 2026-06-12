"""Tests for the production runtime API foundation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from graphrag_v2.api.app import create_app
from graphrag_v2.api.settings import ApiRuntimeSettings


def test_api_runtime_settings_loads_environment(monkeypatch, tmp_path):
    config_path = tmp_path / "settings.yaml"
    index_path = tmp_path / "index"
    monkeypatch.setenv("KGQA_API_INDEX_PATH", str(index_path))
    monkeypatch.setenv("KGQA_API_CONFIG", str(config_path))
    monkeypatch.setenv("KGQA_API_ANSWERER", "llm")
    monkeypatch.setenv("KGQA_API_STRICT_NEO4J", "true")
    monkeypatch.setenv("KGQA_API_AUTH_TOKEN", "dummy-api-token")
    monkeypatch.setenv("KGQA_API_MAX_QUESTION_CHARS", "123")

    settings = ApiRuntimeSettings.from_env()

    assert settings.index_path == index_path
    assert settings.config_path == config_path
    assert settings.answerer == "llm"
    assert settings.strict_neo4j is True
    assert settings.auth_token == "dummy-api-token"
    assert settings.max_question_chars == 123


def test_healthz_returns_healthy_without_dependencies(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
            )
        )
    )

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_healthz_does_not_require_auth_token(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
                auth_token="dummy-api-token",
            )
        )
    )

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_requires_bearer_token_when_configured(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
                auth_token="dummy-api-token",
            )
        )
    )

    response = client.get("/metrics", headers={"X-Request-ID": "request-metrics"})

    assert response.status_code == 401
    assert response.json() == {
        "status": "error",
        "error_type": "Unauthorized",
        "error": "missing bearer token",
        "request_id": "request-metrics",
    }


def test_metrics_reports_request_status_latency_and_error_type(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
                auth_token="dummy-api-token",
            )
        )
    )

    health = client.get("/healthz")
    unauthorized = client.get("/readyz")
    metrics = client.get(
        "/metrics",
        headers={"Authorization": "Bearer dummy-api-token"},
    )

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    body = metrics.text
    assert (
        'kgqa_api_requests_total{method="GET",path="/healthz",status_code="200"} 1'
        in body
    )
    assert (
        'kgqa_api_requests_total{method="GET",path="/readyz",status_code="401"} 1'
        in body
    )
    assert (
        'kgqa_api_request_latency_ms_sum{method="GET",path="/healthz",status_code="200"}'
        in body
    )
    assert (
        'kgqa_api_errors_total{error_type="Unauthorized",method="GET",path="/readyz",status_code="401"} 1'
        in body
    )


def test_readyz_requires_bearer_token_when_configured(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=index_path,
                auth_token="dummy-api-token",
            )
        )
    )

    response = client.get("/readyz", headers={"X-Request-ID": "request-auth"})

    assert response.status_code == 401
    assert response.json() == {
        "status": "error",
        "error_type": "Unauthorized",
        "error": "missing bearer token",
        "request_id": "request-auth",
    }


def test_readyz_accepts_valid_bearer_token(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=index_path,
                auth_token="dummy-api-token",
            )
        )
    )

    response = client.get(
        "/readyz",
        headers={"Authorization": "Bearer dummy-api-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readyz_reports_missing_index(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
            )
        )
    )

    response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["index_path"]["status"] == "failed"


def test_readyz_passes_for_local_index(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(create_app(ApiRuntimeSettings(index_path=index_path)))

    response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["index_path"]["status"] == "passed"
    assert payload["checks"]["graph_store"]["provider"] == "json"


def test_readyz_llm_uses_api_runtime_config(tmp_path, monkeypatch):
    index_path = _write_mock_index(tmp_path / "index")
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: deepseek-v4-flash
    model_provider: deepseek
    api_key: dummy-key
  default_embedding_model:
    type: embedding
    model: text-embedding-3-small
extraction:
  llm_provider: deepseek
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("KGQA_REAL_LLM_CONFIG", raising=False)
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=index_path,
                config_path=config_path,
                answerer="llm",
            )
        )
    )

    response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["llm"]["status"] == "passed"
    assert payload["checks"]["llm"]["provider"] == "deepseek"


def test_ask_returns_graph_grounded_answer(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(create_app(ApiRuntimeSettings(index_path=index_path)))

    response = client.post("/ask", json={"question": "GraphRAG 是什么？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "GraphRAG 是什么？"
    assert payload["answer"]
    assert payload["citations"] == ["chunk_1"]
    assert payload["metadata"]["source_provider"] == "json"


def test_ask_requires_bearer_token_when_configured(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=index_path,
                auth_token="dummy-api-token",
            )
        )
    )

    response = client.post("/ask", json={"question": "GraphRAG 是什么？"})

    assert response.status_code == 401
    assert response.json()["error_type"] == "Unauthorized"


def test_ask_rejects_question_over_configured_length(tmp_path):
    index_path = _write_mock_index(tmp_path / "index")
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=index_path,
                max_question_chars=10,
            )
        )
    )

    response = client.post("/ask", json={"question": "x" * 11})

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert response.json()["error_type"] == "ValidationError"
    assert "question exceeds max length" in response.json()["error"]


def test_ask_returns_stable_readiness_error_contract(tmp_path):
    client = TestClient(
        create_app(
            ApiRuntimeSettings(
                index_path=tmp_path / "missing",
            )
        )
    )

    response = client.post(
        "/ask",
        json={"question": "GraphRAG 是什么？"},
        headers={"X-Request-ID": "request-not-ready"},
    )

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["error_type"] == "ReadinessError"
    assert response.json()["request_id"] == "request-not-ready"


def test_request_log_has_request_id_and_redacts_secrets(
    tmp_path,
    monkeypatch,
    caplog,
    ):
    index_path = _write_mock_index(tmp_path / "index")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-runtime-key")
    monkeypatch.setenv("KGQA_API_AUTH_TOKEN", "dummy-api-token")
    client = TestClient(create_app(ApiRuntimeSettings(index_path=index_path)))

    with caplog.at_level(logging.INFO, logger="graphrag_v2.api"):
        response = client.get(
            "/healthz?token=dummy-runtime-key&api_key=dummy-api-token",
            headers={"X-Request-ID": "request-123"},
        )

    assert response.status_code == 200
    log_payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "graphrag_v2.api"
    ]
    assert log_payloads
    assert log_payloads[-1]["request_id"] == "request-123"
    assert log_payloads[-1]["path"] == "/healthz"
    assert "dummy-runtime-key" not in json.dumps(log_payloads, ensure_ascii=False)
    assert "dummy-api-token" not in json.dumps(log_payloads, ensure_ascii=False)
    assert "***" in json.dumps(log_payloads, ensure_ascii=False)


def _write_mock_index(index_path: Path) -> Path:
    index_path.mkdir(parents=True)
    metadata = {
        "index_id": "test_index",
        "graph_store_provider": "json",
        "run_status": "succeeded",
    }
    (index_path / "index_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "chunk_1",
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "source_path": "doc.md",
                "chunk_index": 0,
                "text": "GraphRAG uses Knowledge Graph evidence.",
            }
        ]
    )
    entities = pd.DataFrame(
        [
            {
                "id": "entity_graphrag",
                "name": "GraphRAG",
                "canonical_name": "graphrag",
                "type": "Technology",
                "description": "GraphRAG",
                "aliases": ["GraphRAG"],
                "evidence_chunk_ids": ["chunk_1"],
            },
            {
                "id": "entity_kg",
                "name": "Knowledge Graph",
                "canonical_name": "knowledge graph",
                "type": "Technology",
                "description": "Knowledge Graph",
                "aliases": ["Knowledge Graph"],
                "evidence_chunk_ids": ["chunk_1"],
            },
        ]
    )
    relationships = pd.DataFrame(
        [
            {
                "id": "rel_1",
                "source_entity_id": "entity_graphrag",
                "target_entity_id": "entity_kg",
                "source_name": "GraphRAG",
                "target_name": "Knowledge Graph",
                "relation": "uses",
                "description": "GraphRAG uses Knowledge Graph",
                "confidence": 0.9,
                "extraction_count": 1,
                "evidence_chunk_ids": ["chunk_1"],
            }
        ]
    )
    text_units.to_parquet(index_path / "text_units.parquet", index=False)
    entities.to_parquet(index_path / "entities.parquet", index=False)
    relationships.to_parquet(index_path / "relationships.parquet", index=False)
    (index_path / "graph.json").write_text(
        json.dumps(
            {
                "nodes": entities.to_dict(orient="records"),
                "edges": relationships.to_dict(orient="records"),
                "statistics": {
                    "num_nodes": 2,
                    "num_edges": 1,
                    "num_rejected_triples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return index_path
