"""Opt-in real LLM + Neo4j smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store import GraphStoreError, Neo4jGraphStore
from graphrag_v2.llm.real_llm_config import RealLLMSettings, resolve_real_llm_settings


def test_real_llm_neo4j_smoke(temp_dir: Path):
    settings = resolve_real_llm_settings()
    _require_real_llm_smoke(settings)
    config = _neo4j_config()
    _require_neo4j(config)

    repo_root = Path(__file__).resolve().parents[3]
    docs_dir = repo_root / "examples" / "docs"
    output_dir = temp_dir / "real-llm-neo4j-smoke"
    config_path = temp_dir / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _smoke_config(settings, config),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    smoke_env = os.environ.copy()
    smoke_env.update(settings.runtime_env())

    index_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--config",
            str(config_path),
            "--extractor",
            "llm",
            "--graph-store",
            "neo4j",
            "--community",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=smoke_env,
    )

    assert index_result.returncode == 0, index_result.stderr
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert metadata["extractor"] == "llm"
    assert metadata["llm_provider"] == settings.provider
    assert metadata["llm_model_name"] == settings.model
    assert metadata["llm_mock_mode"] is False
    assert metadata["llm_total_calls"] > 0
    assert metadata["extraction_max_gleanings"] == 1
    assert metadata["extraction_gleaning_attempts"] >= metadata["num_text_units"]
    assert "extraction_gleaned_entities" in metadata
    assert "extraction_gleaned_relationships" in metadata
    assert "llm_total_tokens" in metadata
    assert "llm_prompt_tokens" in metadata
    assert "llm_completion_tokens" in metadata
    assert "llm_total_latency_seconds" in metadata
    assert metadata["extraction_failed_chunks"] == 0
    assert metadata["graph_store_provider"] == "neo4j"
    assert metadata["graph_store_written"] is True
    assert metadata["graph_store_index_id"] == metadata["index_id"]
    assert metadata["num_communities"] >= 1
    assert metadata["num_community_reports"] >= 1

    eval_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_extraction.py"),
            "--index",
            str(output_dir),
            "--expected",
            str(repo_root / "examples" / "eval" / "expected_graph.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=smoke_env,
    )
    print(eval_result.stdout)

    ask_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "ask",
            "Neo4j 在项目里起什么作用？",
            "--index",
            str(output_dir),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=smoke_env,
    )
    assert ask_result.returncode == 0, ask_result.stderr
    assert "Answer:" in ask_result.stdout
    assert "Graph Evidence:" in ask_result.stdout


def _require_real_llm_smoke(settings: RealLLMSettings) -> None:
    if os.getenv("KGQA_REAL_LLM_SMOKE") != "1":
        pytest.skip("KGQA_REAL_LLM_SMOKE=1 is required for real LLM smoke")
    blocker = settings.blocker_reason()
    if blocker:
        pytest.skip(f"{blocker} for real LLM smoke")


def _smoke_config(
    settings: RealLLMSettings,
    graph_store: GraphStoreConfig,
) -> dict:
    chat_model = {
        "type": "chat",
        "model": settings.model,
        "model_provider": settings.provider,
        "auth_type": "api_key",
        "api_key": None,
        "max_retries": 3,
        "max_retry_wait": 5.0,
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    if settings.api_base:
        chat_model["api_base"] = settings.api_base
    return {
        "models": {
            "default_chat_model": chat_model,
            "default_embedding_model": {
                "type": "embedding",
                "model": "text-embedding-3-small",
            },
        },
        "extraction": {
            "extractor_provider": "llm",
            "llm_provider": settings.provider,
            "llm_model_id": "default_chat_model",
            "max_retries": 2,
            "max_gleanings": 1,
            "fail_on_invalid_chunk": True,
            "default_confidence": 0.7,
        },
        "graph_store": {
            "provider": "neo4j",
            "uri": graph_store.uri,
            "username": graph_store.username,
            "password_env": graph_store.password_env,
            "database": graph_store.database,
        },
        "community": {
            "enabled": True,
            "algorithm": "louvain",
            "min_community_size": 2,
            "generate_reports": True,
            "reporter": "mock",
        },
    }


def _neo4j_config() -> GraphStoreConfig:
    return GraphStoreConfig(
        provider="neo4j",
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password_env="NEO4J_PASSWORD",
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


def _require_neo4j(config: GraphStoreConfig) -> None:
    if not os.getenv(config.password_env):
        pytest.skip(f"{config.password_env} is not set")
    try:
        Neo4jGraphStore(config).get_stats()
    except GraphStoreError as exc:
        pytest.skip(f"Neo4j is not available: {exc}")
