"""Optional Neo4j-backed community pipeline integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store import GraphStoreError, Neo4jGraphStore


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


def test_kgqa_neo4j_community_pipeline(temp_dir: Path):
    config = _neo4j_config()
    _require_neo4j(config)
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    config_path = temp_dir / "settings.yaml"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
graph_store:
  provider: neo4j
  uri: {config.uri}
  username: {config.username}
  password_env: {config.password_env}
  database: {config.database}

community:
  enabled: true
  algorithm: louvain
  min_community_size: 2
  generate_reports: true
  reporter: mock
""",
        encoding="utf-8",
    )

    result = subprocess.run(
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
            "mock",
            "--mode",
            "fusion-only",
            "--graph-store",
            "neo4j",
            "--community",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "communities.parquet").exists()
    assert (output_dir / "community_reports.parquet").exists()
    communities = pd.read_parquet(output_dir / "communities.parquet")
    reports = pd.read_parquet(output_dir / "community_reports.parquet")
    metadata = json.loads((output_dir / "index_metadata.json").read_text())

    assert len(communities) >= 1
    assert len(reports) >= 1
    assert metadata["num_communities"] == len(communities)
    assert metadata["num_community_reports"] == len(reports)
    assert metadata["graph_store_provider"] == "neo4j"
    assert metadata["graph_store_written"] is True
    assert metadata["graph_store_index_id"] == metadata["index_id"]
    assert metadata["graph_store_database"] == config.database
    assert metadata["graph_store_num_text_units"] > 0

    inspect_graph_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "inspect",
            "graph",
            "--index",
            str(output_dir),
            "--config",
            str(config_path),
            "--graph-store",
            "neo4j",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inspect_graph_result.returncode == 0, inspect_graph_result.stderr
    assert f"index_id: {metadata['index_id']}" in inspect_graph_result.stdout
    assert "provider: neo4j" in inspect_graph_result.stdout
    assert "text_units:" in inspect_graph_result.stdout
    assert "schema_ready: True" in inspect_graph_result.stdout
    assert "kgqa_entity_scoped_id_unique" in inspect_graph_result.stdout
    assert "kgqa_entity_scoped_name_index" in inspect_graph_result.stdout

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "inspect",
            "communities",
            "--index",
            str(output_dir),
            "--config",
            str(config_path),
            "--graph-store",
            "neo4j",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "communities:" in inspect_result.stdout

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
    )

    assert ask_result.returncode == 0, ask_result.stderr
    assert "Answer:" in ask_result.stdout
    assert "Graph Evidence:" in ask_result.stdout


def test_kgqa_community_requires_neo4j(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--extractor",
            "mock",
            "--mode",
            "fusion-only",
            "--graph-store",
            "json",
            "--community",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--community requires --graph-store neo4j" in result.stderr
