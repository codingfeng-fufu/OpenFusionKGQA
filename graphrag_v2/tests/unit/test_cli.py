"""Unit tests for CLI helpers."""

import argparse

from graphrag_v2.cli.main import (
    _format_graph_store_stats,
    _run_cli_action,
    _validate_index_mode,
)
from graphrag_v2.graph_store import GraphStoreStats


def test_cli_expected_error_is_sanitized(monkeypatch, capsys):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "secret-api-key")

    exit_code = _run_cli_action(
        "Indexing",
        lambda: (_ for _ in ()).throw(ValueError("bad key secret-api-key")),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Indexing failed:" in captured.err
    assert "secret-api-key" not in captured.err
    assert "***" in captured.err


def test_cli_expected_error_redacts_embedding_api_key(monkeypatch, capsys):
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_KEY", "embedding-secret")

    exit_code = _run_cli_action(
        "Indexing",
        lambda: (_ for _ in ()).throw(
            ValueError("bad embedding key embedding-secret")
        ),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "embedding-secret" not in captured.err
    assert "***" in captured.err


def test_cli_unexpected_error_returns_one(capsys):
    exit_code = _run_cli_action(
        "Indexing",
        lambda: (_ for _ in ()).throw(RuntimeError("internal failure")),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Indexing failed unexpectedly:" in captured.err
    assert "RuntimeError" in captured.err


def test_format_graph_store_stats_includes_health_and_missing_schema():
    stats = GraphStoreStats(
        provider="neo4j",
        index_id="kgqa_demo",
        database="neo4j",
        num_text_units=2,
        num_entities=5,
        num_relationships=3,
        num_rejected_triples=1,
        metadata_path="/tmp/index_metadata.json",
        schema_version="2026-06-06.v1",
        schema_ready=False,
        health_status="degraded",
        schema_constraints=["kgqa_index_id_unique"],
        schema_indexes=[],
        expected_schema_constraints=[
            "kgqa_index_id_unique",
            "kgqa_entity_scoped_id_unique",
        ],
        expected_schema_indexes=["kgqa_entity_scoped_name_index"],
        missing_schema_constraints=["kgqa_entity_scoped_id_unique"],
        missing_schema_indexes=["kgqa_entity_scoped_name_index"],
    )

    output = _format_graph_store_stats(stats)

    assert "health_status: degraded" in output
    assert "schema_version: 2026-06-06.v1" in output
    assert "missing_schema_constraints: 1 [kgqa_entity_scoped_id_unique]" in output
    assert "missing_schema_indexes: 1 [kgqa_entity_scoped_name_index]" in output


def test_validate_index_mode_rejects_strict_neo4j_with_json_store():
    args = argparse.Namespace(
        strict_neo4j=True,
        graph_store="json",
        mode="full",
    )

    try:
        _validate_index_mode(args)
    except ValueError as exc:
        assert "--strict-neo4j requires --graph-store neo4j" in str(exc)
    else:
        raise AssertionError("Expected strict Neo4j validation failure")


def test_validate_index_mode_allows_strict_neo4j_full_mode():
    args = argparse.Namespace(
        strict_neo4j=True,
        graph_store="neo4j",
        mode="full",
    )

    _validate_index_mode(args)
