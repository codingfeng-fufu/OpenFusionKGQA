"""Integration tests for documents-only artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from graphrag_v2.artifacts import (
    CANDIDATE_ENTITIES_COLUMNS,
    CANDIDATE_RELATIONSHIPS_COLUMNS,
    CANDIDATE_TRIPLES_COLUMNS,
    DOCUMENT_METADATA_KEYS,
    DOCUMENT_SCAN_KEYS,
    DOCUMENT_SCAN_RECORD_KEYS,
    ENTITIES_COLUMNS,
    FUSION_METADATA_KEYS,
    GRAPH_KEYS,
    GRAPH_STATISTICS_KEYS,
    JSON_GRAPH_STORE_METADATA_KEYS,
    RELATIONSHIPS_COLUMNS,
    SUCCESS_METADATA_KEYS,
    TEXT_UNITS_COLUMNS,
    append_run_event,
    compute_index_id,
    fail_run_metadata,
    finish_run_metadata,
    start_run_metadata,
)
from graphrag_v2.config import GraphRagConfig
from graphrag_v2.indexing import index_documents_only


def test_write_document_artifacts(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("GraphRAG text", encoding="utf-8")

    metadata = index_documents_only(docs_dir, output_dir, GraphRagConfig())

    assert metadata["num_documents"] == 1
    assert metadata["num_text_units"] >= 1
    assert (output_dir / "text_units.parquet").exists()
    assert (output_dir / "document_scan.json").exists()
    assert (output_dir / "index_metadata.json").exists()

    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    assert tuple(text_units.columns) == TEXT_UNITS_COLUMNS

    document_scan = json.loads((output_dir / "document_scan.json").read_text())
    assert tuple(document_scan.keys()) == DOCUMENT_SCAN_KEYS
    assert tuple(document_scan["records"][0].keys()) == DOCUMENT_SCAN_RECORD_KEYS
    assert document_scan["num_files"] == 1
    assert document_scan["num_included_files"] == 1
    assert document_scan["num_ignored_files"] == 0
    assert document_scan["num_rejected_files"] == 0
    assert document_scan["num_empty_documents"] == 0

    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    _assert_metadata_keys(saved_metadata, SUCCESS_METADATA_KEYS)
    _assert_metadata_keys(saved_metadata, DOCUMENT_METADATA_KEYS)
    assert saved_metadata["index_id"] == compute_index_id(output_dir)
    assert saved_metadata["num_documents"] == 1
    assert saved_metadata["metadata_schema_version"] == 1
    assert saved_metadata["run_status"] == "succeeded"
    assert saved_metadata["run_mode"] == "documents-only"
    assert saved_metadata["run_started_at"]
    assert saved_metadata["run_finished_at"]
    assert saved_metadata["run_elapsed_seconds"] >= 0


def test_documents_only_run_writes_observability_artifacts(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("GraphRAG text", encoding="utf-8")

    metadata = index_documents_only(docs_dir, output_dir, GraphRagConfig())

    events_path = output_dir / "run_events.jsonl"
    summary_path = output_dir / "run_summary.json"
    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert metadata["run_id"].startswith("run_")
    assert saved_metadata["run_id"] == metadata["run_id"]
    assert saved_metadata["run_failed_stage"] is None
    assert saved_metadata["run_stage_timings"]["document"] >= 0
    assert saved_metadata["run_event_count"] == len(events)
    assert saved_metadata["run_events_path"] == str(events_path.resolve())
    assert saved_metadata["run_summary_path"] == str(summary_path.resolve())
    assert [event["event"] for event in events] == [
        "run_start",
        "stage_start",
        "stage_end",
        "run_end",
    ]
    assert events[1]["stage"] == "document"
    assert events[2]["counts"]["num_documents"] == saved_metadata["num_documents"] == 1
    assert events[2]["counts"]["num_text_units"] == saved_metadata["num_text_units"]
    assert summary["summary_schema_version"] == 1
    assert summary["run_id"] == metadata["run_id"]
    assert summary["status"] == "succeeded"
    assert summary["document"]["num_documents"] == saved_metadata["num_documents"] == 1
    assert summary["document"]["num_text_units"] == saved_metadata["num_text_units"]


def test_run_metadata_lifecycle_writes_observability_links_and_summary(
    temp_dir: Path,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("GraphRAG text", encoding="utf-8")

    metadata = start_run_metadata(
        output_dir,
        mode="documents-only",
        input_path=docs_dir,
    )
    append_run_event(
        output_dir,
        run_id=metadata["run_id"],
        index_id=metadata["index_id"],
        stage="run",
        event="run_start",
        status="running",
    )
    append_run_event(
        output_dir,
        run_id=metadata["run_id"],
        index_id=metadata["index_id"],
        stage="run",
        event="run_end",
        status="succeeded",
    )
    metadata = finish_run_metadata(
        output_dir,
        stage_timings={"document": 0.1},
    )

    events_path = output_dir / "run_events.jsonl"
    summary_path = output_dir / "run_summary.json"
    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert metadata["run_id"].startswith("run_")
    assert metadata["index_id"] == compute_index_id(output_dir)
    assert metadata["output_path"] == str(output_dir.resolve())
    assert saved_metadata["run_id"] == metadata["run_id"]
    assert saved_metadata["run_failed_stage"] is None
    assert saved_metadata["run_stage_timings"] == {"document": 0.1}
    assert saved_metadata["run_event_count"] == len(events)
    assert saved_metadata["run_events_path"] == str(events_path.resolve())
    assert saved_metadata["run_summary_path"] == str(summary_path.resolve())
    assert [event["event"] for event in events] == ["run_start", "run_end"]
    assert summary["summary_schema_version"] == 1
    assert summary["run_id"] == metadata["run_id"]
    assert summary["status"] == "succeeded"
    assert summary["document"]["elapsed_seconds"] == 0.1


def test_finish_run_metadata_overwrites_existing_timings_with_empty_dict(
    temp_dir: Path,
):
    output_dir = temp_dir / "artifacts"
    metadata = start_run_metadata(
        output_dir,
        mode="documents-only",
        input_path=temp_dir,
    )
    metadata["run_stage_timings"] = {"document": 0.2}
    (output_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = finish_run_metadata(output_dir, stage_timings={})
    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    summary = json.loads((output_dir / "run_summary.json").read_text())

    assert metadata["run_stage_timings"] == {}
    assert saved_metadata["run_stage_timings"] == {}
    assert summary["document"]["elapsed_seconds"] is None


def test_fail_run_metadata_accepts_positional_error_and_empty_timings(
    temp_dir: Path,
):
    output_dir = temp_dir / "artifacts"
    metadata = start_run_metadata(
        output_dir,
        mode="documents-only",
        input_path=temp_dir,
    )
    metadata["run_stage_timings"] = {"document": 0.2}
    (output_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = fail_run_metadata(
        output_dir,
        ValueError("boom"),
        failed_stage="document",
        stage_timings={},
    )
    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    summary = json.loads((output_dir / "run_summary.json").read_text())

    assert metadata["run_status"] == "failed"
    assert metadata["run_failed_stage"] == "document"
    assert metadata["run_stage_timings"] == {}
    assert saved_metadata["run_error_type"] == "ValueError"
    assert saved_metadata["run_error_message"] == "boom"
    assert saved_metadata["run_stage_timings"] == {}
    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "document"
    assert summary["document"]["elapsed_seconds"] is None
    assert summary["errors"][0]["error_type"] == "ValueError"


def test_run_metadata_redacts_operational_secret_values(monkeypatch, temp_dir: Path):
    monkeypatch.setenv("GRAPHRAG_API_KEY", "metadata-secret")
    input_dir = temp_dir / "metadata-secret-input"
    output_dir = temp_dir / "metadata-secret-output"
    input_dir.mkdir()

    metadata = start_run_metadata(
        output_dir,
        mode="documents-only",
        input_path=input_dir,
    )
    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    returned_blob = json.dumps(metadata, ensure_ascii=False)
    saved_blob = json.dumps(saved_metadata, ensure_ascii=False)

    assert "metadata-secret" not in returned_blob
    assert "metadata-secret" not in saved_blob
    assert "***" in returned_blob
    assert "***" in saved_blob


def test_documents_only_run_clears_stale_generated_artifacts_and_metadata(
    temp_dir: Path,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    output_dir.mkdir()
    (docs_dir / "doc.txt").write_text("Fresh GraphRAG text", encoding="utf-8")

    stale_artifacts = [
        "candidate_entities.parquet",
        "candidate_relationships.parquet",
        "candidate_triples.parquet",
        "entities.parquet",
        "relationships.parquet",
        "rejected_triples.parquet",
        "graph.json",
        "communities.parquet",
        "community_reports.parquet",
        "run_events.jsonl",
        "run_summary.json",
    ]
    for filename in stale_artifacts:
        (output_dir / filename).write_text("stale", encoding="utf-8")
    (output_dir / "document_scan.json").write_text("stale", encoding="utf-8")
    (output_dir / "index_metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "num_entities": 999,
                "num_relationships": 999,
                "graph_store_provider": "neo4j",
                "graph_store_written": True,
                "graph_store_health_status": "degraded",
                "graph_store_schema_version": "stale",
                "graph_store_write_strategy": "staged_replace",
                "graph_store_staging_index_id": "stale_staging",
                "graph_store_missing_schema_constraints": ["stale_constraint"],
                "graph_store_missing_schema_indexes": ["stale_index"],
                "num_communities": 10,
                "llm_total_calls": 42,
                "extraction_parse_failures": 3,
            }
        ),
        encoding="utf-8",
    )

    metadata = index_documents_only(docs_dir, output_dir, GraphRagConfig())

    assert metadata["num_documents"] == 1
    assert (output_dir / "text_units.parquet").exists()
    for filename in [
        filename
        for filename in stale_artifacts
        if filename not in {"run_events.jsonl", "run_summary.json"}
    ]:
        assert not (output_dir / filename).exists()
    document_scan = json.loads((output_dir / "document_scan.json").read_text())
    assert document_scan["num_included_files"] == 1

    saved_metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert saved_metadata["created_at"] == "2026-01-01T00:00:00+00:00"
    assert saved_metadata["run_status"] == "succeeded"
    assert saved_metadata["run_id"].startswith("run_")
    assert saved_metadata["run_failed_stage"] is None
    assert (output_dir / "run_summary.json").exists()
    if (output_dir / "run_events.jsonl").exists():
        assert "stale" not in (output_dir / "run_events.jsonl").read_text(
            encoding="utf-8"
        )
    assert "stale" not in (output_dir / "run_summary.json").read_text(
        encoding="utf-8"
    )
    for key in [
        "num_entities",
        "num_relationships",
        "graph_store_provider",
        "graph_store_written",
        "graph_store_health_status",
        "graph_store_schema_version",
        "graph_store_write_strategy",
        "graph_store_staging_index_id",
        "graph_store_missing_schema_constraints",
        "graph_store_missing_schema_indexes",
        "num_communities",
        "llm_total_calls",
        "extraction_parse_failures",
    ]:
        assert key not in saved_metadata


def test_kgqa_documents_only_cli(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text("# GraphRAG\n\nGraph QA.", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--mode",
            "documents-only",
        ],
        check=False,
        capture_output=True,
        env=_without_real_llm_env(),
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "document_scan.json").exists()
    assert (output_dir / "text_units.parquet").exists()
    assert (output_dir / "index_metadata.json").exists()


def test_kgqa_inspect_run_cli(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "# GraphRAG\n\nGraph QA.",
        encoding="utf-8",
    )

    index_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--mode",
            "documents-only",
        ],
        check=False,
        capture_output=True,
        env=_without_real_llm_env(),
        text=True,
    )
    assert index_result.returncode == 0

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "inspect",
            "run",
            "--index",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inspect_result.returncode == 0
    assert "Run:" in inspect_result.stdout
    assert "summary_status: present" in inspect_result.stdout
    assert "run_id: run_" in inspect_result.stdout
    assert "status: succeeded" in inspect_result.stdout
    assert "failed_stage: None" in inspect_result.stdout
    assert "Stages:" in inspect_result.stdout
    assert "document:" in inspect_result.stdout
    assert "Artifacts:" in inspect_result.stdout
    assert "metadata_path:" in inspect_result.stdout
    assert "run_summary_path:" in inspect_result.stdout
    assert "run_events_path:" in inspect_result.stdout


def test_kgqa_inspect_run_cli_falls_back_to_metadata_when_summary_missing(
    temp_dir: Path,
):
    output_dir = temp_dir / "artifacts"
    output_dir.mkdir()
    metadata = {
        "metadata_schema_version": 1,
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "run_status": "succeeded",
        "run_mode": "documents-only",
        "output_path": str(output_dir),
        "run_started_at": "2026-06-06T12:00:00+00:00",
        "run_finished_at": "2026-06-06T12:00:01+00:00",
        "run_elapsed_seconds": 1.0,
        "run_failed_stage": None,
        "run_stage_timings": {"document": 1.0},
        "num_text_units": 1,
    }
    (output_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "inspect",
            "run",
            "--index",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inspect_result.returncode == 0
    assert "summary_status: missing" in inspect_result.stdout
    assert "run_id: run_20260606T120000Z_abcdef12" in inspect_result.stdout


def test_documents_only_rejected_input_writes_manifest_and_failure_metadata(
    temp_dir: Path,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text("# GraphRAG", encoding="utf-8")
    (docs_dir / "bad.csv").write_text("unsupported", encoding="utf-8")
    config = GraphRagConfig()
    config.input.unsupported_file_policy = "fail"

    with pytest.raises(ValueError, match="document_scan.json"):
        index_documents_only(docs_dir, output_dir, config)

    document_scan = json.loads((output_dir / "document_scan.json").read_text())
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    text_units = pd.read_parquet(output_dir / "text_units.parquet")

    assert tuple(document_scan.keys()) == DOCUMENT_SCAN_KEYS
    assert document_scan["num_files"] == 2
    assert document_scan["num_included_files"] == 1
    assert document_scan["num_rejected_files"] == 1
    assert any(
        record["status"] == "rejected"
        and record["reason"] == "unsupported_extension"
        for record in document_scan["records"]
    )
    assert tuple(text_units.columns) == TEXT_UNITS_COLUMNS
    assert metadata["run_status"] == "failed"
    assert metadata["run_error_type"] == "ValueError"
    assert metadata["num_rejected_files"] == 1


def test_kgqa_extraction_only_cli(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
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
            "extraction-only",
        ],
        check=False,
        capture_output=True,
        env=_without_real_llm_env(),
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "text_units.parquet").exists()
    assert (output_dir / "index_metadata.json").exists()
    assert (output_dir / "candidate_entities.parquet").exists()
    assert (output_dir / "candidate_relationships.parquet").exists()
    assert (output_dir / "candidate_triples.parquet").exists()

    candidate_entities = pd.read_parquet(output_dir / "candidate_entities.parquet")
    candidate_relationships = pd.read_parquet(
        output_dir / "candidate_relationships.parquet"
    )
    candidate_triples = pd.read_parquet(output_dir / "candidate_triples.parquet")

    assert tuple(candidate_entities.columns) == CANDIDATE_ENTITIES_COLUMNS
    assert tuple(candidate_relationships.columns) == CANDIDATE_RELATIONSHIPS_COLUMNS
    assert tuple(candidate_triples.columns) == CANDIDATE_TRIPLES_COLUMNS

    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert metadata["extractor"] == "mock"
    assert metadata["num_candidate_entities"] >= 2
    assert metadata["metadata_schema_version"] == 1
    assert metadata["run_status"] == "succeeded"
    assert metadata["run_mode"] == "extraction-only"


def test_kgqa_extraction_only_llm_cli_requires_real_provider(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j stores graph data.",
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
            "llm",
            "--mode",
            "extraction-only",
        ],
        check=False,
        capture_output=True,
        env=_without_real_llm_env(),
        text=True,
    )

    assert result.returncode == 2
    assert "Indexing failed:" in result.stderr
    assert "requires an API key" in result.stderr
    assert not (output_dir / "candidate_entities.parquet").exists()
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert metadata["run_status"] == "failed"
    assert metadata["run_error_type"] == "LLMProviderError"
    assert "requires an API key" in metadata["run_error_message"]


def test_kgqa_fusion_only_cli(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for filename in [
        "text_units.parquet",
        "candidate_entities.parquet",
        "candidate_relationships.parquet",
        "candidate_triples.parquet",
        "entities.parquet",
        "relationships.parquet",
        "rejected_triples.parquet",
        "graph.json",
    ]:
        assert (output_dir / filename).exists()

    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")
    rejected_triples = pd.read_parquet(output_dir / "rejected_triples.parquet")
    graph = json.loads((output_dir / "graph.json").read_text())
    metadata = json.loads((output_dir / "index_metadata.json").read_text())

    assert tuple(entities.columns) == ENTITIES_COLUMNS
    assert tuple(relationships.columns) == RELATIONSHIPS_COLUMNS
    assert tuple(rejected_triples.columns) == CANDIDATE_TRIPLES_COLUMNS
    assert tuple(graph.keys()) == GRAPH_KEYS
    assert tuple(graph["statistics"].keys()) == GRAPH_STATISTICS_KEYS
    assert len(entities) >= 2
    assert len(relationships) >= 1
    assert graph["statistics"]["num_nodes"] == len(entities)
    assert graph["statistics"]["num_edges"] == len(relationships)
    assert graph["statistics"]["num_rejected_triples"] == len(rejected_triples)
    assert metadata["num_entities"] == len(entities)
    assert metadata["num_relationships"] == len(relationships)
    assert metadata["num_rejected_triples"] == len(rejected_triples)
    assert metadata["fusion_min_confidence"] == 0.4
    assert metadata["graph_store_provider"] == "json"
    assert metadata["graph_store_written"] is True
    assert metadata["graph_store_health_status"] == "ready"
    assert metadata["metadata_schema_version"] == 1
    assert metadata["run_status"] == "succeeded"
    assert metadata["run_mode"] == "fusion-only"
    _assert_metadata_keys(metadata, SUCCESS_METADATA_KEYS)
    _assert_metadata_keys(metadata, DOCUMENT_METADATA_KEYS)
    _assert_metadata_keys(metadata, FUSION_METADATA_KEYS)
    _assert_metadata_keys(metadata, JSON_GRAPH_STORE_METADATA_KEYS)
    assert metadata["graph_store_num_entities"] == len(entities)
    assert metadata["graph_store_num_relationships"] == len(relationships)
    assert metadata["graph_store_num_text_units"] >= 1


def test_kgqa_inspect_graph_cli(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "artifacts"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    index_result = subprocess.run(
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert index_result.returncode == 0

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphrag_v2.cli.main",
            "inspect",
            "graph",
            "--index",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inspect_result.returncode == 0
    assert "provider: json" in inspect_result.stdout
    assert "entities:" in inspect_result.stdout
    assert "relationships:" in inspect_result.stdout
    assert "rejected_triples:" in inspect_result.stdout
    assert "health_status: ready" in inspect_result.stdout
    assert "schema_version: None" in inspect_result.stdout
    assert "schema_ready: None" in inspect_result.stdout


def _without_real_llm_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ZHIPUAI_API_KEY", None)
    env.pop("GRAPHRAG_API_KEY", None)
    return env


def _assert_metadata_keys(metadata: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in metadata]
    assert missing == []
