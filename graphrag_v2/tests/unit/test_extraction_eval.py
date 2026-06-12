"""Tests for the extraction evaluation utility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_evaluate_extraction_passes_with_aliases(temp_dir: Path):
    _write_graph_artifacts(temp_dir)
    expected_path = temp_dir / "expected.json"
    expected_path.write_text(
        json.dumps(
            {
                "required_entities": [
                    {"name": "GraphRAG"},
                    {"name": "Knowledge Graph", "aliases": ["knowledge graphs"]},
                ],
                "required_relationships": [
                    {
                        "source": {"name": "GraphRAG"},
                        "relation": {"name": "uses"},
                        "target": {"name": "Knowledge Graph"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_eval(temp_dir, expected_path)

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["required_entities"] == 2
    assert payload["matched_entities"] == 2
    assert payload["entity_recall"] == 1.0
    assert payload["required_relationships"] == 1
    assert payload["matched_relationships"] == 1
    assert payload["relationship_recall"] == 1.0
    assert payload["missing_entities"] == []
    assert payload["missing_relationships"] == []


def test_evaluate_extraction_fails_for_missing_required_entity(temp_dir: Path):
    _write_graph_artifacts(temp_dir)
    expected_path = temp_dir / "expected.json"
    expected_path.write_text(
        json.dumps({"required_entities": [{"name": "Neo4j"}]}),
        encoding="utf-8",
    )

    result = _run_eval(temp_dir, expected_path)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["required_entities"] == 1
    assert payload["matched_entities"] == 0
    assert payload["entity_recall"] == 0.0
    assert payload["missing_entities"] == ["Neo4j"]


def _write_graph_artifacts(index_path: Path) -> None:
    pd.DataFrame(
        [
            {"id": "e1", "name": "GraphRAG"},
            {"id": "e2", "name": "Knowledge Graph"},
        ]
    ).to_parquet(index_path / "entities.parquet", index=False)
    pd.DataFrame(
        [
            {
                "id": "r1",
                "source_name": "GraphRAG",
                "relation": "uses",
                "target_name": "Knowledge Graph",
            }
        ]
    ).to_parquet(index_path / "relationships.parquet", index=False)


def _run_eval(index_path: Path, expected_path: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_extraction.py"),
            "--index",
            str(index_path),
            "--expected",
            str(expected_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
