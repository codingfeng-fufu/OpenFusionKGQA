"""Tests for QA data source loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from graphrag_v2.graph_store import GraphStoreError
from graphrag_v2.qa import load_qa_data_source


def test_load_qa_data_source_falls_back_to_local_without_neo4j_password(temp_dir: Path, monkeypatch):
    index_dir = temp_dir / "index"
    index_dir.mkdir()
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    (index_dir / "index_metadata.json").write_text(
        json.dumps({"graph_store_provider": "neo4j"}),
        encoding="utf-8",
    )
    (index_dir / "graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "entity_1",
                        "name": "GraphRAG",
                        "canonical_name": "graphrag",
                        "type": "Technology",
                        "description": "GraphRAG",
                        "aliases": ["GraphRAG"],
                        "evidence_chunk_ids": ["chunk_1"],
                    }
                ],
                "edges": [
                    {
                        "id": "rel_1",
                        "source_entity_id": "entity_1",
                        "target_entity_id": "entity_2",
                        "source_name": "GraphRAG",
                        "target_name": "微软",
                        "relation": "uses",
                        "description": "GraphRAG uses Knowledge Graph",
                        "confidence": 0.9,
                        "extraction_count": 1,
                        "evidence_chunk_ids": ["chunk_1"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "source_path": "/tmp/doc.md",
                "chunk_index": 0,
                "text": "GraphRAG uses Knowledge Graph evidence.",
                "n_tokens": 8,
                "metadata": {"source": "test"},
            }
        ]
    ).to_parquet(index_dir / "text_units.parquet", index=False)

    source = load_qa_data_source(index_dir)

    assert source.provider == "json"
    assert len(source.entities()) == 1
    assert len(source.relationships()) == 1
    metadata = source.metadata()
    assert metadata["qa_fallback_from_provider"] == "neo4j"
    assert "Neo4j" in metadata["qa_fallback_reason"]


def test_load_qa_data_source_strict_neo4j_raises_without_connection(
    temp_dir: Path,
    monkeypatch,
):
    index_dir = temp_dir / "index"
    index_dir.mkdir()
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    (index_dir / "index_metadata.json").write_text(
        json.dumps({"graph_store_provider": "neo4j"}),
        encoding="utf-8",
    )

    with pytest.raises(GraphStoreError, match="Neo4j"):
        load_qa_data_source(index_dir, allow_neo4j_fallback=False)
