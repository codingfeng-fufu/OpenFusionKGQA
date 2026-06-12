"""Optional Neo4j integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from graphrag_v2.artifacts import compute_index_id
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.extraction.models import CandidateTriple
from graphrag_v2.graph_fusion.models import FusedEntity, FusedRelationship, FusionResult
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


def test_neo4j_write_is_idempotent(temp_dir: Path):
    config = _neo4j_config()
    _require_neo4j(config)
    store = Neo4jGraphStore(config, index_path=temp_dir)
    result = _fusion_result()

    first = store.write_graph(result)
    second = store.write_graph(result)

    assert first.index_id == compute_index_id(temp_dir)
    assert second.index_id == first.index_id
    assert second.num_entities == first.num_entities
    assert second.num_relationships == first.num_relationships
    assert second.num_text_units == first.num_text_units
    assert second.num_entities >= 2
    assert second.num_relationships >= 1
    assert second.health_status == "ready"
    assert second.schema_ready is True
    assert second.schema_version == "2026-06-06.v1"
    assert second.missing_schema_constraints == []
    assert second.missing_schema_indexes == []
    assert second.write_strategy == "staged_replace"


def test_neo4j_writes_are_isolated_by_index_id(temp_dir: Path):
    config = _neo4j_config()
    _require_neo4j(config)
    first_dir = temp_dir / "first"
    second_dir = temp_dir / "second"

    first = Neo4jGraphStore(config, index_path=first_dir).write_graph(_fusion_result())
    second = Neo4jGraphStore(config, index_path=second_dir).write_graph(_fusion_result())

    first_stats = Neo4jGraphStore(config, index_path=first_dir).get_stats()
    second_stats = Neo4jGraphStore(config, index_path=second_dir).get_stats()

    assert first.index_id != second.index_id
    assert first_stats.num_entities == first.num_entities
    assert second_stats.num_entities == second.num_entities
    assert first_stats.index_id == first.index_id
    assert second_stats.index_id == second.index_id


def _fusion_result() -> FusionResult:
    suffix = "kgqa_neo4j_integration"
    entity_a = FusedEntity(
        id=f"{suffix}_entity_a",
        name="GraphRAG",
        canonical_name=f"{suffix}_graphrag",
        type="Technology",
        description="GraphRAG",
        aliases=["GraphRAG"],
        evidence_chunk_ids=[],
        confidence=0.9,
        metadata={"test": suffix},
    )
    entity_b = FusedEntity(
        id=f"{suffix}_entity_b",
        name="Knowledge Graph",
        canonical_name=f"{suffix}_knowledge_graph",
        type="Technology",
        description="Knowledge Graph",
        aliases=["Knowledge Graph"],
        evidence_chunk_ids=[],
        confidence=0.9,
        metadata={"test": suffix},
    )
    relationship = FusedRelationship(
        id=f"{suffix}_rel_1",
        source_entity_id=entity_a.id,
        target_entity_id=entity_b.id,
        source_name=entity_a.name,
        target_name=entity_b.name,
        relation="uses",
        original_relations=["uses"],
        description="GraphRAG uses Knowledge Graph",
        confidence=0.93,
        evidence_chunk_ids=[],
        extraction_count=1,
        metadata={"test": suffix},
    )
    rejected = CandidateTriple(
        id=f"{suffix}_rejected_1",
        source_name="GraphRAG",
        target_name="Missing",
        relation_mention="uses",
        canonical_relation="uses",
        description="rejected",
        extraction_confidence=0.1,
        relation_alignment_score=1.0,
        evidence_support_score=0.0,
        graph_consistency_score=0.0,
        triple_score=0.335,
        status="rejected",
        evidence_chunk_ids=[],
    )
    return FusionResult(
        entities=[entity_a, entity_b],
        relationships=[relationship],
        rejected_triples=[rejected],
        graph={
            "nodes": [entity_a.__dict__, entity_b.__dict__],
            "edges": [relationship.__dict__],
            "statistics": {
                "num_nodes": 2,
                "num_edges": 1,
                "num_rejected_triples": 1,
            },
        },
    )
