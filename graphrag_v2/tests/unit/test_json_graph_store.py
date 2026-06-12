"""Tests for the JSON graph store."""

from pathlib import Path

import pytest

from graphrag_v2.extraction.models import CandidateTriple
from graphrag_v2.graph_fusion.models import FusedEntity, FusedRelationship, FusionResult
from graphrag_v2.graph_store import GraphStoreError, JsonGraphStore


def test_json_graph_store_writes_and_reads_stats(temp_dir: Path):
    store = JsonGraphStore(temp_dir)
    result = _fusion_result()

    stats = store.write_graph(result)
    read_stats = store.get_stats()

    assert stats.provider == "json"
    assert (temp_dir / "graph.json").exists()
    assert read_stats.num_entities == 2
    assert read_stats.num_relationships == 1
    assert read_stats.num_rejected_triples == 1


def test_json_graph_store_repeated_write_is_stable(temp_dir: Path):
    store = JsonGraphStore(temp_dir)
    result = _fusion_result()

    store.write_graph(result)
    first = store.get_stats()
    store.write_graph(result)
    second = store.get_stats()

    assert second.num_entities == first.num_entities
    assert second.num_relationships == first.num_relationships
    assert second.num_rejected_triples == first.num_rejected_triples


def test_json_graph_store_missing_graph_raises(temp_dir: Path):
    store = JsonGraphStore(temp_dir)

    with pytest.raises(GraphStoreError, match="Missing graph artifact"):
        store.get_stats()


def _fusion_result() -> FusionResult:
    entity_a = FusedEntity(
        id="entity_a",
        name="GraphRAG",
        canonical_name="graphrag",
        type="Technology",
        description="GraphRAG",
        aliases=["GraphRAG"],
        evidence_chunk_ids=["chunk_1"],
        confidence=0.9,
        metadata={"source_candidate_ids": ["e1"]},
    )
    entity_b = FusedEntity(
        id="entity_b",
        name="Knowledge Graph",
        canonical_name="knowledge graph",
        type="Technology",
        description="Knowledge Graph",
        aliases=["Knowledge Graph"],
        evidence_chunk_ids=["chunk_1"],
        confidence=0.9,
        metadata={"source_candidate_ids": ["e2"]},
    )
    relationship = FusedRelationship(
        id="rel_1",
        source_entity_id="entity_a",
        target_entity_id="entity_b",
        source_name="GraphRAG",
        target_name="Knowledge Graph",
        relation="uses",
        original_relations=["uses"],
        description="GraphRAG uses Knowledge Graph",
        confidence=0.93,
        evidence_chunk_ids=["chunk_1"],
        extraction_count=1,
        metadata={"source_triple_ids": ["t1"]},
    )
    rejected = CandidateTriple(
        id="t2",
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
    graph = {
        "nodes": [entity_a.__dict__, entity_b.__dict__],
        "edges": [relationship.__dict__],
        "statistics": {
            "num_nodes": 2,
            "num_edges": 1,
            "num_rejected_triples": 1,
        },
        "created_at": "2026-06-03T00:00:00+00:00",
    }
    return FusionResult(
        entities=[entity_a, entity_b],
        relationships=[relationship],
        rejected_triples=[rejected],
        graph=graph,
    )
