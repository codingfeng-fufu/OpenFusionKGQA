"""Tests for entity resolution."""

from graphrag_v2.extraction.models import ExtractedEntity
from graphrag_v2.graph_fusion.entity_resolution import resolve_entities


def test_resolve_entities_merges_normalized_names():
    entities = [
        ExtractedEntity(
            id="e1",
            name=" GraphRAG ",
            type="Technology",
            description="first",
            confidence=0.8,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="graphrag",
            type="Technology",
            description="second",
            confidence=1.0,
            evidence_chunk_ids=["chunk_2", "chunk_1"],
        ),
    ]

    resolved = resolve_entities(entities)

    assert list(resolved) == ["graphrag"]
    entity = resolved["graphrag"]
    assert entity.name == "GraphRAG"
    assert entity.aliases == ["GraphRAG", "graphrag"]
    assert entity.evidence_chunk_ids == ["chunk_1", "chunk_2"]
    assert entity.confidence == 0.9
    assert entity.metadata["source_candidate_ids"] == ["e1", "e2"]


def test_resolve_entities_applies_cross_lingual_manual_aliases():
    entities = [
        ExtractedEntity(
            id="e1",
            name="Knowledge Graph",
            type="Technology",
            description="first",
            confidence=0.8,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="知识图谱",
            type="Technology",
            description="second",
            confidence=0.9,
            evidence_chunk_ids=["chunk_2"],
        ),
    ]

    resolved = resolve_entities(
        entities,
        entity_aliases={"知识图谱": "knowledge graph"},
    )

    assert list(resolved) == ["knowledge graph"]
    entity = resolved["knowledge graph"]
    assert entity.aliases == ["Knowledge Graph", "knowledge graph", "知识图谱"]
    assert entity.metadata["source_candidate_ids"] == ["e1", "e2"]
