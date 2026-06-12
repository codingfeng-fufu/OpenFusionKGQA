"""Tests for extraction data models."""

from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


def test_create_extraction_result():
    entity = ExtractedEntity(
        id="candidate_entity_1",
        name="GraphRAG",
        type="Technology",
        description="A graph RAG system",
        confidence=0.9,
        evidence_chunk_ids=["chunk_1"],
    )
    relationship = ExtractedRelationship(
        id="candidate_relationship_1",
        source="GraphRAG",
        target="Knowledge Graph",
        relation="uses",
        description="GraphRAG uses knowledge graphs",
        confidence=0.82,
        evidence_chunk_ids=["chunk_1"],
    )
    triple = CandidateTriple(
        id="candidate_triple_1",
        source_name="GraphRAG",
        target_name="Knowledge Graph",
        relation_mention="uses",
        canonical_relation=None,
        description="GraphRAG uses knowledge graphs",
        extraction_confidence=0.82,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    result = ExtractionResult(
        entities=[entity],
        relationships=[relationship],
        triples=[triple],
    )

    assert result.entities[0].name == "GraphRAG"
    assert result.relationships[0].relation == "uses"
    assert result.triples[0].status == "candidate"
