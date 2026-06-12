"""Tests for extraction validation."""

from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from graphrag_v2.extraction.validators import validate_extraction_result


def test_validate_valid_result():
    result = _valid_result()

    assert validate_extraction_result(result) == []


def test_validate_rejects_empty_entity_name():
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                id="entity_1",
                name="",
                type="Technology",
                description="bad",
                confidence=0.8,
                evidence_chunk_ids=["chunk_1"],
            )
        ]
    )

    assert "empty name" in validate_extraction_result(result)[0]


def test_validate_rejects_missing_endpoint():
    result = _valid_result(
        relationship=ExtractedRelationship(
            id="rel_1",
            source="GraphRAG",
            target="Missing",
            relation="uses",
            description="bad",
            confidence=0.8,
            evidence_chunk_ids=["chunk_1"],
        )
    )

    assert any("target not found" in error for error in validate_extraction_result(result))


def test_validate_rejects_invalid_confidence_and_missing_evidence():
    result = _valid_result(
        relationship=ExtractedRelationship(
            id="rel_1",
            source="GraphRAG",
            target="Knowledge Graph",
            relation="uses",
            description="bad",
            confidence=1.2,
            evidence_chunk_ids=[],
        )
    )

    errors = validate_extraction_result(result)

    assert any("invalid confidence" in error for error in errors)
    assert any("no evidence chunks" in error for error in errors)


def _valid_result(relationship: ExtractedRelationship | None = None) -> ExtractionResult:
    entities = [
        ExtractedEntity(
            id="entity_1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="entity_2",
            name="Knowledge Graph",
            type="Technology",
            description="Knowledge Graph",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    rel = relationship or ExtractedRelationship(
        id="rel_1",
        source="GraphRAG",
        target="Knowledge Graph",
        relation="uses",
        description="GraphRAG uses knowledge graphs",
        confidence=0.8,
        evidence_chunk_ids=["chunk_1"],
    )
    triple = CandidateTriple(
        id="triple_1",
        source_name=rel.source,
        target_name=rel.target,
        relation_mention=rel.relation,
        canonical_relation=None,
        description=rel.description,
        extraction_confidence=rel.confidence,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=rel.evidence_chunk_ids,
    )
    return ExtractionResult(entities=entities, relationships=[rel], triples=[triple])
