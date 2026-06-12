"""Tests for graph fusion."""

import json

from graphrag_v2.artifacts import write_fusion_artifacts
from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag_v2.graph_fusion import (
    FUSION_PARAMETERS_VERSION,
    FusionOverrides,
    TRIPLE_SCORING_VERSION,
    export_review_queue,
    fuse_graph,
    load_fusion_overrides,
)
from graphrag_v2.graph_fusion.relation_schema import (
    OPEN_RELATION_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
)


def test_fuse_graph_accepts_and_merges_duplicate_relationships():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="Knowledge Graph",
            type="Technology",
            description="Knowledge Graph",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triples = [
        _triple("t1", "chunk_1"),
        _triple("t2", "chunk_2"),
    ]
    relationships = [
        _relationship("rel_t1", "GraphRAG", "Knowledge Graph", "uses", "chunk_1"),
        _relationship("rel_t2", "GraphRAG", "Knowledge Graph", "uses", "chunk_2"),
    ]

    result = fuse_graph(entities, relationships, triples)

    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.relation == "uses"
    assert relationship.extraction_count == 2
    assert relationship.evidence_chunk_ids == ["chunk_1", "chunk_2"]
    assert relationship.metadata["source_triple_ids"] == ["t1", "t2"]
    assert relationship.metadata["source_relationship_ids"] == ["rel_t1", "rel_t2"]
    assert relationship.metadata["relation_schema_mode"] == "open"
    assert relationship.metadata["relation_schema_version"] == (
        OPEN_RELATION_SCHEMA_VERSION
    )
    assert relationship.metadata["fusion_parameters_version"] == (
        FUSION_PARAMETERS_VERSION
    )
    assert result.rejected_triples == []
    assert result.graph["statistics"]["num_edges"] == 1
    assert result.metadata["fusion_relation_schema_mode"] == "open"
    assert result.metadata["fusion_relation_schema_version"] == (
        OPEN_RELATION_SCHEMA_VERSION
    )
    assert result.metadata["fusion_scoring_version"] == TRIPLE_SCORING_VERSION
    assert result.metadata["fusion_num_accepted_triples"] == 2


def test_fuse_graph_rejects_low_score_triple():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        )
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="GraphRAG",
        target_name="GraphRAG",
        relation_mention="unknown",
        canonical_relation=None,
        description="bad",
        extraction_confidence=0.1,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=[],
    )

    result = fuse_graph(entities, [], [triple])

    assert result.relationships == []
    assert len(result.rejected_triples) == 1
    rejected = result.rejected_triples[0]
    assert rejected.status == "rejected"
    assert rejected.canonical_relation == "unknown"
    assert rejected.relation_alignment_score == 1.0
    assert rejected.evidence_support_score == 0.0
    assert rejected.graph_consistency_score == 0.0
    assert rejected.triple_score is not None
    assert set(rejected.metadata["rejection_reasons"]) == {
        "self_loop",
        "missing_evidence",
        "below_min_confidence",
    }
    assert rejected.metadata["fusion_parameters_version"] == FUSION_PARAMETERS_VERSION


def test_fuse_graph_scores_missing_entity_as_inconsistent():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        )
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="GraphRAG",
        target_name="Missing Entity",
        relation_mention="uses",
        canonical_relation=None,
        description="GraphRAG uses Missing Entity",
        extraction_confidence=0.9,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    result = fuse_graph(entities, [], [triple])

    assert result.relationships == []
    assert len(result.rejected_triples) == 1
    rejected = result.rejected_triples[0]
    assert rejected.status == "rejected"
    assert rejected.graph_consistency_score == 0.0
    assert rejected.triple_score == 0.815
    assert rejected.metadata["rejection_reasons"] == ["missing_target_entity"]


def test_fuse_graph_rejects_missing_evidence_even_above_threshold():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="Knowledge Graph",
            type="Technology",
            description="Knowledge Graph",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triple = _triple("t1", chunk_id="")

    result = fuse_graph(entities, [], [triple])

    assert result.relationships == []
    assert result.rejected_triples[0].metadata["rejection_reasons"] == [
        "missing_evidence"
    ]


def test_fuse_graph_accepts_endpoint_incompatible_relation_as_open_predicate():
    entities = [
        ExtractedEntity(
            id="e1",
            name="Neo4j",
            type="GraphDatabase",
            description="Neo4j",
            confidence=0.95,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="JSON",
            type="DataFormat",
            description="JSON",
            confidence=0.85,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="Neo4j",
        target_name="JSON",
        relation_mention="contrasts_with",
        canonical_relation=None,
        description="Neo4j contrasts with JSON",
        extraction_confidence=0.8,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    result = fuse_graph(entities, [], [triple])

    assert result.rejected_triples == []
    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.relation == "contrasts_with"
    assert relationship.original_relations == ["contrasts_with"]
    assert relationship.metadata["endpoint_warnings"] == []
    assert relationship.metadata["relation_schema_mode"] == "open"


def test_fuse_graph_closed_relation_schema_keeps_endpoint_warnings():
    entities = [
        ExtractedEntity(
            id="e1",
            name="Neo4j",
            type="GraphDatabase",
            description="Neo4j",
            confidence=0.95,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="JSON",
            type="DataFormat",
            description="JSON",
            confidence=0.85,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="Neo4j",
        target_name="JSON",
        relation_mention="contrasts_with",
        canonical_relation=None,
        description="Neo4j contrasts with JSON",
        extraction_confidence=0.8,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    result = fuse_graph(entities, [], [triple], relation_schema_mode="closed")

    assert result.rejected_triples == []
    relationship = result.relationships[0]
    assert relationship.relation == "contrasts_with"
    assert relationship.metadata["relation_schema_mode"] == "closed"
    assert relationship.metadata["relation_schema_version"] == RELATION_SCHEMA_VERSION
    assert relationship.metadata["endpoint_warnings"] == [
        {
            "canonical_relation": "has_fallback",
            "endpoint_reason": "incompatible_source_type",
            "relation_mention": "contrasts_with",
        }
    ]


def test_fuse_graph_preserves_unknown_relation_mention_instead_of_related_to():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="System",
            description="GraphRAG",
            confidence=0.95,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="Entities",
            type="Concept",
            description="Entities",
            confidence=0.85,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="GraphRAG",
        target_name="Entities",
        relation_mention="organizes",
        canonical_relation=None,
        description="GraphRAG organizes entities",
        extraction_confidence=0.9,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    result = fuse_graph(entities, [], [triple])

    assert result.rejected_triples == []
    assert len(result.relationships) == 1
    assert result.relationships[0].relation == "organizes"
    assert result.relationships[0].original_relations == ["organizes"]


def test_fuse_graph_applies_manual_entity_and_relation_overrides():
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="知识图谱",
            type="Technology",
            description="Knowledge Graph",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    triple = CandidateTriple(
        id="t1",
        source_name="GraphRAG",
        target_name="Knowledge Graph",
        relation_mention="employs",
        canonical_relation=None,
        description="GraphRAG employs knowledge graphs",
        extraction_confidence=0.82,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
        metadata={"relationship_id": "rel_t1"},
    )

    result = fuse_graph(
        entities,
        [],
        [triple],
        overrides=FusionOverrides(
            entity_aliases={"知识图谱": "knowledge graph"},
            relation_aliases={"employs": "uses"},
        ),
    )

    assert len(result.relationships) == 1
    assert result.relationships[0].relation == "uses"
    assert result.relationships[0].target_name == "知识图谱"
    assert result.metadata["fusion_entity_override_count"] == 1
    assert result.metadata["fusion_relation_override_count"] == 1


def test_load_fusion_overrides_from_json(temp_dir):
    path = temp_dir / "fusion-overrides.json"
    path.write_text(
        json.dumps(
            {
                "entity_aliases": {"知识图谱": "knowledge graph"},
                "relation_aliases": {"employs": "uses"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    overrides = load_fusion_overrides(path)

    assert overrides.entity_aliases == {"知识图谱": "knowledge graph"}
    assert overrides.relation_aliases == {"employs": "uses"}


def test_export_review_queue_from_local_artifacts(temp_dir):
    entities = [
        ExtractedEntity(
            id="e1",
            name="GraphRAG",
            type="Technology",
            description="GraphRAG",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
        ExtractedEntity(
            id="e2",
            name="Knowledge Graph",
            type="Technology",
            description="Knowledge Graph",
            confidence=0.9,
            evidence_chunk_ids=["chunk_1"],
        ),
    ]
    accepted = _triple("t1", "chunk_1")
    rejected = CandidateTriple(
        id="t_rejected",
        source_name="GraphRAG",
        target_name="Missing",
        relation_mention="uses",
        canonical_relation=None,
        description="GraphRAG uses Missing",
        extraction_confidence=0.9,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )
    result = fuse_graph(entities, [], [accepted, rejected])
    index_dir = temp_dir / "index"
    output_path = temp_dir / "review.jsonl"
    write_fusion_artifacts(index_dir, result, min_confidence=0.4)

    summary = export_review_queue(index_path=index_dir, output_path=output_path)
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["num_records"] == 2
    assert summary["num_accepted"] == 1
    assert summary["num_rejected"] == 1
    assert {record["status"] for record in records} == {"accepted", "rejected"}
    rejected_record = next(record for record in records if record["status"] == "rejected")
    assert rejected_record["rejection_reasons"] == ["missing_target_entity"]


def _triple(id_: str, chunk_id: str) -> CandidateTriple:
    return CandidateTriple(
        id=id_,
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
        evidence_chunk_ids=[chunk_id] if chunk_id else [],
        metadata={"relationship_id": f"rel_{id_}"},
    )


def _relationship(
    id_: str,
    source: str,
    target: str,
    relation: str,
    chunk_id: str,
) -> ExtractedRelationship:
    return ExtractedRelationship(
        id=id_,
        source=source,
        target=target,
        relation=relation,
        description=f"{source} {relation} {target}",
        confidence=0.82,
        evidence_chunk_ids=[chunk_id],
    )
