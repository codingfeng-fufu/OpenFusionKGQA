"""Tests for triple scoring."""

from graphrag_v2.extraction.models import CandidateTriple
from graphrag_v2.graph_fusion.triple_scoring import score_triple


def test_score_triple_uses_formula():
    triple = CandidateTriple(
        id="triple_1",
        source_name="GraphRAG",
        target_name="Knowledge Graph",
        relation_mention="uses",
        canonical_relation=None,
        description="GraphRAG uses knowledge graphs",
        extraction_confidence=0.8,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=["chunk_1"],
    )

    score = score_triple(
        triple,
        relation_alignment_score=1.0,
        evidence_support_score=1.0,
        graph_consistency_score=1.0,
    )

    assert score == 0.93


def test_score_triple_penalizes_missing_evidence_and_self_loop():
    triple = CandidateTriple(
        id="triple_1",
        source_name="GraphRAG",
        target_name="GraphRAG",
        relation_mention="related_to",
        canonical_relation=None,
        description="bad",
        extraction_confidence=0.5,
        relation_alignment_score=None,
        evidence_support_score=None,
        graph_consistency_score=None,
        triple_score=None,
        status="candidate",
        evidence_chunk_ids=[],
    )

    assert score_triple(triple, relation_alignment_score=0.5) == 0.325
