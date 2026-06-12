"""Triple scoring for graph fusion."""

from __future__ import annotations

from graphrag_v2.extraction.models import CandidateTriple


TRIPLE_SCORING_VERSION = "2026-06-06.v1"
TRIPLE_SCORING_WEIGHTS = {
    "extraction_confidence": 0.35,
    "relation_alignment": 0.30,
    "evidence_support": 0.20,
    "graph_consistency": 0.15,
}


def score_triple(
    triple: CandidateTriple,
    relation_alignment_score: float,
    evidence_support_score: float | None = None,
    graph_consistency_score: float | None = None,
) -> float:
    """Score a candidate triple using the MVP scoring formula."""
    evidence_score = (
        evidence_support_score
        if evidence_support_score is not None
        else (1.0 if triple.evidence_chunk_ids else 0.0)
    )
    consistency_score = (
        graph_consistency_score
        if graph_consistency_score is not None
        else (0.0 if triple.source_name == triple.target_name else 1.0)
    )
    score = (
        TRIPLE_SCORING_WEIGHTS["extraction_confidence"]
        * triple.extraction_confidence
        + TRIPLE_SCORING_WEIGHTS["relation_alignment"]
        * relation_alignment_score
        + TRIPLE_SCORING_WEIGHTS["evidence_support"] * evidence_score
        + TRIPLE_SCORING_WEIGHTS["graph_consistency"] * consistency_score
    )
    return round(score, 4)


def scoring_metadata() -> dict:
    return {
        "version": TRIPLE_SCORING_VERSION,
        "weights": dict(TRIPLE_SCORING_WEIGHTS),
    }
