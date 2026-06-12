"""Deterministic mock extractor for tests and offline demos."""

from __future__ import annotations

import hashlib
import re

from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction.base import BaseExtractor
from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


TERM_TYPES = {
    "GraphRAG": "Technology",
    "Knowledge Graph": "Technology",
    "RAG": "Technology",
    "Neo4j": "Database",
    "Graph Database": "Database",
    "JSON": "Format",
    "Community Reports": "Artifact",
}

RELATION_OVERRIDES = {
    ("GraphRAG", "Knowledge Graph"): "uses",
    ("Neo4j", "Graph Database"): "is_a",
    ("JSON", "Neo4j"): "fallback_for",
    ("Neo4j", "JSON"): "has_fallback",
}


class MockExtractor(BaseExtractor):
    """Rule-based extractor with stable IDs."""

    async def extract(self, text_unit: TextUnit) -> ExtractionResult:
        text_lower = text_unit.text.lower()
        matched_terms = [
            term for term in TERM_TYPES if _contains_term(text_lower, term)
        ]

        entities = [
            ExtractedEntity(
                id=_stable_id("candidate_entity", term),
                name=term,
                type=TERM_TYPES[term],
                description=f"Mock extracted entity: {term}",
                confidence=0.9,
                evidence_chunk_ids=[text_unit.chunk_id],
                metadata={"extractor": "mock"},
            )
            for term in matched_terms
        ]

        relationships: list[ExtractedRelationship] = []
        triples: list[CandidateTriple] = []
        for index, (source, target) in enumerate(_adjacent_pairs(matched_terms)):
            relation = RELATION_OVERRIDES.get((source, target), "related_to")
            confidence = 0.82 if relation != "related_to" else 0.72
            rel_id = _stable_id(
                "candidate_relationship",
                text_unit.chunk_id,
                source,
                relation,
                target,
            )
            description = f"{source} {relation} {target}"
            relationships.append(
                ExtractedRelationship(
                    id=rel_id,
                    source=source,
                    target=target,
                    relation=relation,
                    description=description,
                    confidence=confidence,
                    evidence_chunk_ids=[text_unit.chunk_id],
                    metadata={"extractor": "mock", "pair_index": index},
                )
            )
            triples.append(
                CandidateTriple(
                    id=_stable_id("candidate_triple", rel_id),
                    source_name=source,
                    target_name=target,
                    relation_mention=relation,
                    canonical_relation=None,
                    description=description,
                    extraction_confidence=confidence,
                    relation_alignment_score=None,
                    evidence_support_score=None,
                    graph_consistency_score=None,
                    triple_score=None,
                    status="candidate",
                    evidence_chunk_ids=[text_unit.chunk_id],
                    metadata={"relationship_id": rel_id, "extractor": "mock"},
                )
            )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            triples=triples,
        )


def _adjacent_pairs(items: list[str]) -> list[tuple[str, str]]:
    return [(items[index], items[index + 1]) for index in range(len(items) - 1)]


def _stable_id(prefix: str, *parts: str) -> str:
    content = ":".join(parts)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _contains_term(text_lower: str, term: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text_lower) is not None
