"""Data models for candidate knowledge extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractedEntity:
    """Entity candidate extracted from a text unit."""

    id: str
    name: str
    type: str
    description: str
    confidence: float
    evidence_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedRelationship:
    """Relationship candidate extracted from a text unit."""

    id: str
    source: str
    target: str
    relation: str
    description: str
    confidence: float
    evidence_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateTriple:
    """Candidate triple before graph fusion."""

    id: str
    source_name: str
    target_name: str
    relation_mention: str
    canonical_relation: str | None
    description: str
    extraction_confidence: float
    relation_alignment_score: float | None
    evidence_support_score: float | None
    graph_consistency_score: float | None
    triple_score: float | None
    status: str
    evidence_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Extraction output for one text unit."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    triples: list[CandidateTriple] = field(default_factory=list)
