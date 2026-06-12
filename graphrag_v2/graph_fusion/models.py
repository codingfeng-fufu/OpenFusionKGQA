"""Data models for graph fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphrag_v2.extraction.models import CandidateTriple


@dataclass(frozen=True)
class FusedEntity:
    id: str
    name: str
    canonical_name: str
    type: str
    description: str
    aliases: list[str]
    evidence_chunk_ids: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedRelationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    source_name: str
    target_name: str
    relation: str
    original_relations: list[str]
    description: str
    confidence: float
    evidence_chunk_ids: list[str]
    extraction_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredTriple:
    triple: CandidateTriple
    canonical_relation: str
    relation_alignment_score: float
    evidence_support_score: float
    graph_consistency_score: float
    triple_score: float


@dataclass(frozen=True)
class FusionResult:
    entities: list[FusedEntity]
    relationships: list[FusedRelationship]
    rejected_triples: list[CandidateTriple]
    graph: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
