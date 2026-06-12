"""Community aggregation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Community:
    id: str
    level: int
    title: str
    summary: str
    entity_ids: list[str]
    relationship_ids: list[str]
    text_unit_ids: list[str]
    size: int
    rank: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunityReport:
    id: str
    community_id: str
    title: str
    summary: str
    full_content: str
    findings: list[str]
    key_entities: list[str]
    key_relationships: list[str]
    evidence_chunk_ids: list[str]
    rank: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphProjection:
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


@dataclass(frozen=True)
class CommunityPipelineResult:
    communities: list[Community]
    reports: list[CommunityReport]
