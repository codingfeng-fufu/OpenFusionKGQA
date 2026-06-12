"""Data models for graph-grounded QA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RouteType = Literal["local", "global"]


@dataclass(frozen=True)
class RoutingDecision:
    route: RouteType
    reason: str


@dataclass(frozen=True)
class LinkedEntity:
    id: str
    name: str
    canonical_name: str
    type: str | None
    description: str | None
    score: float
    aliases: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelationshipEvidence:
    id: str
    source_entity_id: str
    target_entity_id: str
    source_name: str
    target_name: str
    relation: str
    description: str
    confidence: float
    extraction_count: int
    evidence_chunk_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    hop: int = 1


@dataclass(frozen=True)
class GraphEvidence:
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    relationships: list[RelationshipEvidence] = field(default_factory=list)
    text_chunk_ids: list[str] = field(default_factory=list)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunityEvidence:
    community_id: str
    report_id: str
    title: str
    summary: str
    full_content: str
    findings: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    key_relationships: list[str] = field(default_factory=list)
    rank: float = 0.0
    score: float = 0.0
    evidence_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextEvidence:
    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class CandidateAnswer:
    id: str
    answer_text: str
    answer_type: str
    source_chunk_id: str
    source_span: list[int]
    confidence: float
    source: str = "text"


@dataclass(frozen=True)
class QAResult:
    question: str
    route: RouteType
    answer: str
    citations: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    used_entities: list[str] = field(default_factory=list)
    used_relationships: list[str] = field(default_factory=list)
    used_communities: list[str] = field(default_factory=list)
    used_community_reports: list[str] = field(default_factory=list)
    confidence: float = 0.0
    graph_evidence: GraphEvidence = field(default_factory=GraphEvidence)
    community_evidence: list[CommunityEvidence] = field(default_factory=list)
    text_evidence: list[TextEvidence] = field(default_factory=list)
    source_provider: str = "json"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.refusal_reason is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["refused"] = self.refused
        return data
