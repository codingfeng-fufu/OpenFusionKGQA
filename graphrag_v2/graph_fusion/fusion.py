"""Fuse extraction candidates into graph artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from datetime import UTC, datetime

from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag_v2.graph_fusion.entity_resolution import (
    normalize_entity_name,
    resolve_entities,
)
from graphrag_v2.graph_fusion.models import FusedRelationship, FusionResult
from graphrag_v2.graph_fusion.overrides import FusionOverrides
from graphrag_v2.graph_fusion.relation_alignment import (
    align_open_relation_detail,
    align_relation_detail,
)
from graphrag_v2.graph_fusion.relation_schema import RELATION_SCHEMA_VERSION
from graphrag_v2.graph_fusion.triple_scoring import (
    TRIPLE_SCORING_VERSION,
    score_triple,
    scoring_metadata,
)

DEFAULT_MIN_CONFIDENCE = 0.4
FUSION_PARAMETERS_VERSION = "2026-06-06.v1"


def fuse_graph(
    candidate_entities: list[ExtractedEntity],
    candidate_relationships: list[ExtractedRelationship],
    candidate_triples: list[CandidateTriple],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    overrides: FusionOverrides | None = None,
    relation_schema_mode: str = "open",
) -> FusionResult:
    """Fuse candidates into accepted entities and relationships."""
    overrides = overrides or FusionOverrides()
    relation_schema_mode = _normalize_relation_schema_mode(relation_schema_mode)
    entities_by_name = resolve_entities(
        candidate_entities,
        entity_aliases=overrides.entity_aliases,
    )
    entity_aliases = {
        normalize_entity_name(alias): normalize_entity_name(canonical)
        for alias, canonical in overrides.entity_aliases.items()
    }
    relationships_by_id = {
        relationship.id: relationship for relationship in candidate_relationships
    }
    relationship_by_key: dict[
        tuple[str, str, str],
        dict,
    ] = {}
    rejected: list[CandidateTriple] = []

    for triple in candidate_triples:
        source_key = _resolve_entity_key(triple.source_name, entity_aliases)
        target_key = _resolve_entity_key(triple.target_name, entity_aliases)
        source_entity = entities_by_name.get(source_key)
        target_entity = entities_by_name.get(target_key)
        alignment = _align_relation(
            relation_mention=triple.relation_mention,
            source_type=source_entity.type if source_entity else None,
            target_type=target_entity.type if target_entity else None,
            relation_aliases=overrides.relation_aliases,
            relation_schema_mode=relation_schema_mode,
        )
        evidence_score = 1.0 if triple.evidence_chunk_ids else 0.0
        consistency_score = (
            0.0
            if source_key == target_key
            or source_entity is None
            or target_entity is None
            else 1.0
        )
        triple_score = score_triple(
            triple,
            relation_alignment_score=alignment.score,
            evidence_support_score=evidence_score,
            graph_consistency_score=consistency_score,
        )
        rejection_reasons = _rejection_reasons(
            source_entity=source_entity,
            target_entity=target_entity,
            source_key=source_key,
            target_key=target_key,
            evidence_score=evidence_score,
            triple_score=triple_score,
            min_confidence=min_confidence,
        )
        is_accepted = not rejection_reasons
        effective_relation = _effective_relation(
            triple.relation_mention,
            alignment,
            relation_schema_mode=relation_schema_mode,
        )
        metadata = {
            **triple.metadata,
            "fusion_status": "accepted" if is_accepted else "rejected",
            "rejection_reasons": rejection_reasons,
            "relation_alignment": alignment.to_metadata(),
            "relation_schema_mode": relation_schema_mode,
            "fusion_parameters_version": FUSION_PARAMETERS_VERSION,
            "fusion_min_confidence": min_confidence,
            "triple_scoring_version": TRIPLE_SCORING_VERSION,
        }
        scored_triple = replace(
            triple,
            canonical_relation=alignment.canonical_relation,
            relation_alignment_score=alignment.score,
            evidence_support_score=evidence_score,
            graph_consistency_score=consistency_score,
            triple_score=triple_score,
            status="accepted" if is_accepted else "rejected",
            metadata=metadata,
        )

        if not is_accepted:
            rejected.append(scored_triple)
            continue

        key = (source_entity.id, target_entity.id, effective_relation)
        bucket = relationship_by_key.setdefault(
            key,
            {
                "source": source_entity,
                "target": target_entity,
                "relation": effective_relation,
                "original_relations": set(),
                "evidence_chunk_ids": set(),
                "scores": [],
                "descriptions": [],
                "triple_ids": [],
                "relationship_ids": set(),
                "alignment_scores": [],
                "relation_alignments": [],
                "relation_schema_versions": set(),
                "relation_schema_mode": relation_schema_mode,
                "endpoint_warnings": [],
            },
        )
        bucket["original_relations"].add(triple.relation_mention)
        bucket["evidence_chunk_ids"].update(triple.evidence_chunk_ids)
        bucket["scores"].append(triple_score)
        bucket["descriptions"].append(triple.description)
        bucket["triple_ids"].append(triple.id)
        bucket["alignment_scores"].append(alignment.score)
        bucket["relation_alignments"].append(alignment.to_metadata())
        bucket["relation_schema_versions"].add(alignment.schema_version)
        if not alignment.endpoint_compatible:
            bucket["endpoint_warnings"].append(
                {
                    "relation_mention": triple.relation_mention,
                    "canonical_relation": alignment.canonical_relation,
                    "endpoint_reason": alignment.endpoint_reason,
                }
            )
        relationship_id = triple.metadata.get("relationship_id")
        if relationship_id:
            bucket["relationship_ids"].add(relationship_id)
            relationship = relationships_by_id.get(relationship_id)
            if relationship is not None:
                bucket["evidence_chunk_ids"].update(relationship.evidence_chunk_ids)

    relationships = [
        _build_relationship(bucket)
        for bucket in relationship_by_key.values()
    ]
    entities = list(entities_by_name.values())
    graph = _build_graph(entities, relationships, rejected)
    metadata = _fusion_metadata(
        min_confidence=min_confidence,
        num_accepted_triples=sum(
            relationship.extraction_count for relationship in relationships
        ),
        overrides=overrides,
        relation_schema_mode=relation_schema_mode,
    )
    return FusionResult(
        entities=entities,
        relationships=relationships,
        rejected_triples=rejected,
        graph=graph,
        metadata=metadata,
    )


def _build_relationship(bucket: dict) -> FusedRelationship:
    source = bucket["source"]
    target = bucket["target"]
    relation = bucket["relation"]
    confidence = sum(bucket["scores"]) / len(bucket["scores"])
    relation_schema_versions = sorted(bucket["relation_schema_versions"])
    return FusedRelationship(
        id=_stable_relationship_id(source.id, relation, target.id),
        source_entity_id=source.id,
        target_entity_id=target.id,
        source_name=source.name,
        target_name=target.name,
        relation=relation,
        original_relations=sorted(bucket["original_relations"]),
        description=bucket["descriptions"][0],
        confidence=round(confidence, 4),
        evidence_chunk_ids=sorted(bucket["evidence_chunk_ids"]),
        extraction_count=len(bucket["scores"]),
        metadata={
            "source_triple_ids": bucket["triple_ids"],
            "source_relationship_ids": sorted(bucket["relationship_ids"]),
            "source_entity_candidate_ids": (
                source.metadata.get("source_candidate_ids", [])
                + target.metadata.get("source_candidate_ids", [])
            ),
            "source_entity_ids": [source.id, target.id],
            "relation_schema_mode": bucket["relation_schema_mode"],
            "relation_schema_version": (
                relation_schema_versions[0]
                if len(relation_schema_versions) == 1
                else ",".join(relation_schema_versions)
            ),
            "relation_schema_versions": relation_schema_versions,
            "fusion_parameters_version": FUSION_PARAMETERS_VERSION,
            "triple_scoring_version": TRIPLE_SCORING_VERSION,
            "triple_scores": bucket["scores"],
            "relation_alignment_scores": bucket["alignment_scores"],
            "relation_alignments": bucket["relation_alignments"],
            "endpoint_warnings": bucket["endpoint_warnings"],
            "confidence_aggregation": "mean_triple_score",
        },
    )


def _build_graph(entities, relationships, rejected) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "nodes": [asdict(entity) for entity in entities],
        "edges": [asdict(relationship) for relationship in relationships],
        "statistics": {
            "num_nodes": len(entities),
            "num_edges": len(relationships),
            "num_rejected_triples": len(rejected),
        },
    }


def _stable_relationship_id(source_id: str, relation: str, target_id: str) -> str:
    content = f"{source_id}:{relation}:{target_id}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"rel_{digest}"


def _resolve_entity_key(name: str, entity_aliases: dict[str, str]) -> str:
    normalized = normalize_entity_name(name)
    return entity_aliases.get(normalized, normalized)


def _rejection_reasons(
    *,
    source_entity,
    target_entity,
    source_key: str,
    target_key: str,
    evidence_score: float,
    triple_score: float,
    min_confidence: float,
) -> list[str]:
    reasons: list[str] = []
    if source_entity is None:
        reasons.append("missing_source_entity")
    if target_entity is None:
        reasons.append("missing_target_entity")
    if source_key == target_key:
        reasons.append("self_loop")
    if evidence_score <= 0.0:
        reasons.append("missing_evidence")
    if triple_score < min_confidence:
        reasons.append("below_min_confidence")
    return reasons


def _align_relation(
    *,
    relation_mention: str,
    source_type: str | None,
    target_type: str | None,
    relation_aliases: dict[str, str],
    relation_schema_mode: str,
):
    if relation_schema_mode == "open":
        return align_open_relation_detail(
            relation_mention=relation_mention,
            relation_aliases=relation_aliases,
        )
    return align_relation_detail(
        relation_mention=relation_mention,
        source_type=source_type,
        target_type=target_type,
        relation_aliases=relation_aliases,
    )


def _effective_relation(
    relation_mention: str,
    alignment,
    *,
    relation_schema_mode: str,
) -> str:
    """Choose the relation persisted to the open-corpus graph.

    The schema alignment remains useful as metadata and scoring signal, but it
    should not overwrite open predicates when the schema match is generic or
    endpoint compatibility is only a weak fit.
    """
    if relation_schema_mode == "open":
        return alignment.canonical_relation
    normalized = _normalize_relation_mention(relation_mention)
    if not alignment.endpoint_compatible:
        return normalized
    if alignment.canonical_relation == "related_to" and normalized not in {
        "related_to",
        "related",
        "mentions",
    }:
        return normalized
    return alignment.canonical_relation


def _normalize_relation_mention(value: str) -> str:
    normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
    return normalized or "related_to"


def _normalize_relation_schema_mode(value: str) -> str:
    normalized = str(value or "open").strip().lower()
    if normalized not in {"open", "closed"}:
        raise ValueError("relation_schema_mode must be one of: open, closed")
    return normalized


def _fusion_metadata(
    *,
    min_confidence: float,
    num_accepted_triples: int,
    overrides: FusionOverrides,
    relation_schema_mode: str,
) -> dict:
    return {
        "fusion_parameters_version": FUSION_PARAMETERS_VERSION,
        "fusion_relation_schema_mode": relation_schema_mode,
        "fusion_relation_schema_version": (
            "open" if relation_schema_mode == "open" else RELATION_SCHEMA_VERSION
        ),
        "fusion_scoring_version": TRIPLE_SCORING_VERSION,
        "fusion_scoring_weights": scoring_metadata()["weights"],
        "fusion_min_confidence": min_confidence,
        "fusion_num_accepted_triples": num_accepted_triples,
        "fusion_entity_override_count": len(overrides.entity_aliases),
        "fusion_relation_override_count": len(overrides.relation_aliases),
    }
