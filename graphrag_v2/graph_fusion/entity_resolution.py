"""Entity resolution for extracted candidates."""

from __future__ import annotations

import hashlib
from typing import Mapping

from graphrag_v2.extraction.models import ExtractedEntity
from graphrag_v2.graph_fusion.models import FusedEntity


def resolve_entities(
    entities: list[ExtractedEntity],
    entity_aliases: Mapping[str, str] | None = None,
) -> dict[str, FusedEntity]:
    """Merge candidate entities by normalized name."""
    normalized_aliases = {
        normalize_entity_name(alias): normalize_entity_name(canonical)
        for alias, canonical in (entity_aliases or {}).items()
    }
    grouped: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        normalized_name = normalize_entity_name(entity.name)
        canonical_name = normalized_aliases.get(normalized_name, normalized_name)
        grouped.setdefault(canonical_name, []).append(entity)

    resolved: dict[str, FusedEntity] = {}
    for canonical_name, group in grouped.items():
        display_name = group[0].name.strip()
        evidence = sorted(
            {chunk_id for entity in group for chunk_id in entity.evidence_chunk_ids}
        )
        aliases = sorted(
            {
                entity.name.strip()
                for entity in group
                if entity.name.strip()
            }
            | {canonical_name}
        )
        confidence = sum(entity.confidence for entity in group) / len(group)
        source_candidate_ids = [entity.id for entity in group]
        resolved[canonical_name] = FusedEntity(
            id=_stable_entity_id(canonical_name),
            name=display_name,
            canonical_name=canonical_name,
            type=group[0].type,
            description=group[0].description,
            aliases=aliases,
            evidence_chunk_ids=evidence,
            confidence=round(confidence, 4),
            metadata={
                "source_candidate_ids": source_candidate_ids,
                "source_entity_names": [entity.name for entity in group],
                "source_entity_types": sorted({entity.type for entity in group}),
                "source_descriptions": [entity.description for entity in group],
                "confidence_aggregation": "mean_candidate_confidence",
            },
        )

    return resolved


def normalize_entity_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _stable_entity_id(canonical_name: str) -> str:
    digest = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:16]
    return f"entity_{digest}"
