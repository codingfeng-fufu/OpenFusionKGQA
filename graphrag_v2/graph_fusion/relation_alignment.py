"""Relation alignment compatibility helpers."""

from __future__ import annotations

from typing import Mapping

from graphrag_v2.graph_fusion.relation_schema import (
    OPEN_RELATION_SCHEMA_VERSION,
    RelationAlignment,
    default_relation_schema,
)


def align_relation(
    relation_mention: str,
    source_type: str | None = None,
    target_type: str | None = None,
    relation_aliases: Mapping[str, str] | None = None,
) -> tuple[str, float]:
    """Align a relation mention to the canonical schema."""
    alignment = align_relation_detail(
        relation_mention=relation_mention,
        source_type=source_type,
        target_type=target_type,
        relation_aliases=relation_aliases,
    )
    return alignment.canonical_relation, alignment.score


def align_relation_detail(
    relation_mention: str,
    source_type: str | None = None,
    target_type: str | None = None,
    relation_aliases: Mapping[str, str] | None = None,
) -> RelationAlignment:
    """Return detailed alignment metadata for graph fusion."""
    registry = default_relation_schema(relation_aliases=relation_aliases)
    return registry.align(
        relation_mention=relation_mention,
        source_type=source_type,
        target_type=target_type,
    )


def align_open_relation_detail(
    relation_mention: str,
    relation_aliases: Mapping[str, str] | None = None,
) -> RelationAlignment:
    """Normalize an open predicate without constraining it to a schema."""
    normalized = _normalize_open_relation(relation_mention)
    aliases = {
        _normalize_open_relation(alias): _normalize_open_relation(canonical)
        for alias, canonical in (relation_aliases or {}).items()
    }
    canonical = aliases.get(normalized, normalized)
    return RelationAlignment(
        canonical_relation=canonical,
        score=1.0,
        schema_version=OPEN_RELATION_SCHEMA_VERSION,
        normalized_mention=normalized,
        text_score=1.0,
        endpoint_score=1.0,
        endpoint_compatible=True,
        endpoint_reason="schema_disabled",
    )


def _normalize_open_relation(value: str) -> str:
    normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
    return normalized or "related_to"
