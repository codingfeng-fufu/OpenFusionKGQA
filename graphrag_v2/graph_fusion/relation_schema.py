"""Relation schema registry used by graph fusion."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Mapping


RELATION_SCHEMA_VERSION = "2026-06-06.v1"
OPEN_RELATION_SCHEMA_VERSION = "open"


@dataclass(frozen=True)
class RelationSchemaEntry:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    source_types: tuple[str, ...] | None = None
    target_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RelationAlignment:
    canonical_relation: str
    score: float
    schema_version: str
    normalized_mention: str
    text_score: float
    endpoint_score: float
    endpoint_compatible: bool
    endpoint_reason: str | None = None

    def to_metadata(self) -> dict:
        return {
            "canonical_relation": self.canonical_relation,
            "schema_version": self.schema_version,
            "normalized_mention": self.normalized_mention,
            "text_score": self.text_score,
            "endpoint_score": self.endpoint_score,
            "endpoint_compatible": self.endpoint_compatible,
            "endpoint_reason": self.endpoint_reason,
        }


class RelationSchemaRegistry:
    """Registry for canonical relation names, aliases, and endpoint rules."""

    def __init__(
        self,
        entries: list[RelationSchemaEntry],
        version: str = RELATION_SCHEMA_VERSION,
        relation_aliases: Mapping[str, str] | None = None,
    ):
        self.entries = {entry.canonical_name: entry for entry in entries}
        self.version = version
        self.relation_aliases = {
            _normalize_relation(alias): canonical
            for alias, canonical in (relation_aliases or {}).items()
        }
        unknown = [
            canonical
            for canonical in self.relation_aliases.values()
            if canonical not in self.entries
        ]
        if unknown:
            raise ValueError(
                "Relation override references unknown canonical relation(s): "
                + ", ".join(sorted(set(unknown)))
            )

    def align(
        self,
        relation_mention: str,
        source_type: str | None = None,
        target_type: str | None = None,
    ) -> RelationAlignment:
        mention = _normalize_relation(relation_mention)
        if mention in self.relation_aliases:
            canonical = self.relation_aliases[mention]
            endpoint_score, endpoint_compatible, endpoint_reason = (
                self._endpoint_compatibility(canonical, source_type, target_type)
            )
            return RelationAlignment(
                canonical_relation=canonical,
                score=round(min(0.7 + 0.3 * endpoint_score, 1.0), 4),
                schema_version=self.version,
                normalized_mention=mention,
                text_score=1.0,
                endpoint_score=endpoint_score,
                endpoint_compatible=endpoint_compatible,
                endpoint_reason=endpoint_reason,
            )

        best_relation = "related_to"
        best_text_score = 0.0
        for canonical, entry in self.entries.items():
            candidates = {_normalize_relation(alias) for alias in entry.aliases}
            candidates.add(canonical)
            text_score = max(
                _relation_text_similarity(mention, candidate)
                for candidate in candidates
            )
            if text_score > best_text_score:
                best_text_score = text_score
                best_relation = canonical

        endpoint_score, endpoint_compatible, endpoint_reason = (
            self._endpoint_compatibility(best_relation, source_type, target_type)
        )
        alignment_score = 0.7 * best_text_score + 0.3 * endpoint_score
        if best_text_score < 0.45:
            best_relation = "related_to"
            endpoint_score, endpoint_compatible, endpoint_reason = (
                self._endpoint_compatibility(best_relation, source_type, target_type)
            )
            alignment_score = max(alignment_score, 0.55)

        return RelationAlignment(
            canonical_relation=best_relation,
            score=round(min(alignment_score, 1.0), 4),
            schema_version=self.version,
            normalized_mention=mention,
            text_score=round(best_text_score, 4),
            endpoint_score=endpoint_score,
            endpoint_compatible=endpoint_compatible,
            endpoint_reason=endpoint_reason,
        )

    def _endpoint_compatibility(
        self,
        relation: str,
        source_type: str | None,
        target_type: str | None,
    ) -> tuple[float, bool, str | None]:
        entry = self.entries[relation]
        if not source_type or not target_type:
            return 0.8, True, "missing_endpoint_type"

        source_type_key = _normalize_type(source_type)
        target_type_key = _normalize_type(target_type)
        if entry.source_types and source_type_key not in {
            _normalize_type(value) for value in entry.source_types
        }:
            return 0.3, False, "incompatible_source_type"
        if entry.target_types and target_type_key not in {
            _normalize_type(value) for value in entry.target_types
        }:
            return 0.3, False, "incompatible_target_type"
        if relation == "is_a" and source_type_key == target_type_key:
            return 0.6, True, "same_endpoint_type_for_is_a"
        return 1.0, True, None


def default_relation_schema(
    relation_aliases: Mapping[str, str] | None = None,
) -> RelationSchemaRegistry:
    return RelationSchemaRegistry(
        entries=[
            RelationSchemaEntry(
                canonical_name="uses",
                aliases=(
                    "use",
                    "uses",
                    "used_by",
                    "based_on",
                    "depends_on",
                    "使用",
                    "采用",
                    "基于",
                ),
                description="Source uses or depends on target.",
            ),
            RelationSchemaEntry(
                canonical_name="is_a",
                aliases=("is_a", "isa", "type_of", "is", "是一种", "属于", "是"),
                description="Source is an instance or subtype of target.",
            ),
            RelationSchemaEntry(
                canonical_name="has_fallback",
                aliases=("has_fallback", "fallback_to", "backs_up_with", "备选"),
                description="Source has target as a fallback implementation.",
                source_types=("Technology", "Database", "System", "Store"),
                target_types=("Format", "Database", "System", "Technology"),
            ),
            RelationSchemaEntry(
                canonical_name="fallback_for",
                aliases=("fallback_for", "fallback_of", "备用于"),
                description="Source is a fallback for target.",
                source_types=("Format", "Database", "System", "Technology"),
                target_types=("Technology", "Database", "System", "Store"),
            ),
            RelationSchemaEntry(
                canonical_name="related_to",
                aliases=("related_to", "related", "mentions", "关联", "相关"),
                description="Generic relatedness when no stronger relation fits.",
            ),
        ],
        relation_aliases=relation_aliases,
    )


def _normalize_relation(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _normalize_type(value: str) -> str:
    return str(value).strip().lower()


def _relation_text_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()
