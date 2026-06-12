"""Graph evidence retrieval for QA."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from graphrag_v2.qa.models import GraphEvidence, LinkedEntity, RelationshipEvidence


class GraphRetriever:
    """Retrieve graph evidence around linked entities."""

    def __init__(
        self,
        top_k_relationships: int = 6,
        hops: int = 1,
        adaptive: bool = True,
        adaptive_max_hops: int = 3,
    ):
        self.top_k_relationships = top_k_relationships
        self.hops = hops
        self.adaptive = adaptive
        self.adaptive_max_hops = adaptive_max_hops

    def retrieve(
        self,
        linked_entities: list[LinkedEntity],
        relationships: list[dict[str, Any]],
        question: str | None = None,
    ) -> GraphEvidence:
        if not linked_entities:
            return GraphEvidence(
                retrieval_metadata=_retrieval_metadata(
                    adaptive_enabled=self.adaptive,
                    adaptive_triggered=False,
                    matched_adaptive_cues=[],
                    max_hops=max(self.hops, 1),
                    selected=[],
                )
            )

        hop_plan = _plan_query(
            question=question,
            base_hops=self.hops,
            adaptive_enabled=self.adaptive,
            adaptive_max_hops=self.adaptive_max_hops,
        )
        max_hops = hop_plan.max_hops
        matched_adaptive_cues = hop_plan.matched_adaptive_cues
        adaptive_triggered = hop_plan.adaptive_triggered

        linked_ids = {entity.id for entity in linked_entities if entity.id}
        frontier = set(linked_ids)
        seen_relationships: set[str] = set()
        selected: list[RelationshipEvidence] = []

        relationship_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relationship in relationships:
            relationship_index[str(relationship.get("source_entity_id", ""))].append(relationship)
            relationship_index[str(relationship.get("target_entity_id", ""))].append(relationship)

        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for entity_id in list(frontier):
                for relationship in relationship_index.get(entity_id, []):
                    rel_id = str(relationship.get("id", ""))
                    if not rel_id or rel_id in seen_relationships:
                        continue
                    source_id = str(relationship.get("source_entity_id", ""))
                    target_id = str(relationship.get("target_entity_id", ""))
                    source_name = str(relationship.get("source_name", source_id))
                    target_name = str(relationship.get("target_name", target_id))
                    base_confidence = float(relationship.get("confidence") or 0.0)
                    extraction_count = int(relationship.get("extraction_count") or 0)
                    linked_bonus = 1.0 if source_id in linked_ids or target_id in linked_ids else 0.6
                    score = round(
                        min(
                            1.0,
                            0.55 * base_confidence
                            + 0.25 * min(extraction_count / 3.0, 1.0)
                            + 0.20 * linked_bonus,
                        ),
                        4,
                    )
                    selected.append(
                        RelationshipEvidence(
                            id=rel_id,
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            source_name=source_name,
                            target_name=target_name,
                            relation=str(relationship.get("relation", "related_to")),
                            description=str(relationship.get("description", "")),
                            confidence=base_confidence,
                            extraction_count=extraction_count,
                            evidence_chunk_ids=_as_list(relationship.get("evidence_chunk_ids")),
                            score=score,
                            hop=hop,
                        )
                    )
                    seen_relationships.add(rel_id)
                    if source_id not in frontier:
                        next_frontier.add(source_id)
                    if target_id not in frontier:
                        next_frontier.add(target_id)
            frontier = next_frontier
            if not frontier:
                break

        selected = _select_relationships(
            selected,
            top_k=self.top_k_relationships,
            adaptive_triggered=adaptive_triggered,
        )
        text_chunk_ids = sorted(
            {
                chunk_id
                for relationship in selected
                for chunk_id in relationship.evidence_chunk_ids
            }
            | {
                chunk_id
                for entity in linked_entities
                for chunk_id in entity.evidence_chunk_ids
            }
        )
        return GraphEvidence(
            linked_entities=linked_entities,
            relationships=selected,
            text_chunk_ids=text_chunk_ids,
            retrieval_metadata=_retrieval_metadata(
                adaptive_enabled=self.adaptive,
                adaptive_triggered=adaptive_triggered,
                matched_adaptive_cues=matched_adaptive_cues,
                max_hops=max_hops,
                selected=selected,
            ),
        )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if value != value:  # NaN guard
        return []
    return [str(value)]


@dataclass(frozen=True)
class QueryHopPlan:
    """Retriever-side query plan for adaptive graph expansion."""

    max_hops: int
    adaptive_triggered: bool
    matched_adaptive_cues: list[str]


_ADAPTIVE_PHRASE_CUES = (
    ("same", "same"),
    ("older", "older"),
    ("younger", "younger"),
    ("born first", "born first"),
    ("born later", "born later"),
    ("died first", "died first"),
    ("died later", "died later"),
    ("released first", "released first"),
    ("came out first", "came out first"),
    ("more recently", "more recently"),
    ("director of", "director"),
    ("founder of", "founder"),
    ("father of", "father"),
    ("mother of", "mother"),
    ("husband of", "spouse"),
    ("wife of", "spouse"),
    ("spouse of", "spouse"),
    ("sibling in law", "sibling in law"),
    ("sibling-in-law", "sibling in law"),
    ("nationality of", "nationality"),
    ("place of birth", "place of birth"),
    ("birth place", "place of birth"),
    ("birthplace", "place of birth"),
    ("place of death", "place of death"),
    ("death place", "place of death"),
    ("composer of", "composer"),
)

_ADAPTIVE_QUERY_PATTERNS = (
    (re.compile(r"\bdirector\b"), "director"),
    (re.compile(r"\bfounder\b"), "founder"),
    (re.compile(r"\bcomposer\b"), "composer"),
    (re.compile(r"\bperformer\b"), "performer"),
    (re.compile(r"\b(?:father|mother)\b"), "father_or_mother"),
    (re.compile(r"\b(?:husband|wife|spouse)\b"), "spouse"),
    (re.compile(r"\bsibling[-\s]+in[-\s]+law\b"), "sibling in law"),
    (re.compile(r"\bnationalit(?:y|ies)\b"), "nationality"),
    (re.compile(r"\b(?:born|birthplace|birth\s+place)\b"), "place of birth"),
    (re.compile(r"\b(?:died|die|death\s+place)\b"), "place of death"),
    (
        re.compile(
            r"\b\w+(?:['’]s)\s+(?:father|mother|husband|wife|spouse|sibling[-\s]+in[-\s]+law|nationality|director|founder|composer)\b"
        ),
        "possessive_chain",
    ),
)

_CUE_ALIASES = {
    "father_or_mother": ("father", "mother"),
}


def _plan_query(
    *,
    question: str | None,
    base_hops: int,
    adaptive_enabled: bool,
    adaptive_max_hops: int,
) -> QueryHopPlan:
    base = max(base_hops, 1)
    matched_adaptive_cues = _matched_adaptive_cues(question)
    adaptive_triggered = bool(adaptive_enabled and matched_adaptive_cues)
    max_hops = max(base, adaptive_max_hops) if adaptive_triggered else base
    return QueryHopPlan(
        max_hops=max_hops,
        adaptive_triggered=adaptive_triggered,
        matched_adaptive_cues=matched_adaptive_cues,
    )


def _should_expand_adaptively(question: str | None) -> bool:
    return bool(_matched_adaptive_cues(question))


def _matched_adaptive_cues(question: str | None) -> list[str]:
    if not question:
        return []
    normalized = " ".join(question.lower().replace("-", " ").split())
    matched: list[str] = []
    for phrase, label in _ADAPTIVE_PHRASE_CUES:
        if phrase.replace("-", " ") in normalized:
            _append_unique(matched, label)
    for pattern, label in _ADAPTIVE_QUERY_PATTERNS:
        if pattern.search(normalized):
            for cue in _CUE_ALIASES.get(label, (label,)):
                if cue in normalized or label not in _CUE_ALIASES:
                    _append_unique(matched, cue)
    return matched


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _select_relationships(
    relationships: list[RelationshipEvidence],
    *,
    top_k: int,
    adaptive_triggered: bool,
) -> list[RelationshipEvidence]:
    ranked = sorted(relationships, key=lambda item: (-item.score, item.id))
    if top_k <= 0:
        return []
    if not adaptive_triggered:
        return ranked[:top_k]

    selected: list[RelationshipEvidence] = []
    selected_ids: set[str] = set()

    def add(relationship: RelationshipEvidence) -> None:
        if relationship.id in selected_ids:
            return
        selected.append(relationship)
        selected_ids.add(relationship.id)

    deepest_hop = max((relationship.hop for relationship in ranked), default=1)
    effective_top_k = top_k
    if deepest_hop > 1:
        deepest_candidate = next(
            relationship for relationship in ranked if relationship.hop == deepest_hop
        )
        deepest_path = _backfill_relationship_path(deepest_candidate, ranked)
        effective_top_k = max(top_k, len(deepest_path))
        for relationship in deepest_path:
            add(relationship)

    for relationship in ranked:
        if len(selected) >= effective_top_k:
            break
        add(relationship)

    return selected


def _backfill_relationship_path(
    relationship: RelationshipEvidence,
    ranked: list[RelationshipEvidence],
) -> list[RelationshipEvidence]:
    path = [relationship]
    current = relationship
    for hop in range(relationship.hop - 1, 0, -1):
        current_entities = {current.source_entity_id, current.target_entity_id}
        parent = next(
            (
                candidate
                for candidate in ranked
                if candidate.hop == hop
                and (
                    candidate.source_entity_id in current_entities
                    or candidate.target_entity_id in current_entities
                )
            ),
            None,
        )
        if parent is None:
            break
        path.append(parent)
        current = parent
    return list(reversed(path))


def _retrieval_metadata(
    *,
    adaptive_enabled: bool,
    adaptive_triggered: bool,
    matched_adaptive_cues: list[str],
    max_hops: int,
    selected: list[RelationshipEvidence],
) -> dict[str, Any]:
    counts: dict[int, int] = {}
    for relationship in selected:
        counts[relationship.hop] = counts.get(relationship.hop, 0) + 1
    return {
        "adaptive_enabled": adaptive_enabled,
        "adaptive_triggered": adaptive_triggered,
        "matched_adaptive_cues": matched_adaptive_cues,
        "hop_plan": list(range(1, max_hops + 1)),
        "relationship_count_by_hop": counts,
        "max_retrieved_hop": max(counts) if counts else 0,
    }
