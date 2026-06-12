"""Entity linker for graph-grounded QA."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from graphrag_v2.qa.models import LinkedEntity


class EntityLinker:
    """Match question mentions to graph entities."""

    def __init__(self, top_k: int = 5, min_score: float = 0.35):
        self.top_k = top_k
        self.min_score = min_score

    def link(
        self,
        question: str,
        entities: list[dict[str, Any]],
    ) -> list[LinkedEntity]:
        normalized_question = _normalize(question)
        linked: list[LinkedEntity] = []
        for entity in entities:
            name = _entity_name(entity)
            canonical_name = _normalize(entity.get("canonical_name") or name)
            aliases = _as_list(entity.get("aliases"))
            terms = [name, entity.get("canonical_name", ""), *aliases]
            score = max(_score_term(normalized_question, term) for term in terms if term)
            if score < self.min_score:
                continue
            linked.append(
                LinkedEntity(
                    id=str(entity.get("id", "")),
                    name=name,
                    canonical_name=canonical_name,
                    type=entity.get("type"),
                    description=entity.get("description"),
                    score=round(score, 4),
                    aliases=aliases,
                    evidence_chunk_ids=_as_list(entity.get("evidence_chunk_ids")),
                )
            )
        linked.sort(key=lambda item: (-item.score, item.name))
        return linked[: self.top_k]


def _entity_name(entity: dict[str, Any]) -> str:
    return str(
        entity.get("name")
        or entity.get("title")
        or entity.get("canonical_name")
        or entity.get("id")
        or ""
    )


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _score_term(question: str, term: str) -> float:
    normalized_term = _normalize(term)
    if not normalized_term:
        return 0.0
    if normalized_term in question:
        return 1.0
    ratio = SequenceMatcher(None, question, normalized_term).ratio()
    if len(normalized_term) >= 4:
        ratio = max(ratio, SequenceMatcher(None, normalized_term, question).ratio())
    return ratio


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
