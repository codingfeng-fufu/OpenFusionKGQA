"""Community evidence retrieval for QA."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from graphrag_v2.qa.models import CommunityEvidence


class CommunityRetriever:
    """Retrieve top community reports for global questions."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        communities: list[dict[str, Any]],
        reports: list[dict[str, Any]],
    ) -> list[CommunityEvidence]:
        report_by_community = {
            str(report.get("community_id", "")): report for report in reports
        }
        community_by_id = {str(community.get("id", "")): community for community in communities}
        scored: list[CommunityEvidence] = []
        for report in reports:
            community_id = str(report.get("community_id", ""))
            community = community_by_id.get(community_id, {})
            rank = float(community.get("rank") or report.get("rank") or 0.0)
            rank_score = min(max(rank, 0.0), 1.0)
            text = " ".join(
                [
                    str(report.get("title", "")),
                    str(report.get("summary", "")),
                    str(report.get("full_content", "")),
                    " ".join(_as_list(report.get("findings"))),
                    " ".join(_as_list(report.get("key_entities"))),
                    " ".join(_as_list(report.get("key_relationships"))),
                ]
            )
            text_score = SequenceMatcher(None, _normalize(question), _normalize(text)).ratio()
            keyword_score = _keyword_overlap(question, text)
            score = round(min(1.0, 0.5 * rank_score + 0.3 * text_score + 0.2 * keyword_score), 4)
            scored.append(
                CommunityEvidence(
                    community_id=community_id,
                    report_id=str(report.get("id", f"report_{community_id}")),
                    title=str(report.get("title", "")),
                    summary=str(report.get("summary", "")),
                    full_content=str(report.get("full_content", "")),
                    findings=_as_list(report.get("findings")),
                    key_entities=_as_list(report.get("key_entities")),
                    key_relationships=_as_list(report.get("key_relationships")),
                    rank=rank,
                    score=score,
                    evidence_chunk_ids=_as_list(report.get("evidence_chunk_ids")),
                    metadata={},
                )
            )
        scored.sort(key=lambda item: (-item.score, item.report_id))
        return scored[: self.top_k]


def _keyword_overlap(question: str, text: str) -> float:
    normalized_question = _normalize(question)
    normalized_text = _normalize(text)
    if not normalized_question or not normalized_text:
        return 0.0
    question_terms = [term for term in normalized_question.split() if term]
    if not question_terms:
        return 0.0
    matched = sum(1 for term in question_terms if term in normalized_text)
    return matched / len(question_terms)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


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
