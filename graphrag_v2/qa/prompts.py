"""Prompt helpers for optional LLM answer generation."""

from __future__ import annotations

from graphrag_v2.qa.models import CommunityEvidence, GraphEvidence, TextEvidence


QA_ANSWER_PROMPT_VERSION = "2026-06-09.qa.v4"
TEXT_EVIDENCE_MAX_CHARS = 1200


def build_prompt(
    question: str,
    route: str,
    graph_evidence: GraphEvidence,
    community_evidence: list[CommunityEvidence],
    text_evidence: list[TextEvidence],
) -> str:
    return "\n\n".join(
        [
            f"Question: {question}",
            f"Route: {route}",
            "Graph Evidence:\n" + _format_graph_evidence(graph_evidence),
            "Community Evidence:\n" + _format_community_evidence(community_evidence),
            "Text Evidence:\n" + _format_text_evidence(text_evidence),
            "\n".join(
                [
                    "Candidate answer extraction:",
                    "- Before refusing, scan the text evidence for names, dates, places, yes/no facts, and comparison outcomes that directly answer the question.",
                    "- If one or more candidates are supported, choose the best candidate from the structured candidate list and answer concisely.",
                    "- Say insufficient evidence only if no candidate answer is grounded in the provided evidence.",
                    "When a structured candidate list is provided, return the selected candidate id as requested; do not invent answers outside the list.",
                ]
            ),
        ]
    )


def _format_graph_evidence(graph_evidence: GraphEvidence) -> str:
    lines = []
    for entity in graph_evidence.linked_entities:
        lines.append(
            f"- Entity {entity.id}: {entity.name} ({entity.type or 'unknown'}) score={entity.score:.2f}"
        )
    for relationship in graph_evidence.relationships:
        lines.append(
            f"- Relationship {relationship.id}: {relationship.source_name} "
            f"{relationship.relation} {relationship.target_name} score={relationship.score:.2f}"
        )
    return "\n".join(lines) if lines else "- none"


def _format_community_evidence(community_evidence: list[CommunityEvidence]) -> str:
    lines = []
    for community in community_evidence:
        lines.append(
            f"- Report {community.report_id}: {community.title} score={community.score:.2f}"
        )
    return "\n".join(lines) if lines else "- none"


def _format_text_evidence(text_evidence: list[TextEvidence]) -> str:
    lines = []
    for item in text_evidence:
        text = item.text[:TEXT_EVIDENCE_MAX_CHARS]
        lines.append(f"- {item.chunk_id}: {text}")
    return "\n".join(lines) if lines else "- none"
