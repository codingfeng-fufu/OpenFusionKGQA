"""Community ranking helpers."""

from __future__ import annotations

import networkx as nx


def rank_community(graph: nx.Graph, entity_ids: list[str], relationship_ids: list[str]) -> float:
    """Rank a community by size, density, confidence, and evidence coverage."""
    if not entity_ids:
        return 0.0

    subgraph = graph.subgraph(entity_ids)
    size_score = min(len(entity_ids) / 5.0, 1.0)
    density_score = nx.density(subgraph) if len(entity_ids) > 1 else 0.0
    edge_confidences = [
        float(data.get("confidence") or 0.0)
        for _, _, data in subgraph.edges(data=True)
    ]
    confidence_score = (
        sum(edge_confidences) / len(edge_confidences) if edge_confidences else 0.0
    )
    evidence_count = len(
        {
            chunk_id
            for _, _, data in subgraph.edges(data=True)
            for chunk_id in data.get("evidence_chunk_ids", [])
        }
    )
    evidence_score = min(evidence_count / max(len(relationship_ids), 1), 1.0)
    score = (
        0.40 * size_score
        + 0.20 * density_score
        + 0.25 * confidence_score
        + 0.15 * evidence_score
    )
    return round(score, 4)
