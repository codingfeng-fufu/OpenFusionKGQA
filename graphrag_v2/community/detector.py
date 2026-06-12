"""Community detection from graph projections."""

from __future__ import annotations

import hashlib

import networkx as nx
from networkx.algorithms import community as nx_community

from graphrag_v2.community.models import Community, GraphProjection
from graphrag_v2.community.ranking import rank_community


def build_projection_graph(projection: GraphProjection) -> nx.Graph:
    """Build an undirected NetworkX projection from entities and relationships."""
    graph = nx.Graph()
    for entity in projection.entities:
        graph.add_node(
            entity["id"],
            name=entity.get("name", entity["id"]),
            canonical_name=entity.get("canonical_name", ""),
            type=entity.get("type", ""),
            description=entity.get("description", ""),
            evidence_chunk_ids=entity.get("evidence_chunk_ids") or [],
        )
    for relationship in projection.relationships:
        source_id = relationship["source_entity_id"]
        target_id = relationship["target_entity_id"]
        if source_id not in graph or target_id not in graph:
            continue
        confidence = relationship.get("confidence")
        extraction_count = relationship.get("extraction_count")
        weight = float(confidence or extraction_count or 1.0)
        graph.add_edge(
            source_id,
            target_id,
            id=relationship["id"],
            relation=relationship.get("relation", "related_to"),
            description=relationship.get("description", ""),
            confidence=float(confidence or 0.0),
            extraction_count=int(extraction_count or 0),
            evidence_chunk_ids=relationship.get("evidence_chunk_ids") or [],
            weight=weight,
        )
    return graph


def detect_communities(
    projection: GraphProjection,
    min_community_size: int = 2,
    max_level: int = 1,
) -> list[Community]:
    """Detect level-0 Louvain communities with stable IDs."""
    graph = build_projection_graph(projection)
    if graph.number_of_nodes() == 0:
        return []

    raw_communities: list[set[str]] = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component).copy()
        if subgraph.number_of_nodes() < min_community_size:
            continue
        if subgraph.number_of_edges() == 0:
            raw_communities.append(set(subgraph.nodes))
            continue
        detected = nx_community.louvain_communities(
            subgraph,
            weight="weight",
            seed=42,
        )
        raw_communities.extend(set(nodes) for nodes in detected)

    communities: list[Community] = []
    for nodes in raw_communities:
        if len(nodes) < min_community_size:
            continue
        entity_ids = sorted(nodes)
        relationship_ids = sorted(
            data["id"]
            for source, target, data in graph.edges(entity_ids, data=True)
            if source in nodes and target in nodes
        )
        text_unit_ids = sorted(
            {
                chunk_id
                for entity_id in entity_ids
                for chunk_id in graph.nodes[entity_id].get("evidence_chunk_ids", [])
            }
            | {
                chunk_id
                for source, target, data in graph.edges(entity_ids, data=True)
                if source in nodes and target in nodes
                for chunk_id in data.get("evidence_chunk_ids", [])
            }
        )
        names = sorted(graph.nodes[entity_id].get("name", entity_id) for entity_id in entity_ids)
        community_id = _stable_community_id(entity_ids)
        title = _community_title(names)
        rank = rank_community(graph, entity_ids, relationship_ids)
        communities.append(
            Community(
                id=community_id,
                level=0,
                title=title,
                summary="",
                entity_ids=entity_ids,
                relationship_ids=relationship_ids,
                text_unit_ids=text_unit_ids,
                size=len(entity_ids),
                rank=rank,
                metadata={
                    "algorithm": "louvain",
                    "max_level": max_level,
                    "entity_names": names,
                },
            )
        )

    return sorted(communities, key=lambda community: community.id)


def _stable_community_id(entity_ids: list[str]) -> str:
    digest = hashlib.sha256("|".join(entity_ids).encode("utf-8")).hexdigest()[:16]
    return f"community_{digest}"


def _community_title(names: list[str]) -> str:
    if not names:
        return "Community"
    if len(names) == 1:
        return f"{names[0]} Community"
    if len(names) == 2:
        return f"{names[0]} and {names[1]} Community"
    return f"{names[0]}, {names[1]} and Others"
