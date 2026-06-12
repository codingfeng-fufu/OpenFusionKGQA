"""Tests for community detection."""

from graphrag_v2.community import GraphProjection, build_projection_graph, detect_communities


def test_detect_communities_filters_small_components_and_stable_ids():
    projection = GraphProjection(
        entities=[
            _entity("e1", "GraphRAG", ["chunk_1"]),
            _entity("e2", "Knowledge Graph", ["chunk_1"]),
            _entity("e3", "Neo4j", ["chunk_2"]),
            _entity("e4", "Graph Database", ["chunk_2"]),
            _entity("e5", "JSON", []),
        ],
        relationships=[
            _relationship("r1", "e1", "e2", 0.9, ["chunk_1"]),
            _relationship("r2", "e3", "e4", 0.8, ["chunk_2"]),
        ],
    )

    communities = detect_communities(projection, min_community_size=2)
    repeated = detect_communities(projection, min_community_size=2)

    assert [community.id for community in communities] == [
        community.id for community in repeated
    ]
    assert len(communities) == 2
    assert all(community.size == 2 for community in communities)
    assert all("e5" not in community.entity_ids for community in communities)
    assert {tuple(community.relationship_ids) for community in communities} == {
        ("r1",),
        ("r2",),
    }


def test_build_projection_graph_uses_confidence_as_weight():
    projection = GraphProjection(
        entities=[_entity("e1", "GraphRAG"), _entity("e2", "Knowledge Graph")],
        relationships=[_relationship("r1", "e1", "e2", 0.73, ["chunk_1"])],
    )

    graph = build_projection_graph(projection)

    assert graph["e1"]["e2"]["weight"] == 0.73
    assert graph["e1"]["e2"]["evidence_chunk_ids"] == ["chunk_1"]


def _entity(id_: str, name: str, evidence_chunk_ids=None) -> dict:
    return {
        "id": id_,
        "name": name,
        "canonical_name": name.lower(),
        "type": "Technology",
        "description": name,
        "evidence_chunk_ids": evidence_chunk_ids or [],
    }


def _relationship(
    id_: str,
    source_id: str,
    target_id: str,
    confidence: float,
    evidence_chunk_ids=None,
) -> dict:
    return {
        "id": id_,
        "source_entity_id": source_id,
        "target_entity_id": target_id,
        "source_name": source_id,
        "target_name": target_id,
        "relation": "uses",
        "description": "uses",
        "confidence": confidence,
        "extraction_count": 1,
        "evidence_chunk_ids": evidence_chunk_ids or [],
        "original_relations": ["uses"],
    }
