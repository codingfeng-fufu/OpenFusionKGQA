"""Tests for QA entity linking."""

from graphrag_v2.qa import EntityLinker


def test_entity_linker_matches_exact_mentions():
    linker = EntityLinker(top_k=5)
    entities = [
        {
            "id": "entity_1",
            "name": "GraphRAG",
            "canonical_name": "graphrag",
            "type": "Technology",
            "description": "GraphRAG",
            "aliases": ["GraphRAG"],
            "evidence_chunk_ids": ["chunk_1"],
        },
        {
            "id": "entity_2",
            "name": "微软",
            "canonical_name": "微软",
            "type": "Organization",
            "description": "微软",
            "aliases": ["Microsoft"],
            "evidence_chunk_ids": ["chunk_2"],
        },
    ]

    linked = linker.link("GraphRAG 是微软开发的吗？", entities)

    assert {item.name for item in linked} == {"GraphRAG", "微软"}
    assert all(item.score == 1.0 for item in linked)
    assert linked[0].evidence_chunk_ids
