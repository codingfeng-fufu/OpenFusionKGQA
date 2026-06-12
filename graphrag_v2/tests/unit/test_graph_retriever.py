"""Tests for QA graph retrieval."""

import pytest

from graphrag_v2.qa import GraphRetriever, LinkedEntity


def test_graph_retriever_collects_relevant_relationships_and_chunks():
    linked_entities = [
        LinkedEntity(
            id="entity_1",
            name="GraphRAG",
            canonical_name="graphrag",
            type="Technology",
            description="GraphRAG",
            score=1.0,
            evidence_chunk_ids=["chunk_1"],
        ),
        LinkedEntity(
            id="entity_2",
            name="微软",
            canonical_name="微软",
            type="Organization",
            description="微软",
            score=1.0,
            evidence_chunk_ids=["chunk_2"],
        ),
    ]
    relationships = [
        {
            "id": "rel_1",
            "source_entity_id": "entity_2",
            "target_entity_id": "entity_1",
            "source_name": "微软",
            "target_name": "GraphRAG",
            "relation": "uses",
            "description": "微软开发了 GraphRAG",
            "confidence": 0.9,
            "extraction_count": 1,
            "evidence_chunk_ids": ["chunk_1"],
        },
        {
            "id": "rel_2",
            "source_entity_id": "entity_1",
            "target_entity_id": "entity_3",
            "source_name": "GraphRAG",
            "target_name": "Leiden算法",
            "relation": "uses",
            "description": "GraphRAG uses Leiden",
            "confidence": 0.8,
            "extraction_count": 1,
            "evidence_chunk_ids": ["chunk_3"],
        },
    ]

    evidence = GraphRetriever(top_k_relationships=2).retrieve(linked_entities, relationships)

    assert [entity.id for entity in evidence.linked_entities] == ["entity_1", "entity_2"]
    assert {relationship.id for relationship in evidence.relationships} == {"rel_1", "rel_2"}
    assert set(evidence.text_chunk_ids) == {"chunk_1", "chunk_2", "chunk_3"}


def test_graph_retriever_keeps_simple_questions_to_one_hop():
    linked_entities = _comparison_linked_entities()
    relationships = _comparison_chain_relationships()

    evidence = GraphRetriever(top_k_relationships=6).retrieve(
        linked_entities,
        relationships,
        question="What is Film A?",
    )

    assert {relationship.id for relationship in evidence.relationships} == {
        "rel_film_pair",
        "rel_film_b_director_a",
    }
    assert "rel_director_a_director_b" not in {
        relationship.id for relationship in evidence.relationships
    }
    assert "chunk_director_b" not in evidence.text_chunk_ids
    assert evidence.retrieval_metadata == {
        "adaptive_enabled": True,
        "adaptive_triggered": False,
        "matched_adaptive_cues": [],
        "hop_plan": [1],
        "relationship_count_by_hop": {1: 2},
        "max_retrieved_hop": 1,
    }


def test_graph_retriever_does_not_expand_plain_which_questions():
    linked_entities = _comparison_linked_entities()
    relationships = _comparison_chain_relationships()

    evidence = GraphRetriever(top_k_relationships=6).retrieve(
        linked_entities,
        relationships,
        question="Which country is Film A from?",
    )

    assert {relationship.id for relationship in evidence.relationships} == {
        "rel_film_pair",
        "rel_film_b_director_a",
    }
    assert "chunk_director_b" not in evidence.text_chunk_ids


def test_graph_retriever_expands_comparison_questions_to_second_hop():
    linked_entities = _comparison_linked_entities()
    relationships = _comparison_chain_relationships()

    evidence = GraphRetriever(top_k_relationships=6).retrieve(
        linked_entities,
        relationships,
        question="Which film has the director who died later, Film A or Film B?",
    )

    by_id = {relationship.id: relationship for relationship in evidence.relationships}
    assert set(by_id) == {
        "rel_film_pair",
        "rel_film_b_director_a",
        "rel_director_a_director_b",
    }
    assert by_id["rel_director_a_director_b"].hop == 2
    assert "chunk_director_b" in evidence.text_chunk_ids
    assert evidence.retrieval_metadata["adaptive_enabled"] is True
    assert evidence.retrieval_metadata["adaptive_triggered"] is True
    assert "died later" in evidence.retrieval_metadata["matched_adaptive_cues"]
    assert evidence.retrieval_metadata["hop_plan"] == [1, 2, 3]
    assert evidence.retrieval_metadata["relationship_count_by_hop"] == {1: 2, 2: 1}
    assert evidence.retrieval_metadata["max_retrieved_hop"] == 2


@pytest.mark.parametrize(
    "question, expected_cue",
    [
        ("Where was Film A's director born?", "director"),
        ("Where did Company A's founder die?", "founder"),
        ("Who is Alice's sibling in law?", "sibling in law"),
        ("What nationality is Film A's composer?", "nationality"),
        ("Where was Person A's mother born?", "mother"),
        ("Where did Person A's spouse die?", "spouse"),
    ],
)
def test_graph_retriever_adaptive_query_planner_covers_chain_question_variants(
    question,
    expected_cue,
):
    evidence = GraphRetriever(top_k_relationships=6).retrieve(
        _comparison_linked_entities(),
        _comparison_chain_relationships(),
        question=question,
    )

    assert evidence.retrieval_metadata["adaptive_enabled"] is True
    assert evidence.retrieval_metadata["adaptive_triggered"] is True
    assert expected_cue in evidence.retrieval_metadata["matched_adaptive_cues"]
    assert evidence.retrieval_metadata["hop_plan"] == [1, 2, 3]


def test_graph_retriever_adaptive_query_planner_retrieves_third_hop_evidence():
    linked_entities = [
        LinkedEntity(
            id="entity_film",
            name="Film A",
            canonical_name="film a",
            type="Film",
            description="Film A",
            score=1.0,
            evidence_chunk_ids=["chunk_film"],
        )
    ]
    relationships = [
        _relationship("rel_film_director", "entity_film", "entity_director", "Film A", "Director A", "chunk_director"),
        _relationship("rel_director_mother", "entity_director", "entity_mother", "Director A", "Mother A", "chunk_mother"),
        _relationship("rel_mother_birthplace", "entity_mother", "entity_city", "Mother A", "City A", "chunk_city"),
    ]

    evidence = GraphRetriever(top_k_relationships=6).retrieve(
        linked_entities,
        relationships,
        question="Where was Film A's director's mother born?",
    )

    by_id = {relationship.id: relationship for relationship in evidence.relationships}
    assert by_id["rel_film_director"].hop == 1
    assert by_id["rel_director_mother"].hop == 2
    assert by_id["rel_mother_birthplace"].hop == 3
    assert "chunk_city" in evidence.text_chunk_ids
    assert evidence.retrieval_metadata["relationship_count_by_hop"] == {1: 1, 2: 1, 3: 1}
    assert evidence.retrieval_metadata["max_retrieved_hop"] == 3


def test_graph_retriever_adaptive_selection_keeps_deeper_hop_when_top_k_is_tight():
    linked_entities = [
        LinkedEntity(
            id="entity_film",
            name="Film A",
            canonical_name="film a",
            type="Film",
            description="Film A",
            score=1.0,
            evidence_chunk_ids=["chunk_film"],
        )
    ]
    relationships = [
        _relationship("rel_1_high_score_extra", "entity_film", "entity_extra_1", "Film A", "Extra 1", "chunk_extra_1"),
        _relationship("rel_2_high_score_extra", "entity_film", "entity_extra_2", "Film A", "Extra 2", "chunk_extra_2"),
        _relationship("rel_3_film_director", "entity_film", "entity_director", "Film A", "Director A", "chunk_director"),
        _relationship("rel_4_director_birthplace", "entity_director", "entity_city", "Director A", "City A", "chunk_city"),
    ]

    evidence = GraphRetriever(top_k_relationships=2).retrieve(
        linked_entities,
        relationships,
        question="Where was Film A's director born?",
    )

    by_id = {relationship.id: relationship for relationship in evidence.relationships}
    assert "rel_4_director_birthplace" in by_id
    assert by_id["rel_4_director_birthplace"].hop == 2
    assert "chunk_city" in evidence.text_chunk_ids
    assert evidence.retrieval_metadata["relationship_count_by_hop"] == {1: 1, 2: 1}


def test_graph_retriever_adaptive_selection_fills_remaining_slots_by_global_score():
    linked_entities = [
        LinkedEntity(
            id="entity_film",
            name="Film A",
            canonical_name="film a",
            type="Film",
            description="Film A",
            score=1.0,
            evidence_chunk_ids=["chunk_film"],
        )
    ]
    relationships = [
        _relationship("rel_1_high_score_extra", "entity_film", "entity_extra", "Film A", "Extra", "chunk_extra"),
        _relationship("rel_5_high_score_extra", "entity_film", "entity_extra_2", "Film A", "Extra 2", "chunk_extra_2"),
        _relationship("rel_2_film_director", "entity_film", "entity_director", "Film A", "Director A", "chunk_director"),
        _relationship("rel_3_director_mother", "entity_director", "entity_mother", "Director A", "Mother A", "chunk_mother"),
        _relationship("rel_4_mother_birthplace", "entity_mother", "entity_city", "Mother A", "City A", "chunk_city"),
        _relationship("rel_0_low_score_branch", "entity_extra", "entity_branch", "Extra", "Branch", "chunk_branch"),
    ]

    evidence = GraphRetriever(top_k_relationships=5).retrieve(
        linked_entities,
        relationships,
        question="Where was Film A's director's mother born?",
    )

    selected_ids = {relationship.id for relationship in evidence.relationships}
    assert selected_ids == {
        "rel_1_high_score_extra",
        "rel_5_high_score_extra",
        "rel_2_film_director",
        "rel_3_director_mother",
        "rel_4_mother_birthplace",
    }
    assert "rel_0_low_score_branch" not in selected_ids


def test_graph_retriever_adaptive_selection_keeps_complete_deep_path_when_it_exceeds_top_k():
    linked_entities = [
        LinkedEntity(
            id="entity_film",
            name="Film A",
            canonical_name="film a",
            type="Film",
            description="Film A",
            score=1.0,
            evidence_chunk_ids=["chunk_film"],
        )
    ]
    relationships = [
        _relationship("rel_film_director", "entity_film", "entity_director", "Film A", "Director A", "chunk_director"),
        _relationship("rel_director_mother", "entity_director", "entity_mother", "Director A", "Mother A", "chunk_mother"),
        _relationship("rel_mother_birthplace", "entity_mother", "entity_city", "Mother A", "City A", "chunk_city"),
    ]

    evidence = GraphRetriever(top_k_relationships=2).retrieve(
        linked_entities,
        relationships,
        question="Where was Film A's director's mother born?",
    )

    by_id = {relationship.id: relationship for relationship in evidence.relationships}
    assert list(by_id) == [
        "rel_film_director",
        "rel_director_mother",
        "rel_mother_birthplace",
    ]
    assert [relationship.hop for relationship in evidence.relationships] == [1, 2, 3]
    assert evidence.retrieval_metadata["relationship_count_by_hop"] == {1: 1, 2: 1, 3: 1}


def _comparison_linked_entities():
    return [
        LinkedEntity(
            id="entity_film_a",
            name="Film A",
            canonical_name="film a",
            type="Film",
            description="Film A",
            score=1.0,
            evidence_chunk_ids=["chunk_film_a"],
        ),
        LinkedEntity(
            id="entity_film_b",
            name="Film B",
            canonical_name="film b",
            type="Film",
            description="Film B",
            score=1.0,
            evidence_chunk_ids=["chunk_film_b"],
        ),
    ]


def _comparison_chain_relationships():
    return [
        {
            "id": "rel_film_pair",
            "source_entity_id": "entity_film_a",
            "target_entity_id": "entity_film_b",
            "source_name": "Film A",
            "target_name": "Film B",
            "relation": "supports_answer",
            "description": "Film A and Film B are compared.",
            "confidence": 1.0,
            "extraction_count": 1,
            "evidence_chunk_ids": ["chunk_film_a", "chunk_film_b"],
        },
        {
            "id": "rel_film_b_director_a",
            "source_entity_id": "entity_film_b",
            "target_entity_id": "entity_director_a",
            "source_name": "Film B",
            "target_name": "Director A",
            "relation": "supports_answer",
            "description": "Film B requires Director A evidence.",
            "confidence": 1.0,
            "extraction_count": 1,
            "evidence_chunk_ids": ["chunk_film_b", "chunk_director_a"],
        },
        {
            "id": "rel_director_a_director_b",
            "source_entity_id": "entity_director_a",
            "target_entity_id": "entity_director_b",
            "source_name": "Director A",
            "target_name": "Director B",
            "relation": "supports_answer",
            "description": "Director B died in 2012.",
            "confidence": 1.0,
            "extraction_count": 1,
            "evidence_chunk_ids": ["chunk_director_a", "chunk_director_b"],
        },
    ]


def _relationship(
    rel_id: str,
    source_id: str,
    target_id: str,
    source_name: str,
    target_name: str,
    chunk_id: str,
):
    return {
        "id": rel_id,
        "source_entity_id": source_id,
        "target_entity_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "relation": "supports_answer",
        "description": f"{source_name} supports {target_name}.",
        "confidence": 1.0,
        "extraction_count": 1,
        "evidence_chunk_ids": [chunk_id],
    }
