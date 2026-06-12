"""Tests for QA routing."""

from graphrag_v2.qa import LinkedEntity, QueryRouter


def test_query_router_prefers_local_for_linked_entities():
    router = QueryRouter()
    decision = router.route(
        "GraphRAG 是什么？",
        linked_entities=[
            LinkedEntity(
                id="entity_1",
                name="GraphRAG",
                canonical_name="graphrag",
                type="Technology",
                description="desc",
                score=1.0,
            )
        ],
        community_report_count=1,
    )

    assert decision.route == "local"
    assert "Linked entities" in decision.reason


def test_query_router_prefers_global_for_summary_questions():
    router = QueryRouter()
    decision = router.route("这批文档主要讲了哪些主题？", community_report_count=2)

    assert decision.route == "global"
    assert "global cue" in decision.reason
