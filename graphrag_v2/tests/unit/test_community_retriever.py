"""Tests for QA community retrieval."""

from graphrag_v2.qa import CommunityRetriever


def test_community_retriever_scores_and_returns_top_report():
    retriever = CommunityRetriever(top_k=1)
    communities = [
        {
            "id": "community_1",
            "rank": 0.9,
            "size": 2,
            "title": "GraphRAG 技术社区",
        }
    ]
    reports = [
        {
            "id": "report_1",
            "community_id": "community_1",
            "title": "GraphRAG 技术社区",
            "summary": "GraphRAG 与知识图谱相关",
            "full_content": "GraphRAG 与知识图谱相关",
            "findings": ["GraphRAG uses knowledge graphs"],
            "key_entities": ["GraphRAG"],
            "key_relationships": ["rel_1"],
            "evidence_chunk_ids": ["chunk_1"],
            "rank": 0.9,
        }
    ]

    evidence = retriever.retrieve(
        "这批文档主要讲了哪些主题？",
        communities,
        reports,
    )

    assert len(evidence) == 1
    assert evidence[0].report_id == "report_1"
    assert evidence[0].score > 0
