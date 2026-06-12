"""Tests for community reports."""

from graphrag_v2.community import Community, GraphProjection, MockCommunityReporter


def test_mock_community_reporter_generates_stable_report():
    community = Community(
        id="community_1",
        level=0,
        title="GraphRAG and Knowledge Graph Community",
        summary="",
        entity_ids=["e1", "e2"],
        relationship_ids=["r1"],
        text_unit_ids=["chunk_1"],
        size=2,
        rank=0.86,
    )
    projection = GraphProjection(
        entities=[
            {
                "id": "e1",
                "name": "GraphRAG",
                "type": "Technology",
                "description": "GraphRAG",
                "evidence_chunk_ids": ["chunk_1"],
            },
            {
                "id": "e2",
                "name": "Knowledge Graph",
                "type": "Technology",
                "description": "Knowledge Graph",
                "evidence_chunk_ids": ["chunk_1"],
            },
        ],
        relationships=[
            {
                "id": "r1",
                "source_name": "GraphRAG",
                "target_name": "Knowledge Graph",
                "relation": "uses",
                "description": "GraphRAG uses Knowledge Graph",
                "confidence": 0.93,
                "evidence_chunk_ids": ["chunk_1"],
            }
        ],
    )

    communities, reports = MockCommunityReporter().generate([community], projection)

    assert communities[0].summary.startswith("This community contains 2 entities")
    assert len(reports) == 1
    report = reports[0]
    assert report.id == "report_community_1"
    assert report.community_id == "community_1"
    assert report.key_entities == ["GraphRAG", "Knowledge Graph"]
    assert report.key_relationships == ["r1"]
    assert report.evidence_chunk_ids == ["chunk_1"]
    assert "Strongest relationship" in report.findings[1]
