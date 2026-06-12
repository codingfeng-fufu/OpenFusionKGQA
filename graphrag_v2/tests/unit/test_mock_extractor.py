"""Tests for mock extraction."""

import pytest

from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction import MockExtractor


@pytest.mark.asyncio
async def test_mock_extractor_extracts_stable_candidates():
    text_unit = TextUnit(
        chunk_id="chunk_1",
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        chunk_index=0,
        text="GraphRAG uses Knowledge Graph evidence and Neo4j stores Graph Database data.",
        n_tokens=12,
    )
    extractor = MockExtractor()

    first = await extractor.extract(text_unit)
    second = await extractor.extract(text_unit)

    assert [entity.name for entity in first.entities] == [
        "GraphRAG",
        "Knowledge Graph",
        "Neo4j",
        "Graph Database",
    ]
    assert first.relationships[0].relation == "uses"
    assert first.relationships[-1].relation == "is_a"
    assert [entity.id for entity in first.entities] == [
        entity.id for entity in second.entities
    ]
    assert [triple.id for triple in first.triples] == [
        triple.id for triple in second.triples
    ]


@pytest.mark.asyncio
async def test_mock_extractor_returns_empty_for_no_terms():
    text_unit = TextUnit(
        chunk_id="chunk_1",
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        chunk_index=0,
        text="No configured terms are present.",
        n_tokens=5,
    )

    result = await MockExtractor().extract(text_unit)

    assert result.entities == []
    assert result.relationships == []
    assert result.triples == []
