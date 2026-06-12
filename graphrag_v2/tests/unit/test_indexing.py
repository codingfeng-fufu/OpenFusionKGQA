"""Tests for indexing orchestration helpers."""

from __future__ import annotations

import asyncio

import pytest

from graphrag_v2.config import GraphRagConfig
from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction.models import ExtractionResult
from graphrag_v2.graph_store import GraphStoreStats
from graphrag_v2.indexing import (
    _extract_candidates,
    _graph_store_metadata,
    index_extraction_only,
)


class TrackingExtractor:
    """Extractor stub that records active calls for concurrency assertions."""

    concurrent_requests = 2

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def extract(self, text_unit: TextUnit) -> ExtractionResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ExtractionResult()


class OneChunkFailsExtractor:
    """Extractor stub that fails one chunk and records failed chunk metadata."""

    concurrent_requests = 1

    class Stats:
        def __init__(self):
            self.failed_chunks = 0
            self.failed_chunk_ids: list[str] = []

    def __init__(self):
        self.stats = self.Stats()
        self.calls = 0

    async def extract(self, text_unit: TextUnit) -> ExtractionResult:
        self.calls += 1
        if text_unit.chunk_id == "chunk_1" or self.calls == 2:
            self.stats.failed_chunks += 1
            self.stats.failed_chunk_ids.append(text_unit.chunk_id)
            raise ValueError("parse failed")
        return ExtractionResult()

    def get_metadata(self) -> dict:
        return {
            "extraction_failed_chunks": self.stats.failed_chunks,
            "extraction_failed_chunk_ids": list(self.stats.failed_chunk_ids),
        }


def test_graph_store_metadata_includes_health_schema_and_write_strategy():
    stats = GraphStoreStats(
        provider="neo4j",
        index_id="kgqa_demo",
        database="neo4j",
        num_text_units=2,
        num_entities=3,
        num_relationships=1,
        num_rejected_triples=0,
        schema_ready=False,
        schema_constraints=["kgqa_index_id_unique"],
        schema_indexes=["kgqa_entity_scoped_name_index"],
        schema_version="2026-06-06.v1",
        missing_schema_constraints=["kgqa_relationship_scoped_id_unique"],
        missing_schema_indexes=["kgqa_text_unit_scoped_id_index"],
        health_status="degraded",
        write_strategy="staged_replace",
        staging_index_id="kgqa_demo__staging__123",
    )

    metadata = _graph_store_metadata(stats)

    assert metadata["graph_store_health_status"] == "degraded"
    assert metadata["graph_store_schema_version"] == "2026-06-06.v1"
    assert metadata["graph_store_missing_schema_constraints"] == [
        "kgqa_relationship_scoped_id_unique"
    ]
    assert metadata["graph_store_missing_schema_indexes"] == [
        "kgqa_text_unit_scoped_id_index"
    ]
    assert metadata["graph_store_write_strategy"] == "staged_replace"
    assert metadata["graph_store_staging_index_id"] == "kgqa_demo__staging__123"


@pytest.mark.asyncio
async def test_extract_candidates_honors_concurrent_requests():
    extractor = TrackingExtractor()
    text_units = [
        TextUnit(
            chunk_id=f"chunk_{index}",
            doc_id="doc_1",
            source_path="/tmp/doc.md",
            chunk_index=index,
            text="GraphRAG uses Knowledge Graph evidence.",
            n_tokens=8,
        )
        for index in range(4)
    ]

    entities, relationships, triples = await _extract_candidates(
        text_units=text_units,
        extractor=extractor,
        fail_on_invalid_chunk=True,
    )

    assert entities == []
    assert relationships == []
    assert triples == []
    assert extractor.max_active == 2


@pytest.mark.asyncio
async def test_extract_candidates_tolerates_failed_chunk_and_records_metadata():
    extractor = OneChunkFailsExtractor()
    text_units = [
        TextUnit(
            chunk_id=f"chunk_{index}",
            doc_id="doc_1",
            source_path="/tmp/doc.md",
            chunk_index=index,
            text="GraphRAG uses Knowledge Graph evidence.",
            n_tokens=8,
        )
        for index in range(3)
    ]

    entities, relationships, triples = await _extract_candidates(
        text_units=text_units,
        extractor=extractor,
        fail_on_invalid_chunk=False,
    )

    assert entities == []
    assert relationships == []
    assert triples == []
    assert extractor.get_metadata()["extraction_failed_chunks"] == 1
    assert extractor.get_metadata()["extraction_failed_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_index_extraction_only_persists_failed_chunk_stats_when_tolerant(
    tmp_path,
    monkeypatch,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("First document.", encoding="utf-8")
    (docs / "two.md").write_text("Second document.", encoding="utf-8")
    output = tmp_path / "index"
    extractor = OneChunkFailsExtractor()
    config = GraphRagConfig()
    config.extraction.fail_on_invalid_chunk = False

    import graphrag_v2.indexing as indexing

    monkeypatch.setattr(indexing, "_create_extractor", lambda *_args, **_kwargs: extractor)

    metadata = await index_extraction_only(
        input_path=docs,
        output_path=output,
        config=config,
        extractor_name="llm",
    )

    assert metadata["run_status"] == "succeeded"
    assert metadata["extraction_failed_chunks"] == 1
    assert len(metadata["extraction_failed_chunk_ids"]) == 1
