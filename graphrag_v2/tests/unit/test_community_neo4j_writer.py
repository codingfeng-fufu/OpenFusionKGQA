"""Tests for Neo4j community write-back."""

from pathlib import Path

import pytest

from graphrag_v2.artifacts import compute_index_id
from graphrag_v2.community import Community, CommunityReport, Neo4jCommunityStore
from graphrag_v2.community.neo4j_writer import (
    COMMUNITY_CONSTRAINTS,
    CLEAR_COMMUNITIES,
    CLEAR_COMMUNITY_REPORTS,
    MERGE_COMMUNITY,
    MERGE_COMMUNITY_REPORT,
)
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store import GraphStoreError


def test_neo4j_community_store_reports_missing_password(temp_dir: Path, monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_MISSING_NEO4J_PASSWORD")
    monkeypatch.delenv(config.password_env, raising=False)

    with pytest.raises(GraphStoreError, match="password environment variable"):
        Neo4jCommunityStore(config, index_path=temp_dir).get_stats()


def test_neo4j_community_store_reads_projection_and_writes(temp_dir: Path, monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver()

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    store = Neo4jCommunityStore(config, index_path=temp_dir)
    projection = store.read_projection()
    stats = store.write([_community()], [_report()])

    assert len(projection.entities) == 2
    assert len(projection.relationships) == 1
    assert stats == {"num_communities": 1, "num_community_reports": 1}
    queries = [call[0] for call in fake_driver.session_obj.writes]
    assert queries[: len(COMMUNITY_CONSTRAINTS)] == COMMUNITY_CONSTRAINTS
    assert CLEAR_COMMUNITY_REPORTS in queries
    assert CLEAR_COMMUNITIES in queries
    assert MERGE_COMMUNITY in queries
    assert MERGE_COMMUNITY_REPORT in queries
    scoped_params = [
        params
        for _, params in fake_driver.session_obj.writes
        if params.get("id") == "community_1"
    ][0]
    assert scoped_params["index_id"] == compute_index_id(temp_dir)


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()
        self.closed = False

    def session(self, database: str):
        self.session_obj.database = database
        return self.session_obj

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.writes = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute_read(self, func, *args):
        return func(FakeTx(), *args)

    def execute_write(self, func, query, params):
        self.writes.append((query, params))
        return func(FakeTx(), query, params)


class FakeTx:
    def run(self, query, **params):
        return FakeResult(query)


class FakeResult:
    def __init__(self, query):
        self.query = query

    def __iter__(self):
        if "AS entity" in self.query:
            return iter(
                [
                    {"entity": {"id": "e1", "name": "GraphRAG"}},
                    {"entity": {"id": "e2", "name": "Knowledge Graph"}},
                ]
            )
        if "AS relationship" in self.query:
            return iter(
                [
                    {
                        "relationship": {
                            "id": "r1",
                            "source_entity_id": "e1",
                            "target_entity_id": "e2",
                        }
                    }
                ]
            )
        return iter([])

    def consume(self):
        return None

    def single(self):
        return {"num_communities": 1, "num_community_reports": 1}


def _community() -> Community:
    return Community(
        id="community_1",
        level=0,
        title="GraphRAG Community",
        summary="summary",
        entity_ids=["e1", "e2"],
        relationship_ids=["r1"],
        text_unit_ids=["chunk_1"],
        size=2,
        rank=0.86,
    )


def _report() -> CommunityReport:
    return CommunityReport(
        id="report_community_1",
        community_id="community_1",
        title="GraphRAG Community",
        summary="summary",
        full_content="full",
        findings=["finding"],
        key_entities=["GraphRAG"],
        key_relationships=["r1"],
        evidence_chunk_ids=["chunk_1"],
        rank=0.86,
    )
