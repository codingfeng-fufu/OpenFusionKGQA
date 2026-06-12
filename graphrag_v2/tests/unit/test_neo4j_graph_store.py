"""Tests for the Neo4j graph store skeleton."""

from pathlib import Path

import pytest

from graphrag_v2.artifacts import compute_index_id
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_fusion.models import FusedEntity, FusedRelationship, FusionResult
from graphrag_v2.graph_store import GraphStoreError, Neo4jGraphStore
from graphrag_v2.graph_store.cypher import (
    CLEAR_INDEX,
    CLEAR_STAGING_INDEX,
    CONSTRAINTS,
    EXPECTED_CONSTRAINT_NAMES,
    EXPECTED_INDEX_NAMES,
    INDEXES,
    MERGE_ENTITY,
    MERGE_INDEX,
    MERGE_RELATIONSHIP,
    PROMOTE_STAGED_INDEX,
    PROMOTE_STAGED_NODES,
    PROMOTE_STAGED_RELATIONSHIPS,
)


def test_neo4j_store_reports_missing_password(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_MISSING_NEO4J_PASSWORD")
    monkeypatch.delenv(config.password_env, raising=False)
    store = Neo4jGraphStore(config)

    with pytest.raises(GraphStoreError) as exc_info:
        store.get_stats()

    assert "Neo4j password environment variable is not set" in str(exc_info.value)


def test_neo4j_store_writes_constraints_entities_and_relationships(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver()

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    index_path = Path("/tmp/kgqa-test-index")
    store = Neo4jGraphStore(config, index_path=index_path)
    stats = store.write_graph(_fusion_result())

    assert stats.provider == "neo4j"
    assert stats.index_id == compute_index_id(index_path)
    assert stats.database == "neo4j"
    assert stats.num_text_units == 0
    assert stats.num_entities == 2
    assert stats.num_relationships == 1
    assert stats.schema_version == "2026-06-06.v1"
    assert stats.health_status == "ready"
    assert stats.schema_ready is True
    assert stats.schema_constraints == EXPECTED_CONSTRAINT_NAMES
    assert stats.schema_indexes == EXPECTED_INDEX_NAMES
    assert stats.expected_schema_constraints == EXPECTED_CONSTRAINT_NAMES
    assert stats.expected_schema_indexes == EXPECTED_INDEX_NAMES
    assert stats.missing_schema_constraints == []
    assert stats.missing_schema_indexes == []
    assert fake_driver.closed is True
    queries = [call[0] for call in fake_driver.session_obj.writes]
    assert queries[: len(CONSTRAINTS)] == CONSTRAINTS
    assert queries[len(CONSTRAINTS) : len(CONSTRAINTS) + len(INDEXES)] == INDEXES
    assert CLEAR_INDEX in queries
    assert CLEAR_STAGING_INDEX in queries
    assert PROMOTE_STAGED_NODES in queries
    assert PROMOTE_STAGED_RELATIONSHIPS in queries
    assert PROMOTE_STAGED_INDEX in queries
    assert MERGE_ENTITY in queries
    assert MERGE_RELATIONSHIP in queries
    assert fake_driver.session_obj.write_options[0]["timeout"] == 30.0
    assert fake_driver.session_obj.write_options[0]["metadata"]["operation"] == "ensure_schema"
    scoped_params = _find_batch_record(fake_driver.session_obj.writes, "entity_a")
    assert scoped_params["index_id"] == stats.staging_index_id


def test_neo4j_store_get_stats_uses_driver(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver()

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    index_path = Path("/tmp/kgqa-test-index")
    stats = Neo4jGraphStore(config, index_path=index_path).get_stats()

    assert stats.provider == "neo4j"
    assert stats.index_id == compute_index_id(index_path)
    assert stats.num_entities == 2
    assert stats.num_relationships == 1
    assert stats.schema_version == "2026-06-06.v1"
    assert stats.health_status == "ready"
    assert stats.schema_ready is True
    assert stats.expected_schema_constraints == EXPECTED_CONSTRAINT_NAMES
    assert stats.expected_schema_indexes == EXPECTED_INDEX_NAMES
    assert stats.missing_schema_constraints == []
    assert stats.missing_schema_indexes == []
    assert fake_driver.closed is True


def test_neo4j_store_reports_missing_schema_names(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_session = FakeSession(
        constraint_names=EXPECTED_CONSTRAINT_NAMES[:-1],
        index_names=EXPECTED_INDEX_NAMES[:-1],
    )
    fake_driver = FakeDriver(session_obj=fake_session)

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    stats = Neo4jGraphStore(config, index_path=Path("/tmp/kgqa-test-index")).get_stats()

    assert stats.schema_ready is False
    assert stats.health_status == "degraded"
    assert stats.missing_schema_constraints == [EXPECTED_CONSTRAINT_NAMES[-1]]
    assert stats.missing_schema_indexes == [EXPECTED_INDEX_NAMES[-1]]


def test_neo4j_store_retries_transient_write_failure(monkeypatch):
    config = GraphStoreConfig(
        password_env="KGQA_TEST_NEO4J_PASSWORD",
        max_transaction_retries=1,
        transaction_retry_backoff_seconds=0.01,
    )
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver(
        session_obj=FakeSession(fail_once_on_query=CLEAR_STAGING_INDEX)
    )

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    stats = Neo4jGraphStore(config, index_path=Path("/tmp/kgqa-test-index")).write_graph(
        _fusion_result()
    )

    assert stats.schema_ready is True
    assert fake_driver.session_obj.failures_seen == 1
    assert CLEAR_STAGING_INDEX in [call[0] for call in fake_driver.session_obj.writes]


def test_neo4j_store_reports_non_retryable_write_failure_without_secret(monkeypatch):
    config = GraphStoreConfig(
        password_env="KGQA_TEST_NEO4J_PASSWORD",
        max_transaction_retries=1,
    )
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver(session_obj=FakeSession(fail_always_on_query=MERGE_ENTITY))

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    with pytest.raises(GraphStoreError) as exc_info:
        Neo4jGraphStore(config, index_path=Path("/tmp/kgqa-test-index")).write_graph(
            _fusion_result()
        )

    message = str(exc_info.value)
    assert "operation=merge_entities" in message
    assert "index_id=" in message
    assert "secret" not in message


def test_neo4j_store_uses_staging_index_before_promote(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver()

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    index_path = Path("/tmp/kgqa-test-index")
    stats = Neo4jGraphStore(config, index_path=index_path).write_graph(_fusion_result())

    canonical_id = compute_index_id(index_path)
    assert stats.write_strategy == "staged_replace"
    assert stats.staging_index_id is not None
    assert stats.staging_index_id.startswith(f"{canonical_id}__staging__")
    writes = fake_driver.session_obj.writes
    merge_index_params = next(params for query, params in writes if query == MERGE_INDEX)
    assert merge_index_params["index_id"] == stats.staging_index_id
    assert CLEAR_INDEX in [query for query, _ in writes]
    assert PROMOTE_STAGED_NODES in [query for query, _ in writes]
    assert PROMOTE_STAGED_RELATIONSHIPS in [query for query, _ in writes]
    assert any(query == PROMOTE_STAGED_INDEX for query, _ in writes)


def test_neo4j_store_promotes_staged_records_with_independent_queries(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver()

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    Neo4jGraphStore(config, index_path=Path("/tmp/kgqa-test-index")).write_graph(
        _fusion_result()
    )

    queries = [query for query, _ in fake_driver.session_obj.writes]
    promote_steps = [
        CLEAR_INDEX,
        PROMOTE_STAGED_NODES,
        PROMOTE_STAGED_RELATIONSHIPS,
        PROMOTE_STAGED_INDEX,
    ]
    positions = [queries.index(query) for query in promote_steps]
    assert positions == sorted(positions)


def test_neo4j_store_failure_during_staging_does_not_clear_canonical(monkeypatch):
    config = GraphStoreConfig(password_env="KGQA_TEST_NEO4J_PASSWORD")
    monkeypatch.setenv(config.password_env, "secret")
    fake_driver = FakeDriver(
        session_obj=FakeSession(fail_always_on_query=MERGE_ENTITY)
    )

    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda uri, auth, **kwargs: fake_driver,
    )

    with pytest.raises(GraphStoreError):
        Neo4jGraphStore(config, index_path=Path("/tmp/kgqa-test-index")).write_graph(
            _fusion_result()
        )

    queries = [query for query, _ in fake_driver.session_obj.writes]
    assert CLEAR_INDEX not in queries
    assert PROMOTE_STAGED_INDEX not in queries


class FakeDriver:
    def __init__(self, session_obj=None):
        self.session_obj = session_obj or FakeSession()
        self.closed = False

    def session(self, database: str):
        self.session_obj.database = database
        return self.session_obj

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(
        self,
        fail_once_on_query: str | None = None,
        fail_always_on_query: str | None = None,
        constraint_names: list[str] | None = None,
        index_names: list[str] | None = None,
    ):
        self.writes = []
        self.write_options = []
        self.read_options = []
        self.database = None
        self.fail_once_on_query = fail_once_on_query
        self.fail_always_on_query = fail_always_on_query
        self.constraint_names = constraint_names or EXPECTED_CONSTRAINT_NAMES
        self.index_names = index_names or EXPECTED_INDEX_NAMES
        self.failures_seen = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute_write(self, func, query, params):
        if query == self.fail_once_on_query and self.failures_seen == 0:
            self.failures_seen += 1
            raise TransientWriteError("temporary unavailable")
        if query == self.fail_always_on_query:
            raise RuntimeError("permanent failure")
        self.writes.append((query, params))
        self.write_options.append(
            {
                "metadata": getattr(func, "metadata", None),
                "timeout": getattr(func, "timeout", None),
            }
        )
        return func(
            FakeTx(
                constraint_names=self.constraint_names,
                index_names=self.index_names,
            ),
            query,
            params,
        )

    def execute_read(self, func, *args):
        self.read_options.append(
            {
                "metadata": getattr(func, "metadata", None),
                "timeout": getattr(func, "timeout", None),
            }
        )
        return func(
            FakeTx(
                constraint_names=self.constraint_names,
                index_names=self.index_names,
            ),
            *args,
        )


class FakeTx:
    def __init__(
        self,
        constraint_names: list[str] | None = None,
        index_names: list[str] | None = None,
    ):
        self.constraint_names = constraint_names or EXPECTED_CONSTRAINT_NAMES
        self.index_names = index_names or EXPECTED_INDEX_NAMES

    def run(self, query, **params):
        if "SHOW CONSTRAINTS" in query:
            return FakeResult(self.constraint_names)
        if "SHOW INDEXES" in query:
            return FakeResult(self.index_names)
        return FakeResult()


class FakeResult:
    def __init__(self, names: list[str] | None = None):
        self.names = names or []

    def __iter__(self):
        return iter({"name": name} for name in self.names)

    def consume(self):
        return None

    def single(self):
        return {"num_text_units": 0, "num_entities": 2, "num_relationships": 1}


class TransientWriteError(RuntimeError):
    code = "Neo.TransientError.Transaction.LockClientStopped"


def _find_batch_record(writes, record_id: str) -> dict:
    for _, params in writes:
        if isinstance(params, list):
            for item in params:
                if item.get("id") == record_id:
                    return item
        elif params.get("id") == record_id:
            return params
    raise AssertionError(f"Missing fake write record: {record_id}")


def _fusion_result() -> FusionResult:
    entity_a = FusedEntity(
        id="entity_a",
        name="GraphRAG",
        canonical_name="graphrag",
        type="Technology",
        description="GraphRAG",
        aliases=["GraphRAG"],
        evidence_chunk_ids=["chunk_1"],
        confidence=0.9,
        metadata={"source_candidate_ids": ["e1"]},
    )
    entity_b = FusedEntity(
        id="entity_b",
        name="Knowledge Graph",
        canonical_name="knowledge graph",
        type="Technology",
        description="Knowledge Graph",
        aliases=["Knowledge Graph"],
        evidence_chunk_ids=["chunk_1"],
        confidence=0.9,
        metadata={"source_candidate_ids": ["e2"]},
    )
    relationship = FusedRelationship(
        id="rel_1",
        source_entity_id="entity_a",
        target_entity_id="entity_b",
        source_name="GraphRAG",
        target_name="Knowledge Graph",
        relation="uses",
        original_relations=["uses"],
        description="GraphRAG uses Knowledge Graph",
        confidence=0.93,
        evidence_chunk_ids=["chunk_1"],
        extraction_count=1,
        metadata={"source_triple_ids": ["t1"]},
    )
    return FusionResult(
        entities=[entity_a, entity_b],
        relationships=[relationship],
        rejected_triples=[],
        graph={
            "nodes": [entity_a.__dict__, entity_b.__dict__],
            "edges": [relationship.__dict__],
            "statistics": {
                "num_nodes": 2,
                "num_edges": 1,
                "num_rejected_triples": 0,
            },
        },
    )
