"""Neo4j projection and community write-back."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graphrag_v2.artifacts.index_id import resolve_index_id
from graphrag_v2.community.models import Community, CommunityReport, GraphProjection
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store import GraphStoreError
from graphrag_v2.graph_store.neo4j_utils import create_neo4j_driver


COMMUNITY_CONSTRAINTS = [
    (
        "CREATE CONSTRAINT kgqa_community_scoped_id_unique IF NOT EXISTS "
        "FOR (c:Community) REQUIRE (c.index_id, c.id) IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT kgqa_community_report_scoped_id_unique IF NOT EXISTS "
        "FOR (r:CommunityReport) REQUIRE (r.index_id, r.id) IS UNIQUE"
    ),
]

READ_ENTITIES = """
MATCH (e:Entity {index_id: $index_id})
RETURN e {
  .id,
  .name,
  .canonical_name,
  .type,
  .description,
  .evidence_chunk_ids
} AS entity
ORDER BY e.id
"""

READ_RELATIONSHIPS = """
MATCH (source:Entity {index_id: $index_id})-[r:RELATION {index_id: $index_id}]->(target:Entity {index_id: $index_id})
RETURN {
  id: r.id,
  source_entity_id: source.id,
  target_entity_id: target.id,
  source_name: source.name,
  target_name: target.name,
  relation: r.relation,
  description: r.description,
  confidence: r.confidence,
  extraction_count: r.extraction_count,
  evidence_chunk_ids: r.evidence_chunk_ids,
  original_relations: r.original_relations
} AS relationship
ORDER BY r.id
"""

MERGE_COMMUNITY = """
MERGE (c:Community {index_id: $index_id, id: $id})
SET c.index_id = $index_id,
    c.level = $level,
    c.title = $title,
    c.summary = $summary,
    c.rank = $rank,
    c.size = $size,
    c.entity_ids = $entity_ids,
    c.relationship_ids = $relationship_ids,
    c.text_unit_ids = $text_unit_ids,
    c.metadata = $metadata,
    c.created_at = coalesce(c.created_at, datetime())
WITH c
UNWIND $entity_ids AS entity_id
MATCH (e:Entity {index_id: $index_id, id: entity_id})
MERGE (c)-[:CONTAINS_ENTITY]->(e)
"""

MERGE_COMMUNITY_REPORT = """
MERGE (r:CommunityReport {index_id: $index_id, id: $id})
SET r.index_id = $index_id,
    r.community_id = $community_id,
    r.title = $title,
    r.summary = $summary,
    r.full_content = $full_content,
    r.findings = $findings,
    r.key_entities = $key_entities,
    r.key_relationships = $key_relationships,
    r.evidence_chunk_ids = $evidence_chunk_ids,
    r.rank = $rank,
    r.metadata = $metadata
WITH r
MATCH (c:Community {index_id: $index_id, id: $community_id})
MERGE (c)-[:HAS_REPORT]->(r)
WITH r
UNWIND $evidence_chunk_ids AS chunk_id
MATCH (t:TextUnit {index_id: $index_id, id: chunk_id})
MERGE (r)-[:SUPPORTED_BY]->(t)
"""

COMMUNITY_STATS = """
RETURN
  count { MATCH (:Community {index_id: $index_id}) } AS num_communities,
  count { MATCH (:CommunityReport {index_id: $index_id}) } AS num_community_reports
"""

CLEAR_COMMUNITY_REPORTS = """
MATCH (r:CommunityReport {index_id: $index_id})
DETACH DELETE r
"""

CLEAR_COMMUNITIES = """
MATCH (c:Community {index_id: $index_id})
DETACH DELETE c
"""


class Neo4jCommunityStore:
    """Read graph projections from Neo4j and write community artifacts back."""

    provider = "neo4j"

    def __init__(self, config: GraphStoreConfig, index_path: str | Path | None = None):
        self.config = config
        self.index_path = Path(index_path) if index_path is not None else None

    def read_projection(self) -> GraphProjection:
        index_id = self._index_id()
        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as session:
                entities = session.execute_read(_read_entities, index_id)
                relationships = session.execute_read(_read_relationships, index_id)
        except Exception as exc:
            raise GraphStoreError(f"Neo4j community projection failed: {exc}") from exc
        finally:
            driver.close()
        return GraphProjection(entities=entities, relationships=relationships)

    def write(
        self,
        communities: list[Community],
        reports: list[CommunityReport],
    ) -> dict[str, int]:
        index_id = self._index_id()
        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as session:
                for query in COMMUNITY_CONSTRAINTS:
                    session.execute_write(_run_query, query, {})
                session.execute_write(
                    _run_query,
                    CLEAR_COMMUNITY_REPORTS,
                    {"index_id": index_id},
                )
                session.execute_write(
                    _run_query,
                    CLEAR_COMMUNITIES,
                    {"index_id": index_id},
                )
                for community in communities:
                    session.execute_write(
                        _run_query,
                        MERGE_COMMUNITY,
                        _scope_record(_clean_record(asdict(community)), index_id),
                    )
                for report in reports:
                    session.execute_write(
                        _run_query,
                        MERGE_COMMUNITY_REPORT,
                        _scope_record(_clean_record(asdict(report)), index_id),
                    )
                return session.execute_read(_read_stats, index_id)
        except Exception as exc:
            raise GraphStoreError(f"Neo4j community write failed: {exc}") from exc
        finally:
            driver.close()

    def get_stats(self) -> dict[str, int]:
        index_id = self._index_id()
        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as session:
                return session.execute_read(_read_stats, index_id)
        except Exception as exc:
            raise GraphStoreError(f"Neo4j community inspection failed: {exc}") from exc
        finally:
            driver.close()

    def _driver(self):
        return create_neo4j_driver(self.config)

    def _index_id(self) -> str:
        if self.index_path is None:
            raise GraphStoreError("Neo4j community operations require an index path.")
        return resolve_index_id(self.index_path)


def _read_entities(tx, index_id: str) -> list[dict[str, Any]]:
    return [
        dict(record["entity"])
        for record in tx.run(READ_ENTITIES, index_id=index_id)
    ]


def _read_relationships(tx, index_id: str) -> list[dict[str, Any]]:
    return [
        dict(record["relationship"])
        for record in tx.run(READ_RELATIONSHIPS, index_id=index_id)
    ]


def _read_stats(tx, index_id: str) -> dict[str, int]:
    record = tx.run(COMMUNITY_STATS, index_id=index_id).single()
    return {
        "num_communities": int(record["num_communities"]) if record else 0,
        "num_community_reports": int(record["num_community_reports"]) if record else 0,
    }


def _run_query(tx, query: str, params: dict[str, Any]):
    tx.run(query, **params).consume()


def _scope_record(record: dict[str, Any], index_id: str) -> dict[str, Any]:
    return {"index_id": index_id, **record}


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if key == "metadata":
            cleaned[key] = json.dumps(value or {}, ensure_ascii=False, sort_keys=True)
        else:
            cleaned[key] = _clean_value(value)
    return cleaned


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
