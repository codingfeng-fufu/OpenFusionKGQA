"""Neo4j graph store."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from graphrag_v2.artifacts.index_id import resolve_index_id
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_fusion.models import FusionResult
from graphrag_v2.graph_store.base import GraphStoreError, GraphStoreStats
from graphrag_v2.graph_store.cypher import (
    CLEAR_INDEX,
    CLEAR_STAGING_INDEX,
    CONSTRAINTS,
    EXPECTED_CONSTRAINT_NAMES,
    EXPECTED_INDEX_NAMES,
    INDEXES,
    MERGE_INDEX,
    MERGE_ENTITY,
    MERGE_RELATIONSHIP,
    MERGE_TEXT_UNIT,
    NEO4J_SCHEMA_VERSION,
    PROMOTE_STAGED_INDEX,
    PROMOTE_STAGED_NODES,
    PROMOTE_STAGED_RELATIONSHIPS,
)
from graphrag_v2.graph_store.neo4j_utils import (
    create_neo4j_driver,
    verify_neo4j_connection,
)


class Neo4jGraphStore:
    """Neo4j graph store with idempotent MERGE writes."""

    provider = "neo4j"

    def __init__(self, config: GraphStoreConfig, index_path: str | Path | None = None):
        self.config = config
        self.index_path = Path(index_path) if index_path is not None else None

    def write_graph(self, fusion_result: FusionResult) -> GraphStoreStats:
        canonical_index_id = self._index_id(required=True)
        use_staging = (
            self.config.replace_index_on_write
            and self.config.staged_replace_on_write
        )
        write_index_id = (
            self._staging_index_id(canonical_index_id)
            if use_staging
            else canonical_index_id
        )
        write_strategy = (
            "staged_replace"
            if use_staging
            else "in_place_replace"
            if self.config.replace_index_on_write
            else "in_place_merge"
        )
        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as session:
                self.ensure_schema(session=session)
                if use_staging:
                    self._execute_write(
                        session,
                        _run_query,
                        CLEAR_STAGING_INDEX,
                        {"staging_index_id": write_index_id},
                        operation="clear_staging_index",
                        index_id=write_index_id,
                    )
                elif self.config.replace_index_on_write:
                    self._execute_write(
                        session,
                        _run_query,
                        CLEAR_INDEX,
                        {"index_id": canonical_index_id},
                        operation="clear_index",
                        index_id=canonical_index_id,
                    )
                self._execute_write(
                    session,
                    _run_query,
                    MERGE_INDEX,
                    {
                        "index_id": write_index_id,
                        "output_path": str(self.index_path.resolve()),
                        "database": self.config.database,
                    },
                    operation="merge_index",
                    index_id=write_index_id,
                )
                self._write_records(
                    session,
                    MERGE_TEXT_UNIT,
                    self._load_text_units(write_index_id),
                    operation="merge_text_units",
                    index_id=write_index_id,
                )
                self._write_records(
                    session,
                    MERGE_ENTITY,
                    [
                        _scope_record(_clean_record(asdict(entity)), write_index_id)
                        for entity in fusion_result.entities
                    ],
                    operation="merge_entities",
                    index_id=write_index_id,
                )
                self._write_records(
                    session,
                    MERGE_RELATIONSHIP,
                    [
                        _scope_record(
                            _clean_record(asdict(relationship)),
                            write_index_id,
                        )
                        for relationship in fusion_result.relationships
                    ],
                    operation="merge_relationships",
                    index_id=write_index_id,
                )
                if use_staging:
                    self._execute_write(
                        session,
                        _run_query,
                        CLEAR_INDEX,
                        {"index_id": canonical_index_id},
                        operation="clear_index",
                        index_id=canonical_index_id,
                    )
                    self._execute_write(
                        session,
                        _run_query,
                        PROMOTE_STAGED_NODES,
                        {
                            "canonical_index_id": canonical_index_id,
                            "staging_index_id": write_index_id,
                        },
                        operation="promote_staged_nodes",
                        index_id=canonical_index_id,
                    )
                    self._execute_write(
                        session,
                        _run_query,
                        PROMOTE_STAGED_RELATIONSHIPS,
                        {
                            "canonical_index_id": canonical_index_id,
                            "staging_index_id": write_index_id,
                        },
                        operation="promote_staged_relationships",
                        index_id=canonical_index_id,
                    )
                    self._execute_write(
                        session,
                        _run_query,
                        PROMOTE_STAGED_INDEX,
                        {
                            "canonical_index_id": canonical_index_id,
                            "staging_index_id": write_index_id,
                        },
                        operation="promote_staged_index",
                        index_id=canonical_index_id,
                    )
                counts = self._execute_read(
                    session,
                    _read_stats,
                    canonical_index_id,
                    operation="read_stats",
                    index_id=canonical_index_id,
                )
                schema = self._read_schema_status(
                    session,
                    index_id=canonical_index_id,
                )
        except GraphStoreError:
            if use_staging:
                self._clear_staging_best_effort(driver, write_index_id)
            raise
        except Exception as exc:
            if use_staging:
                self._clear_staging_best_effort(driver, write_index_id)
            raise self._operation_error(
                "write",
                exc,
                index_id=canonical_index_id,
            ) from exc
        finally:
            driver.close()

        return GraphStoreStats(
            provider=self.provider,
            index_id=canonical_index_id,
            database=self.config.database,
            num_text_units=counts["num_text_units"],
            num_entities=counts["num_entities"],
            num_relationships=counts["num_relationships"],
            num_rejected_triples=len(fusion_result.rejected_triples),
            graph_path=None,
            metadata_path=self._metadata_path(),
            schema_ready=schema["schema_ready"],
            schema_constraints=schema["schema_constraints"],
            schema_indexes=schema["schema_indexes"],
            schema_version=schema["schema_version"],
            expected_schema_constraints=schema["expected_schema_constraints"],
            expected_schema_indexes=schema["expected_schema_indexes"],
            missing_schema_constraints=schema["missing_schema_constraints"],
            missing_schema_indexes=schema["missing_schema_indexes"],
            health_status=schema["health_status"],
            write_strategy=write_strategy,
            staging_index_id=write_index_id if use_staging else None,
        )

    def get_stats(self) -> GraphStoreStats:
        index_id = self._index_id(required=False)
        if index_id is None:
            self.preflight()
            return GraphStoreStats(
                provider=self.provider,
                index_id=None,
                database=self.config.database,
                num_entities=0,
                num_relationships=0,
                num_rejected_triples=0,
                num_text_units=0,
                graph_path=None,
                metadata_path=None,
            )

        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as session:
                counts = self._execute_read(
                    session,
                    _read_stats,
                    index_id,
                    operation="read_stats",
                    index_id=index_id,
                )
                schema = self._read_schema_status(session, index_id=index_id)
        except GraphStoreError:
            raise
        except Exception as exc:
            raise self._operation_error("inspect", exc, index_id=index_id) from exc
        finally:
            driver.close()

        return GraphStoreStats(
            provider=self.provider,
            index_id=index_id,
            database=self.config.database,
            num_text_units=counts["num_text_units"],
            num_entities=counts["num_entities"],
            num_relationships=counts["num_relationships"],
            num_rejected_triples=0,
            graph_path=None,
            metadata_path=self._metadata_path(),
            schema_ready=schema["schema_ready"],
            schema_constraints=schema["schema_constraints"],
            schema_indexes=schema["schema_indexes"],
            schema_version=schema["schema_version"],
            expected_schema_constraints=schema["expected_schema_constraints"],
            expected_schema_indexes=schema["expected_schema_indexes"],
            missing_schema_constraints=schema["missing_schema_constraints"],
            missing_schema_indexes=schema["missing_schema_indexes"],
            health_status=schema["health_status"],
        )

    def preflight(self) -> None:
        verify_neo4j_connection(self.config)

    def _driver(self):
        return create_neo4j_driver(self.config)

    def ensure_schema(self, session=None) -> dict[str, Any]:
        """Ensure Neo4j constraints and indexes exist."""
        if session is not None:
            self._ensure_schema_in_session(session)
            return self._read_schema_status(session, index_id=None)

        driver = self._driver()
        try:
            with driver.session(database=self.config.database) as managed_session:
                self._ensure_schema_in_session(managed_session)
                return self._read_schema_status(managed_session, index_id=None)
        except GraphStoreError:
            raise
        except Exception as exc:
            raise self._operation_error("ensure_schema", exc, index_id=None) from exc
        finally:
            driver.close()

    def _ensure_schema_in_session(self, session) -> None:
        for query in CONSTRAINTS:
            self._execute_write(
                session,
                _run_query,
                query,
                {},
                operation="ensure_schema",
                index_id=None,
            )
        for query in INDEXES:
            self._execute_write(
                session,
                _run_query,
                query,
                {},
                operation="ensure_schema",
                index_id=None,
            )

    def _write_records(
        self,
        session,
        query: str,
        records: list[dict[str, Any]],
        operation: str,
        index_id: str,
    ) -> None:
        for batch in _chunks(records, self.config.batch_size):
            self._execute_write(
                session,
                _run_many,
                query,
                batch,
                operation=operation,
                index_id=index_id,
            )

    def _read_schema_status(self, session, index_id: str | None) -> dict[str, Any]:
        schema = self._execute_read(
            session,
            _read_schema_status,
            operation="read_schema",
            index_id=index_id,
        )
        constraints = schema["schema_constraints"]
        indexes = schema["schema_indexes"]
        missing_constraints = sorted(
            set(EXPECTED_CONSTRAINT_NAMES) - set(constraints)
        )
        missing_indexes = sorted(set(EXPECTED_INDEX_NAMES) - set(indexes))
        schema_ready = not missing_constraints and not missing_indexes
        return {
            "schema_version": NEO4J_SCHEMA_VERSION,
            "schema_constraints": constraints,
            "schema_indexes": indexes,
            "expected_schema_constraints": EXPECTED_CONSTRAINT_NAMES,
            "expected_schema_indexes": EXPECTED_INDEX_NAMES,
            "missing_schema_constraints": missing_constraints,
            "missing_schema_indexes": missing_indexes,
            "schema_ready": schema_ready,
            "health_status": "ready" if schema_ready else "degraded",
        }

    def _execute_write(
        self,
        session,
        func,
        *args,
        operation: str,
        index_id: str | None,
    ):
        transaction_func = _with_transaction_config(
            func,
            metadata=self._transaction_metadata(operation, index_id),
            timeout=self.config.transaction_timeout_seconds,
        )
        return self._execute_with_retry(
            lambda: session.execute_write(transaction_func, *args),
            operation=operation,
            index_id=index_id,
        )

    def _execute_read(
        self,
        session,
        func,
        *args,
        operation: str,
        index_id: str | None,
    ):
        transaction_func = _with_transaction_config(
            func,
            metadata=self._transaction_metadata(operation, index_id),
            timeout=self.config.transaction_timeout_seconds,
        )
        return self._execute_with_retry(
            lambda: session.execute_read(transaction_func, *args),
            operation=operation,
            index_id=index_id,
        )

    def _execute_with_retry(self, call, operation: str, index_id: str | None):
        attempts = self.config.max_transaction_retries + 1
        for attempt in range(attempts):
            try:
                return call()
            except Exception as exc:
                if attempt >= attempts - 1 or not _is_retryable_error(exc):
                    raise self._operation_error(operation, exc, index_id=index_id) from exc
                time.sleep(
                    self.config.transaction_retry_backoff_seconds * (attempt + 1)
                )
        raise GraphStoreError(
            f"Neo4j operation failed unexpectedly: operation={operation}"
        )

    def _transaction_metadata(
        self,
        operation: str,
        index_id: str | None,
    ) -> dict[str, Any]:
        metadata = {
            "app": "kgqa",
            "operation": operation,
            "database": self.config.database,
        }
        if index_id is not None:
            metadata["index_id"] = index_id
        return metadata

    def _operation_error(
        self,
        operation: str,
        exc: Exception,
        index_id: str | None,
    ) -> GraphStoreError:
        index_part = f", index_id={index_id}" if index_id else ""
        return GraphStoreError(
            "Neo4j graph store operation failed "
            f"(operation={operation}, uri={self.config.uri}, "
            f"database={self.config.database}{index_part}): {exc}"
        )

    def _load_text_units(self, index_id: str) -> list[dict[str, Any]]:
        if self.index_path is None:
            return []
        path = self.index_path / "text_units.parquet"
        if not path.exists():
            return []
        rows = pd.read_parquet(path).to_dict(orient="records")
        return [
            _scope_record(_clean_record({"id": row.pop("chunk_id"), **row}), index_id)
            for row in rows
        ]

    def _index_id(self, required: bool) -> str | None:
        if self.index_path is None:
            if required:
                raise GraphStoreError("Neo4j graph writes require an index path.")
            return None
        return resolve_index_id(self.index_path)

    def _metadata_path(self) -> str | None:
        if self.index_path is None:
            return None
        return str((self.index_path / "index_metadata.json").resolve())

    def _staging_index_id(self, canonical_index_id: str) -> str:
        return f"{canonical_index_id}__staging__{int(time.time() * 1000)}"

    def _clear_staging_best_effort(self, driver, staging_index_id: str) -> None:
        try:
            with driver.session(database=self.config.database) as session:
                self._execute_write(
                    session,
                    _run_query,
                    CLEAR_STAGING_INDEX,
                    {"staging_index_id": staging_index_id},
                    operation="clear_staging_index",
                    index_id=staging_index_id,
                )
        except Exception:
            pass


def _run_query(tx, query: str, params: dict[str, Any]):
    tx.run(query, **params).consume()


def _run_many(tx, query: str, records: list[dict[str, Any]]) -> None:
    for params in records:
        tx.run(query, **params).consume()


def _with_transaction_config(func, metadata: dict[str, Any], timeout: float):
    try:
        from neo4j import unit_of_work

        return unit_of_work(metadata=metadata, timeout=timeout)(func)
    except Exception:
        func.metadata = metadata
        func.timeout = timeout
        return func


def _read_stats(tx, index_id: str) -> dict[str, int]:
    record = tx.run(
        """
        RETURN
          count { MATCH (:TextUnit {index_id: $index_id}) } AS num_text_units,
          count { MATCH (:Entity {index_id: $index_id}) } AS num_entities,
          count { MATCH ()-[:RELATION {index_id: $index_id}]->() } AS num_relationships
        """,
        index_id=index_id,
    ).single()
    return {
        "num_text_units": int(record["num_text_units"]) if record else 0,
        "num_entities": int(record["num_entities"]) if record else 0,
        "num_relationships": int(record["num_relationships"]) if record else 0,
    }


def _read_schema_status(tx) -> dict[str, list[str]]:
    constraints = _read_schema_names(tx, "SHOW CONSTRAINTS YIELD name RETURN name")
    indexes = _read_schema_names(tx, "SHOW INDEXES YIELD name RETURN name")
    return {
        "schema_constraints": sorted(constraints),
        "schema_indexes": sorted(indexes),
    }


def _read_schema_names(tx, query: str) -> list[str]:
    result = tx.run(query)
    names: list[str] = []
    for record in result:
        name = record.get("name") if hasattr(record, "get") else record["name"]
        if name:
            names.append(str(name))
    return names


def _scope_record(record: dict[str, Any], index_id: str) -> dict[str, Any]:
    return {"index_id": index_id, **record}


def _chunks(records: list[dict[str, Any]], size: int):
    for start in range(0, len(records), size):
        yield records[start:start + size]


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
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_retryable_error(exc: Exception) -> bool:
    classification = getattr(exc, "classification", None)
    if str(classification).lower().endswith("transienterror"):
        return True
    code = str(getattr(exc, "code", "") or "").lower()
    if "transienterror" in code:
        return True
    name = exc.__class__.__name__.lower()
    return "transient" in name or "serviceunavailable" in name
