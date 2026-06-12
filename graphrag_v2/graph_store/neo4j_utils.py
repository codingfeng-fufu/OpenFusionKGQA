"""Shared Neo4j driver helpers."""

from __future__ import annotations

import os

from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store.base import GraphStoreError


def create_neo4j_driver(config: GraphStoreConfig):
    """Create a Neo4j driver from graph store config."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise GraphStoreError(
            "Neo4j driver is not installed. Install the optional 'neo4j' "
            "package before using --graph-store neo4j."
        ) from exc

    password = os.getenv(config.password_env)
    if not password:
        raise GraphStoreError(
            f"Neo4j password environment variable is not set: {config.password_env}"
        )

    return GraphDatabase.driver(
        config.uri,
        auth=(config.username, password),
        connection_timeout=config.connection_timeout_seconds,
    )


def verify_neo4j_connection(config: GraphStoreConfig) -> None:
    """Verify that Neo4j is reachable and the configured database can be queried."""
    driver = create_neo4j_driver(config)
    try:
        with driver.session(database=config.database) as session:
            session.run("RETURN 1 AS ok").single()
    except Exception as exc:
        raise GraphStoreError(f"Neo4j connection check failed: {exc}") from exc
    finally:
        driver.close()
