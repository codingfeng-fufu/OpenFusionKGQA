"""Graph store factory."""

from __future__ import annotations

from pathlib import Path

from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store.base import GraphStoreError
from graphrag_v2.graph_store.json_store import JsonGraphStore
from graphrag_v2.graph_store.neo4j_store import Neo4jGraphStore


def create_graph_store(
    provider: str,
    index_path: str | Path,
    config: GraphStoreConfig,
):
    normalized = provider.strip().lower()
    if normalized == "json":
        return JsonGraphStore(index_path)
    if normalized == "neo4j":
        return Neo4jGraphStore(config, index_path=index_path)
    raise GraphStoreError(f"Unsupported graph store provider: {provider}")
