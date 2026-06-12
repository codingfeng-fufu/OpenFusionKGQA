"""Base graph store contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graphrag_v2.graph_fusion.models import FusionResult


class GraphStoreError(RuntimeError):
    """Raised when a graph store operation fails."""


@dataclass(frozen=True)
class GraphStoreStats:
    provider: str
    num_entities: int
    num_relationships: int
    num_rejected_triples: int
    graph_path: str | None = None
    metadata_path: str | None = None
    index_id: str | None = None
    database: str | None = None
    num_text_units: int = 0
    schema_ready: bool | None = None
    schema_constraints: list[str] | None = None
    schema_indexes: list[str] | None = None
    schema_version: str | None = None
    expected_schema_constraints: list[str] | None = None
    expected_schema_indexes: list[str] | None = None
    missing_schema_constraints: list[str] | None = None
    missing_schema_indexes: list[str] | None = None
    health_status: str | None = None
    write_strategy: str | None = None
    staging_index_id: str | None = None


class GraphStore(Protocol):
    provider: str

    def write_graph(self, fusion_result: FusionResult) -> GraphStoreStats:
        """Persist a fused graph and return store statistics."""

    def get_stats(self) -> GraphStoreStats:
        """Return graph store statistics."""
