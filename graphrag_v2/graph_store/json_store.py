"""Local JSON graph store fallback."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from graphrag_v2.artifacts.index_id import resolve_index_id
from graphrag_v2.graph_fusion.models import FusionResult
from graphrag_v2.graph_store.base import GraphStoreError, GraphStoreStats


class JsonGraphStore:
    """Graph store backed by the local artifact directory."""

    provider = "json"

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.graph_path = self.index_path / "graph.json"
        self.metadata_path = self.index_path / "index_metadata.json"

    def write_graph(self, fusion_result: FusionResult) -> GraphStoreStats:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(
            json.dumps(fusion_result.graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return GraphStoreStats(
            provider=self.provider,
            index_id=resolve_index_id(self.index_path),
            num_text_units=self._count("text_units.parquet", 0),
            num_entities=len(fusion_result.entities),
            num_relationships=len(fusion_result.relationships),
            num_rejected_triples=len(fusion_result.rejected_triples),
            graph_path=str(self.graph_path.resolve()),
            metadata_path=str(self.metadata_path.resolve()),
            health_status="ready",
        )

    def get_stats(self) -> GraphStoreStats:
        graph = self._read_graph()
        statistics = graph.get("statistics", {})
        return GraphStoreStats(
            provider=self._metadata().get("graph_store_provider", self.provider),
            index_id=self._metadata().get("index_id"),
            num_text_units=self._count("text_units.parquet", 0),
            num_entities=self._count("entities.parquet", statistics.get("num_nodes", 0)),
            num_relationships=self._count(
                "relationships.parquet",
                statistics.get("num_edges", 0),
            ),
            num_rejected_triples=self._count(
                "rejected_triples.parquet",
                statistics.get("num_rejected_triples", 0),
            ),
            graph_path=str(self.graph_path.resolve()),
            metadata_path=str(self.metadata_path.resolve()),
            health_status="ready",
        )

    def find_entity(self, name: str) -> dict | None:
        normalized = " ".join(name.strip().lower().split())
        for node in self._read_graph().get("nodes", []):
            if node.get("canonical_name") == normalized:
                return node
        return None

    def get_neighbors(self, entity_id: str, hops: int = 1) -> list[dict]:
        if hops < 1:
            return []
        graph = self._read_graph()
        frontier = {entity_id}
        seen = {entity_id}
        edges = graph.get("edges", [])
        neighbors: list[dict] = []
        for _ in range(hops):
            next_frontier = set()
            for edge in edges:
                source = edge.get("source_entity_id")
                target = edge.get("target_entity_id")
                if source in frontier and target not in seen:
                    neighbors.append(edge)
                    next_frontier.add(target)
                elif target in frontier and source not in seen:
                    neighbors.append(edge)
                    next_frontier.add(source)
            seen.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return neighbors

    def _read_graph(self) -> dict:
        if not self.graph_path.exists():
            raise GraphStoreError(f"Missing graph artifact: {self.graph_path}")
        return json.loads(self.graph_path.read_text(encoding="utf-8"))

    def _metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _count(self, filename: str, fallback: int) -> int:
        path = self.index_path / filename
        if not path.exists():
            return int(fallback)
        return len(pd.read_parquet(path))


def stats_to_dict(stats: GraphStoreStats) -> dict:
    return asdict(stats)
