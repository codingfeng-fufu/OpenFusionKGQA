"""QA data sources backed by local artifacts or Neo4j."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from graphrag_v2.community.neo4j_writer import Neo4jCommunityStore
from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.graph_store import GraphStoreError


class QADataSource(Protocol):
    provider: str

    def metadata(self) -> dict[str, Any]:
        """Return index metadata."""

    def entities(self) -> list[dict[str, Any]]:
        """Return entity records."""

    def relationships(self) -> list[dict[str, Any]]:
        """Return relationship records."""

    def communities(self) -> list[dict[str, Any]]:
        """Return community records."""

    def community_reports(self) -> list[dict[str, Any]]:
        """Return community report records."""

    def text_units(self) -> list[dict[str, Any]]:
        """Return text unit records."""


class LocalArtifactQADataSource:
    """Read QA inputs from local index artifacts."""

    provider = "json"

    def __init__(
        self,
        index_path: str | Path,
        fallback_from_provider: str | None = None,
        fallback_reason: str | None = None,
    ):
        self.index_path = Path(index_path)
        self._metadata = self._read_json(self.index_path / "index_metadata.json")
        self._fallback_from_provider = fallback_from_provider
        self._fallback_reason = fallback_reason

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        if self._fallback_from_provider:
            metadata["qa_fallback_from_provider"] = self._fallback_from_provider
        if self._fallback_reason:
            metadata["qa_fallback_reason"] = self._fallback_reason
        return metadata

    def entities(self) -> list[dict[str, Any]]:
        if (self.index_path / "entities.parquet").exists():
            return self._read_parquet(self.index_path / "entities.parquet")
        return self._graph_nodes()

    def relationships(self) -> list[dict[str, Any]]:
        if (self.index_path / "relationships.parquet").exists():
            return self._read_parquet(self.index_path / "relationships.parquet")
        return self._graph_edges()

    def communities(self) -> list[dict[str, Any]]:
        path = self.index_path / "communities.parquet"
        return self._read_parquet(path) if path.exists() else []

    def community_reports(self) -> list[dict[str, Any]]:
        path = self.index_path / "community_reports.parquet"
        return self._read_parquet(path) if path.exists() else []

    def text_units(self) -> list[dict[str, Any]]:
        path = self.index_path / "text_units.parquet"
        return self._read_parquet(path) if path.exists() else []

    def _graph(self) -> dict[str, Any]:
        path = self.index_path / "graph.json"
        if not path.exists():
            return {"nodes": [], "edges": []}
        return self._read_json(path)

    def _graph_nodes(self) -> list[dict[str, Any]]:
        return [_normalize_record(node) for node in self._graph().get("nodes", [])]

    def _graph_edges(self) -> list[dict[str, Any]]:
        return [_normalize_record(edge) for edge in self._graph().get("edges", [])]

    def _read_parquet(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            _normalize_record(record)
            for record in pd.read_parquet(path).to_dict(orient="records")
        ]

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


class Neo4jArtifactQADataSource(LocalArtifactQADataSource):
    """Read graph projection from Neo4j and other artifacts locally."""

    provider = "neo4j"

    def __init__(
        self,
        index_path: str | Path,
        config: GraphStoreConfig | None = None,
    ):
        super().__init__(index_path)
        self.config = config or GraphStoreConfig()
        self._community_store = Neo4jCommunityStore(self.config, index_path=self.index_path)
        self._projection = None

    def _read_projection(self):
        if self._projection is None:
            self._projection = self._community_store.read_projection()
        return self._projection

    def entities(self) -> list[dict[str, Any]]:
        return [_normalize_record(entity) for entity in self._read_projection().entities]

    def relationships(self) -> list[dict[str, Any]]:
        return [
            _normalize_record(relationship)
            for relationship in self._read_projection().relationships
        ]


def load_qa_data_source(
    index_path: str | Path,
    config: GraphStoreConfig | None = None,
    prefer_neo4j: bool = True,
    allow_neo4j_fallback: bool = True,
) -> QADataSource:
    """Load the best available QA data source for an index."""
    local_source = LocalArtifactQADataSource(index_path)
    metadata = local_source.metadata()
    provider = str(metadata.get("graph_store_provider") or "json").strip().lower()
    if prefer_neo4j and provider == "neo4j":
        try:
            neo4j_source = Neo4jArtifactQADataSource(index_path, config=config)
            neo4j_source.entities()
            neo4j_source.relationships()
            return neo4j_source
        except GraphStoreError as exc:
            if not allow_neo4j_fallback:
                raise
            return LocalArtifactQADataSource(
                index_path,
                fallback_from_provider="neo4j",
                fallback_reason=str(exc),
            )
    return local_source


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in record.items()}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{") or stripped.startswith("("):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return value
            return _normalize_value(parsed)
        return value
    if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple, bytes)):
        try:
            return _normalize_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    return value
