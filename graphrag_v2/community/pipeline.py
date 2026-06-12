"""Community pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from graphrag_v2.community.detector import detect_communities
from graphrag_v2.community.models import CommunityPipelineResult
from graphrag_v2.community.neo4j_writer import Neo4jCommunityStore
from graphrag_v2.community.reporter import MockCommunityReporter
from graphrag_v2.config import GraphRagConfig
from graphrag_v2.graph_store import GraphStoreError


def run_community_pipeline(
    output_path: str | Path,
    config: GraphRagConfig,
) -> CommunityPipelineResult:
    """Run Neo4j-backed community detection and report generation."""
    if config.community.algorithm != "louvain":
        raise ValueError(
            f"Unsupported community algorithm: {config.community.algorithm}"
        )
    if config.community.reporter != "mock":
        raise ValueError(f"Unsupported community reporter: {config.community.reporter}")

    store = Neo4jCommunityStore(config.graph_store, index_path=output_path)
    projection = store.read_projection()
    if not projection.entities:
        raise GraphStoreError("Neo4j graph has no Entity nodes to cluster.")

    communities = detect_communities(
        projection,
        min_community_size=config.community.min_community_size,
        max_level=config.community.max_level,
    )
    reports = []
    if config.community.generate_reports:
        communities, reports = MockCommunityReporter().generate(
            communities=communities,
            projection=projection,
        )

    store.write(communities, reports)
    return CommunityPipelineResult(communities=communities, reports=reports)
