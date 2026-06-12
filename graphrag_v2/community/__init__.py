"""GraphRAG-style community aggregation."""

from graphrag_v2.community.detector import build_projection_graph, detect_communities
from graphrag_v2.community.models import (
    Community,
    CommunityPipelineResult,
    CommunityReport,
    GraphProjection,
)
from graphrag_v2.community.neo4j_writer import Neo4jCommunityStore
from graphrag_v2.community.pipeline import run_community_pipeline
from graphrag_v2.community.reporter import MockCommunityReporter

__all__ = [
    "Community",
    "CommunityPipelineResult",
    "CommunityReport",
    "GraphProjection",
    "MockCommunityReporter",
    "Neo4jCommunityStore",
    "build_projection_graph",
    "detect_communities",
    "run_community_pipeline",
]
