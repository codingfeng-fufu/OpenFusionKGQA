"""Community detection and report configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CommunityConfig(BaseModel):
    """Configuration for GraphRAG-style community aggregation."""

    enabled: bool = Field(default=False, description="Enable community pipeline.")
    algorithm: str = Field(default="louvain", description="Community algorithm.")
    max_level: int = Field(default=1, description="Maximum community hierarchy level.")
    min_community_size: int = Field(default=2, description="Minimum community size.")
    generate_reports: bool = Field(default=True, description="Generate reports.")
    reporter: str = Field(default="mock", description="Community reporter provider.")
