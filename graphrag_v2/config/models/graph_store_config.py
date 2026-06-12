"""Graph store configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GraphStoreConfig(BaseModel):
    """Configuration for graph store backends."""

    provider: str = Field(default="json", description="Graph store provider.")
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI.")
    username: str = Field(default="neo4j", description="Neo4j username.")
    password_env: str = Field(
        default="NEO4J_PASSWORD",
        description="Environment variable containing the Neo4j password.",
    )
    database: str = Field(default="neo4j", description="Neo4j database name.")
    fallback: str = Field(default="json", description="Fallback graph store provider.")
    batch_size: int = Field(
        default=500,
        ge=1,
        description="Number of records to write per Neo4j transaction.",
    )
    replace_index_on_write: bool = Field(
        default=True,
        description="Replace the current scoped Neo4j index before writing.",
    )
    staged_replace_on_write: bool = Field(
        default=True,
        description=(
            "Write Neo4j replacements to a staging index and promote only "
            "after all records are written successfully."
        ),
    )
    connection_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Neo4j driver connection timeout in seconds.",
    )
    transaction_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Neo4j transaction timeout in seconds.",
    )
    max_transaction_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum retries for Neo4j transient transaction failures.",
    )
    transaction_retry_backoff_seconds: float = Field(
        default=0.5,
        gt=0,
        description="Initial backoff before retrying a Neo4j transaction.",
    )
