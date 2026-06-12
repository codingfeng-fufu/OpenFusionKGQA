"""Graph fusion configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class FusionConfig(BaseModel):
    """Configuration for graph fusion behavior."""

    relation_schema_mode: str = Field(
        default="open",
        description=(
            "Relation schema mode. 'open' keeps extracted predicates without "
            "schema scoring penalties; 'closed' aligns predicates to the built-in "
            "relation schema."
        ),
    )

    @model_validator(mode="after")
    def _validate_model(self):
        mode = self.relation_schema_mode.strip().lower()
        if mode not in {"open", "closed"}:
            raise ValueError("relation_schema_mode must be one of: open, closed")
        self.relation_schema_mode = mode
        return self
