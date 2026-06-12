"""In-memory storage compatibility adapter."""

from __future__ import annotations

from typing import Any

from graphrag_v2.pipeline.context import PipelineStorage


class _CompletedAwaitable:
    def __await__(self):
        if False:
            yield None
        return None


class MemoryPipelineStorage(PipelineStorage):
    """PipelineStorage with legacy sync-style set support."""

    def __init__(self):
        super().__init__(base_dir="memory")

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        return _CompletedAwaitable()
