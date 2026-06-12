"""Pipeline 工作流类型定义。

定义工作流函数的类型和输出格式。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext


@dataclass
class WorkflowFunctionOutput:
    """工作流函数的输出数据容器。
    
    Attributes:
        result: 工作流函数的结果。可以是任何类型，主要用于日志记录。
                每个工作流函数应该将正式输出写入提供的存储中。
        stop: 标志，指示工作流是否应该在此函数之后停止。
              仅在继续执行可能导致不稳定失败时使用。
    """

    result: Any | None
    """工作流函数的结果。"""

    stop: bool = False
    """是否停止工作流。"""

    @property
    def outputs(self) -> dict[str, Any]:
        """Legacy output mapping used by older tests."""
        if self.result is None:
            return {}
        name = getattr(self.result, "attrs", {}).get("name")
        if name:
            return {name: self.result}
        return {
            "documents": self.result,
            "text_units": self.result,
            "entities": self.result,
            "relationships": self.result,
            "communities": self.result,
            "community_reports": self.result,
        }


WorkflowFunction = Callable[
    [GraphRagConfig, PipelineRunContext],
    Awaitable[WorkflowFunctionOutput],
]
"""工作流函数类型。

接受配置和运行上下文，返回工作流输出的异步函数。
"""

Workflow = tuple[str, WorkflowFunction]
"""工作流类型。

一个元组，包含工作流名称和工作流函数。
"""
