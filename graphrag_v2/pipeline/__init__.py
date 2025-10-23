"""Pipeline 模块。

提供索引 Pipeline 的核心功能。
"""

from graphrag_v2.pipeline.context import (
    PipelineCache,
    PipelineRunContext,
    PipelineRunStats,
    PipelineStorage,
    WorkflowCallbacks,
)
from graphrag_v2.pipeline.factory import PipelineFactory
from graphrag_v2.pipeline.pipeline import Pipeline
from graphrag_v2.pipeline.runner import PipelineRunResult, create_run_context, run_pipeline
from graphrag_v2.pipeline.workflow import Workflow, WorkflowFunction, WorkflowFunctionOutput

# 便捷函数
create_pipeline = PipelineFactory.create_pipeline

# 别名以保持向后兼容
PipelineRunner = run_pipeline

__all__ = [
    "Pipeline",
    "Workflow",
    "WorkflowFunction",
    "WorkflowFunctionOutput",
    "PipelineFactory",
    "PipelineRunner",
    "PipelineRunResult",
    "create_pipeline",
    "create_run_context",
    "run_pipeline",
    "PipelineCache",
    "PipelineRunContext",
    "PipelineRunStats",
    "PipelineStorage",
    "WorkflowCallbacks",
]

