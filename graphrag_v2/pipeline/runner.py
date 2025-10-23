"""Pipeline 运行器。

执行 Pipeline 中的所有工作流。
"""

import logging
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import (
    PipelineCache,
    PipelineRunContext,
    PipelineRunStats,
    PipelineStorage,
    WorkflowCallbacks,
)
from graphrag_v2.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineRunResult:
    """Pipeline 运行结果。
    
    Attributes:
        workflow_name: 工作流名称
        result: 工作流结果
        errors: 错误列表
        runtime: 运行时间（秒）
    """

    workflow_name: str
    """工作流名称。"""

    result: Any | None
    """工作流结果。"""

    errors: list[str] | None = None
    """错误列表。"""

    runtime: float = 0.0
    """运行时间（秒）。"""


def create_run_context(
    input_storage: PipelineStorage,
    output_storage: PipelineStorage,
    cache: PipelineCache | None = None,
    callbacks: WorkflowCallbacks | None = None,
    state: dict[str, Any] | None = None,
    previous_storage: PipelineStorage | None = None,
) -> PipelineRunContext:
    """创建 Pipeline 运行上下文。
    
    Args:
        input_storage: 输入存储
        output_storage: 输出存储
        cache: 缓存（可选）
        callbacks: 回调（可选）
        state: 状态（可选）
        previous_storage: 上一次运行的存储（可选）
        
    Returns:
        Pipeline 运行上下文
    """
    return PipelineRunContext(
        stats=PipelineRunStats(),
        input_storage=input_storage,
        output_storage=output_storage,
        previous_storage=previous_storage,
        cache=cache or PipelineCache(),
        callbacks=callbacks or WorkflowCallbacks(),
        state=state or {},
    )


async def run_pipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> AsyncIterable[PipelineRunResult]:
    """运行 Pipeline。
    
    按顺序执行 Pipeline 中的所有工作流。
    
    Args:
        pipeline: Pipeline 实例
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Yields:
        每个工作流的运行结果
    """
    logger.info(f"开始运行 Pipeline，共 {len(pipeline)} 个工作流")
    
    pipeline_start_time = time.time()
    
    for workflow_name, workflow_fn in pipeline.run():
        logger.info(f"开始执行工作流: {workflow_name}")
        
        # 调用回调
        context.callbacks.on_workflow_start(workflow_name)
        
        workflow_start_time = time.time()
        errors = []
        result = None
        
        try:
            # 执行工作流
            output = await workflow_fn(config, context)
            result = output.result
            
            # 检查是否需要停止
            if output.stop:
                logger.warning(f"工作流 {workflow_name} 请求停止 Pipeline")
                yield PipelineRunResult(
                    workflow_name=workflow_name,
                    result=result,
                    errors=["工作流请求停止"],
                    runtime=time.time() - workflow_start_time,
                )
                break
            
        except Exception as e:
            logger.error(f"工作流 {workflow_name} 执行失败: {e}", exc_info=True)
            errors.append(str(e))
            context.callbacks.on_error(workflow_name, e)
        
        workflow_runtime = time.time() - workflow_start_time
        
        # 调用回调
        context.callbacks.on_workflow_end(workflow_name, result)
        
        logger.info(f"工作流 {workflow_name} 完成，耗时: {workflow_runtime:.2f}秒")
        
        # 返回结果
        yield PipelineRunResult(
            workflow_name=workflow_name,
            result=result,
            errors=errors if errors else None,
            runtime=workflow_runtime,
        )
    
    # 更新总运行时间
    context.stats.total_runtime = time.time() - pipeline_start_time
    
    logger.info(f"Pipeline 运行完成，总耗时: {context.stats.total_runtime:.2f}秒")
    logger.info(f"统计信息: {context.stats}")

