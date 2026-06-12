"""Pipeline 工厂。

用于创建和注册工作流 Pipeline。
"""

import logging
from typing import ClassVar

from graphrag_v2.config.enums import IndexingMethod
from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.pipeline import Pipeline
from graphrag_v2.pipeline.workflow import WorkflowFunction

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Pipeline 工厂类。
    
    用于注册和创建工作流 Pipeline。
    
    Attributes:
        workflows: 已注册的工作流函数字典
        pipelines: 已注册的 Pipeline 定义字典
    """

    workflows: ClassVar[dict[str, WorkflowFunction]] = {}
    """已注册的工作流函数。"""

    pipelines: ClassVar[dict[str, list[str]]] = {}
    """已注册的 Pipeline 定义。"""

    @classmethod
    def register(cls, name: str, workflow: WorkflowFunction) -> None:
        """注册自定义工作流函数。
        
        Args:
            name: 工作流名称
            workflow: 工作流函数
        """
        cls.workflows[name] = workflow
        logger.debug(f"已注册工作流: {name}")

    @classmethod
    def register_all(cls, workflows: dict[str, WorkflowFunction]) -> None:
        """批量注册工作流函数。
        
        Args:
            workflows: 工作流函数字典
        """
        for name, workflow in workflows.items():
            cls.register(name, workflow)

    @classmethod
    def register_pipeline(cls, name: str, workflows: list[str]) -> None:
        """注册新的 Pipeline 方法。
        
        Args:
            name: Pipeline 名称
            workflows: 工作流名称列表
        """
        cls.pipelines[name] = workflows
        logger.debug(f"已注册 Pipeline: {name} with {len(workflows)} workflows")

    @classmethod
    def create_pipeline(
        cls,
        config: GraphRagConfig,
        method: IndexingMethod | str = IndexingMethod.Standard,
        workflows: list[WorkflowFunction] | list[str] | None = None,
    ) -> Pipeline:
        """创建 Pipeline。
        
        Args:
            config: GraphRAG 配置
            method: 索引方法
            
        Returns:
            Pipeline 实例
        """
        # 获取工作流列表
        workflow_items = workflows or config.workflows or cls.pipelines.get(method, [])
        
        logger.info(f"创建 Pipeline，方法: {method}, 工作流: {workflow_items}")
        
        # 创建工作流元组列表
        workflow_tuples = []
        for item in workflow_items:
            if isinstance(item, str):
                name = item
                workflow_fn = cls.workflows.get(name)
            else:
                workflow_fn = item
                name = getattr(item, "__name__", "workflow")

            if workflow_fn is None:
                logger.warning(f"工作流 {name} 未注册，跳过")
                continue
            workflow_tuples.append((name, workflow_fn))
        
        return Pipeline(workflow_tuples)


# --- 注册默认工作流 ---
from graphrag_v2.workflows import (
    create_base_text_units,
    create_communities,
    create_community_reports,
    extract_graph,
    generate_embeddings,
    load_documents,
)

PipelineFactory.register("load_documents", load_documents)
PipelineFactory.register("create_base_text_units", create_base_text_units)
PipelineFactory.register("extract_graph", extract_graph)
PipelineFactory.register("create_communities", create_communities)
PipelineFactory.register("create_community_reports", create_community_reports)
PipelineFactory.register("generate_embeddings", generate_embeddings)

# --- 注册默认 Pipeline ---
_standard_workflows = [
    "load_documents",
    "create_base_text_units",
    "extract_graph",
    "create_communities",
    "create_community_reports",
    "generate_embeddings",
]

PipelineFactory.register_pipeline(IndexingMethod.Standard, _standard_workflows)
