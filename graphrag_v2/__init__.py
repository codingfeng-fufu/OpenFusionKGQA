"""GraphRAG v2 - 基于微软 GraphRAG 学习的知识图谱增强检索系统。

这是一个从零开始构建的 GraphRAG 实现，参考微软开源的 GraphRAG 项目，
学习并应用最佳实践。

主要模块:
- config: 配置管理
- data_model: 数据模型定义
- index: 索引构建
- query: 查询引擎
- prompts: Prompt 模板
"""

__version__ = "0.1.0"
__author__ = "GraphRAG v2 Team"

from graphrag_v2.config import GraphRagConfig
from graphrag_v2.config.loader import create_default_config, load_config

__all__ = [
    "GraphRagConfig",
    "create_default_config",
    "load_config",
]

