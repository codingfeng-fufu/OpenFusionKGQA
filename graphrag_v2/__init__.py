"""OpenFusionKGQA internal Python package.

`graphrag_v2` 是历史包名。公开项目名是 OpenFusionKGQA，主线能力是
open-corpus knowledge graph fusion 和 graph-grounded QA。

主要模块:
- config: 配置管理
- data_model: 数据模型定义
- indexing: 索引构建
- qa: 图谱问答主路径
- prompts: Prompt 模板
"""

__version__ = "0.2.0-beta.1"
__author__ = "OpenFusionKGQA maintainers"

from graphrag_v2.config import GraphRagConfig
from graphrag_v2.config.loader import create_default_config, load_config

__all__ = [
    "GraphRagConfig",
    "create_default_config",
    "load_config",
]
