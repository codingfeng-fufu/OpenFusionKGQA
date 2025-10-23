"""GraphRAG v2 数据模型。

定义了 GraphRAG 系统中使用的所有核心数据结构。

核心数据类:
- Identified: 带有 ID 的基础类
- Named: 带有名称/标题的基础类
- Entity: 实体数据模型
- Relationship: 关系数据模型
- Community: 社区数据模型
- CommunityReport: 社区报告数据模型
- TextUnit: 文本单元数据模型
- Document: 文档数据模型
- Covariate: 协变量数据模型（如声明）
"""

from graphrag_v2.data_model.community import Community
from graphrag_v2.data_model.community_report import CommunityReport
from graphrag_v2.data_model.covariate import Covariate
from graphrag_v2.data_model.document import Document
from graphrag_v2.data_model.entity import Entity
from graphrag_v2.data_model.identified import Identified
from graphrag_v2.data_model.named import Named
from graphrag_v2.data_model.relationship import Relationship
from graphrag_v2.data_model.text_unit import TextUnit

__all__ = [
    "Identified",
    "Named",
    "Entity",
    "Relationship",
    "Community",
    "CommunityReport",
    "TextUnit",
    "Document",
    "Covariate",
]

