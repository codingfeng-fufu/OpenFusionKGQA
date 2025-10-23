"""文本单元数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.identified import Identified


@dataclass
class TextUnit(Identified):
    """文本单元数据模型。
    
    表示文档分块后的一个文本单元。
    文本单元是 GraphRAG 的基本处理单位，每个文本单元会被提取实体和关系。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        text: 文本内容
        entity_ids: 文本单元中的实体 ID 列表（可选）
        relationship_ids: 文本单元中的关系 ID 列表（可选）
        covariate_ids: 文本单元相关的协变量 ID 字典（可选）
        n_tokens: 文本的 token 数量（可选）
        document_ids: 文本单元所属的文档 ID 列表（可选）
        attributes: 额外的属性字典（可选）
    """

    text: str = ""
    """文本内容（默认为空字符串）。"""

    entity_ids: list[str] | None = None
    """文本单元中的实体 ID 列表（可选）。"""

    relationship_ids: list[str] | None = None
    """文本单元中的关系 ID 列表（可选）。"""

    covariate_ids: dict[str, list[str]] | None = None
    """文本单元相关的协变量 ID 字典（可选）。
    
    键是协变量类型（如 "claims"），值是该类型的协变量 ID 列表。
    """

    n_tokens: int | None = None
    """文本的 token 数量（可选）。"""

    document_ids: list[str] | None = None
    """文本单元所属的文档 ID 列表（可选）。"""

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        text_key: str = "text",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        covariates_key: str = "covariate_ids",
        n_tokens_key: str = "n_tokens",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "TextUnit":
        """从字典创建文本单元对象。
        
        Args:
            d: 包含文本单元数据的字典
            id_key: ID 字段的键名
            short_id_key: 短 ID 字段的键名
            text_key: 文本字段的键名
            entities_key: 实体 ID 字段的键名
            relationships_key: 关系 ID 字段的键名
            covariates_key: 协变量 ID 字段的键名
            n_tokens_key: token 数量字段的键名
            document_ids_key: 文档 ID 字段的键名
            attributes_key: 属性字段的键名
            
        Returns:
            TextUnit: 文本单元对象
        """
        return TextUnit(
            id=d[id_key],
            short_id=d.get(short_id_key),
            text=d[text_key],
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            covariate_ids=d.get(covariates_key),
            n_tokens=d.get(n_tokens_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )

