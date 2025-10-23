"""实体数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.named import Named


@dataclass
class Entity(Named):
    """实体数据模型。
    
    表示知识图谱中的一个实体（如人物、组织、地点等）。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        title: 实体名称
        type: 实体类型（如 person, organization, geo 等，可选）
        description: 实体描述（可选）
        description_embedding: 描述的向量嵌入（可选）
        name_embedding: 名称的向量嵌入（可选）
        community_ids: 实体所属的社区 ID 列表（可选）
        text_unit_ids: 实体出现的文本单元 ID 列表（可选）
        rank: 实体的重要性排名，数值越大越重要（可选，默认为 1）
        attributes: 额外的属性字典（可选）
    """

    type: str | None = None
    """实体类型（如 person, organization, geo 等，可选）。"""

    description: str | None = None
    """实体描述（可选）。"""

    description_embedding: list[float] | None = None
    """描述的向量嵌入（可选）。"""

    name_embedding: list[float] | None = None
    """名称的向量嵌入（可选）。"""

    community_ids: list[str] | None = None
    """实体所属的社区 ID 列表（可选）。"""

    text_unit_ids: list[str] | None = None
    """实体出现的文本单元 ID 列表（可选）。"""

    rank: int | None = 1
    """实体的重要性排名，数值越大越重要（可选，默认为 1）。
    
    可以基于中心性或其他图指标计算。
    """

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。

    可以包含开始时间、结束时间等信息，会被包含在搜索提示中。
    """

    @property
    def name(self) -> str:
        """实体名称（title的别名，用于向后兼容）。"""
        return self.title

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        title_key: str = "title",
        type_key: str = "type",
        description_key: str = "description",
        description_embedding_key: str = "description_embedding",
        name_embedding_key: str = "name_embedding",
        community_key: str = "community",
        text_unit_ids_key: str = "text_unit_ids",
        rank_key: str = "degree",
        attributes_key: str = "attributes",
    ) -> "Entity":
        """从字典创建实体对象。
        
        Args:
            d: 包含实体数据的字典
            id_key: ID 字段的键名
            short_id_key: 短 ID 字段的键名
            title_key: 标题字段的键名
            type_key: 类型字段的键名
            description_key: 描述字段的键名
            description_embedding_key: 描述嵌入字段的键名
            name_embedding_key: 名称嵌入字段的键名
            community_key: 社区字段的键名
            text_unit_ids_key: 文本单元 ID 字段的键名
            rank_key: 排名字段的键名
            attributes_key: 属性字段的键名
            
        Returns:
            Entity: 实体对象
        """
        return Entity(
            id=d[id_key],
            title=d[title_key],
            short_id=d.get(short_id_key),
            type=d.get(type_key),
            description=d.get(description_key),
            name_embedding=d.get(name_embedding_key),
            description_embedding=d.get(description_embedding_key),
            community_ids=d.get(community_key),
            rank=d.get(rank_key, 1),
            text_unit_ids=d.get(text_unit_ids_key),
            attributes=d.get(attributes_key),
        )

