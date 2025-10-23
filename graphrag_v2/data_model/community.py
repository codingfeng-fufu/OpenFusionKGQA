"""社区数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.named import Named


@dataclass
class Community(Named):
    """社区数据模型。
    
    表示通过社区检测算法（如 Leiden）发现的实体社区。
    社区是层次化的，可以有父社区和子社区。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        title: 社区名称
        level: 社区层级（字符串形式，如 "0", "1", "2"）
        parent: 父社区的 ID
        children: 子社区的 ID 列表
        entity_ids: 社区中的实体 ID 列表（可选）
        relationship_ids: 社区中的关系 ID 列表（可选）
        text_unit_ids: 社区相关的文本单元 ID 列表（可选）
        covariate_ids: 社区相关的协变量 ID 字典（可选，如声明）
        attributes: 额外的属性字典（可选）
        size: 社区大小（文本单元数量，可选）
        period: 社区时间段（可选）
    """

    level: str | int = "0"
    """社区层级（字符串或整数形式，如 "0", "1", "2"，默认为 "0"）。"""

    parent: str | None = None
    """父社区的 ID（可选）。"""

    children: list[str] | None = None
    """子社区的 ID 列表（可选）。"""

    entity_ids: list[str] | None = None
    """社区中的实体 ID 列表（可选）。"""

    relationship_ids: list[str] | None = None
    """社区中的关系 ID 列表（可选）。"""

    text_unit_ids: list[str] | None = None
    """社区相关的文本单元 ID 列表（可选）。"""

    covariate_ids: dict[str, list[str]] | None = None
    """社区相关的协变量 ID 字典（可选）。
    
    键是协变量类型（如 "claims"），值是该类型的协变量 ID 列表。
    """

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。
    
    会被包含在搜索提示中。
    """

    size: int | None = None
    """社区大小（文本单元数量，可选）。"""

    period: str | None = None
    """社区时间段（可选）。"""

    @property
    def entities(self) -> list[str] | None:
        """实体列表（entity_ids的别名，用于向后兼容）。"""
        return self.entity_ids

    @entities.setter
    def entities(self, value: list[str] | None):
        """设置实体列表。"""
        self.entity_ids = value

    @property
    def relationships(self) -> list[str] | None:
        """关系列表（relationship_ids的别名，用于向后兼容）。"""
        return self.relationship_ids

    @relationships.setter
    def relationships(self, value: list[str] | None):
        """设置关系列表。"""
        self.relationship_ids = value

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        short_id_key: str = "human_readable_id",
        level_key: str = "level",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        text_units_key: str = "text_unit_ids",
        covariates_key: str = "covariate_ids",
        parent_key: str = "parent",
        children_key: str = "children",
        attributes_key: str = "attributes",
        size_key: str = "size",
        period_key: str = "period",
    ) -> "Community":
        """从字典创建社区对象。
        
        Args:
            d: 包含社区数据的字典
            id_key: ID 字段的键名
            title_key: 标题字段的键名
            short_id_key: 短 ID 字段的键名
            level_key: 层级字段的键名
            entities_key: 实体 ID 字段的键名
            relationships_key: 关系 ID 字段的键名
            text_units_key: 文本单元 ID 字段的键名
            covariates_key: 协变量 ID 字段的键名
            parent_key: 父社区字段的键名
            children_key: 子社区字段的键名
            attributes_key: 属性字段的键名
            size_key: 大小字段的键名
            period_key: 时间段字段的键名
            
        Returns:
            Community: 社区对象
        """
        return Community(
            id=d[id_key],
            title=d[title_key],
            level=d[level_key],
            parent=d[parent_key],
            children=d[children_key],
            short_id=d.get(short_id_key),
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            text_unit_ids=d.get(text_units_key),
            covariate_ids=d.get(covariates_key),
            attributes=d.get(attributes_key),
            size=d.get(size_key),
            period=d.get(period_key),
        )

