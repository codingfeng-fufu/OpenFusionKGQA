"""关系数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.identified import Identified


@dataclass
class Relationship(Identified):
    """关系数据模型。
    
    表示知识图谱中两个实体之间的关系。
    这是一个通用的关系模型，可以表示任意两个实体之间的任意类型的关系。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        source: 源实体名称
        target: 目标实体名称
        weight: 边的权重（可选，默认为 1.0）
        description: 关系描述（可选）
        description_embedding: 描述的向量嵌入（可选）
        text_unit_ids: 关系出现的文本单元 ID 列表（可选）
        rank: 关系的重要性排名，数值越大越重要（可选，默认为 1）
        attributes: 额外的属性字典（可选）
    """

    source: str = ""
    """源实体名称（默认为空字符串）。"""

    target: str = ""
    """目标实体名称（默认为空字符串）。"""

    weight: float | None = 1.0
    """边的权重（可选，默认为 1.0）。"""

    description: str | None = None
    """关系描述（可选）。"""

    description_embedding: list[float] | None = None
    """描述的向量嵌入（可选）。"""

    text_unit_ids: list[str] | None = None
    """关系出现的文本单元 ID 列表（可选）。"""

    rank: int | None = 1
    """关系的重要性排名，数值越大越重要（可选，默认为 1）。
    
    可以基于中心性或其他图指标计算。
    """

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。
    
    会被包含在搜索提示中。
    """

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        rank_key: str = "rank",
        weight_key: str = "weight",
        text_unit_ids_key: str = "text_unit_ids",
        attributes_key: str = "attributes",
    ) -> "Relationship":
        """从字典创建关系对象。
        
        Args:
            d: 包含关系数据的字典
            id_key: ID 字段的键名
            short_id_key: 短 ID 字段的键名
            source_key: 源实体字段的键名
            target_key: 目标实体字段的键名
            description_key: 描述字段的键名
            rank_key: 排名字段的键名
            weight_key: 权重字段的键名
            text_unit_ids_key: 文本单元 ID 字段的键名
            attributes_key: 属性字段的键名
            
        Returns:
            Relationship: 关系对象
        """
        return Relationship(
            id=d[id_key],
            short_id=d.get(short_id_key),
            source=d[source_key],
            target=d[target_key],
            rank=d.get(rank_key, 1),
            description=d.get(description_key),
            weight=d.get(weight_key, 1.0),
            text_unit_ids=d.get(text_unit_ids_key),
            attributes=d.get(attributes_key),
        )

