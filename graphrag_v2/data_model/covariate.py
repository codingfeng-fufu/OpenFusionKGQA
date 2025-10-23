"""协变量数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.identified import Identified


@dataclass
class Covariate(Identified):
    """协变量数据模型。
    
    协变量是与主体（如实体）相关联的元数据，例如实体声明（claims）。
    每个主体可以关联多种类型的协变量。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        subject_id: 主体 ID（如实体 ID）
        subject_type: 主体类型（默认为 "entity"）
        covariate_type: 协变量类型（默认为 "claim"）
        text_unit_ids: 协变量信息出现的文本单元 ID 列表（可选）
        attributes: 额外的属性字典（可选）
    """

    subject_id: str = ""
    """主体 ID（如实体 ID，默认为空字符串）。"""

    subject_type: str = "entity"
    """主体类型（默认为 "entity"）。"""

    covariate_type: str = "claim"
    """协变量类型（默认为 "claim"）。"""

    text_unit_ids: list[str] | None = None
    """协变量信息出现的文本单元 ID 列表（可选）。"""

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        subject_id_key: str = "subject_id",
        covariate_type_key: str = "covariate_type",
        short_id_key: str = "human_readable_id",
        text_unit_ids_key: str = "text_unit_ids",
        attributes_key: str = "attributes",
    ) -> "Covariate":
        """从字典创建协变量对象。
        
        Args:
            d: 包含协变量数据的字典
            id_key: ID 字段的键名
            subject_id_key: 主体 ID 字段的键名
            covariate_type_key: 协变量类型字段的键名
            short_id_key: 短 ID 字段的键名
            text_unit_ids_key: 文本单元 ID 字段的键名
            attributes_key: 属性字段的键名
            
        Returns:
            Covariate: 协变量对象
        """
        return Covariate(
            id=d[id_key],
            short_id=d.get(short_id_key),
            subject_id=d[subject_id_key],
            covariate_type=d.get(covariate_type_key, "claim"),
            text_unit_ids=d.get(text_unit_ids_key),
            attributes=d.get(attributes_key),
        )

