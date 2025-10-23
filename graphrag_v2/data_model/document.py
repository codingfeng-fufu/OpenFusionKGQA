"""文档数据模型。"""

from dataclasses import dataclass, field
from typing import Any

from graphrag_v2.data_model.named import Named


@dataclass
class Document(Named):
    """文档数据模型。
    
    表示系统中的一个文档。
    文档会被分块成多个文本单元进行处理。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        title: 文档标题
        type: 文档类型（默认为 "text"）
        text_unit_ids: 文档中的文本单元 ID 列表（默认为空列表）
        text: 文档的原始文本内容（默认为空字符串）
        attributes: 额外的属性字典（可选，如作者、创建时间等）
    """

    type: str = "text"
    """文档类型（默认为 "text"）。"""

    text_unit_ids: list[str] = field(default_factory=list)
    """文档中的文本单元 ID 列表（默认为空列表）。"""

    text: str = ""
    """文档的原始文本内容（默认为空字符串）。"""

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。
    
    可以包含作者、创建时间、元数据等结构化属性。
    """

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        title_key: str = "title",
        type_key: str = "type",
        text_key: str = "text",
        text_units_key: str = "text_units",
        attributes_key: str = "attributes",
    ) -> "Document":
        """从字典创建文档对象。
        
        Args:
            d: 包含文档数据的字典
            id_key: ID 字段的键名
            short_id_key: 短 ID 字段的键名
            title_key: 标题字段的键名
            type_key: 类型字段的键名
            text_key: 文本字段的键名
            text_units_key: 文本单元字段的键名
            attributes_key: 属性字段的键名
            
        Returns:
            Document: 文档对象
        """
        return Document(
            id=d[id_key],
            short_id=d.get(short_id_key),
            title=d[title_key],
            type=d.get(type_key, "text"),
            text=d[text_key],
            text_unit_ids=d.get(text_units_key, []),
            attributes=d.get(attributes_key),
        )

