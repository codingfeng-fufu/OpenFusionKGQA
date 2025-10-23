"""带有名称/标题的基础数据类。"""

from dataclasses import dataclass

from graphrag_v2.data_model.identified import Identified


@dataclass
class Named(Identified):
    """带有名称/标题的基础数据类。

    继承自 Identified，添加了 title 字段。
    用于所有需要名称的数据模型（如实体、社区等）。

    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        title: 名称/标题
    """

    title: str = ""
    """名称/标题（默认为空字符串）。"""

