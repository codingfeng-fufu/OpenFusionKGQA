"""带有 ID 的基础数据类。"""

from dataclasses import dataclass


@dataclass
class Identified:
    """带有 ID 的基础数据类。

    所有需要唯一标识的数据模型都应该继承此类。

    Attributes:
        id: 唯一标识符（通常是 UUID）
        short_id: 人类可读的短 ID，用于在提示或报告中引用（可选）
    """

    id: str = ""
    """唯一标识符（默认为空字符串）。"""

    short_id: str | None = None
    """人类可读的短 ID，用于在提示或报告中引用（可选）。"""

