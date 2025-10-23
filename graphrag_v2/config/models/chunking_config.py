"""文本分块配置模型。

参考: graphrag/config/models/chunking_config.py
"""

from pydantic import BaseModel, Field

from graphrag_v2.config.defaults import chunks_defaults
from graphrag_v2.config.enums import ChunkStrategyType


class ChunkingConfig(BaseModel):
    """文本分块配置类。
    
    定义如何将输入文档分割成文本块（chunks）。
    """

    size: int = Field(
        description="分块大小（token 数量）",
        default=chunks_defaults.size,
    )
    
    overlap: int = Field(
        description="分块重叠大小（token 数量）",
        default=chunks_defaults.overlap,
    )
    
    group_by_columns: list[str] = Field(
        description="分组列名",
        default=chunks_defaults.group_by_columns,
    )
    
    strategy: ChunkStrategyType = Field(
        description="分块策略",
        default=chunks_defaults.strategy,
    )
    
    encoding_model: str = Field(
        description="编码模型名称（用于计算 token）",
        default=chunks_defaults.encoding_model,
    )

    def validate_overlap(self):
        """验证重叠大小。

        确保重叠大小小于分块大小。

        Raises:
            ValueError: 如果重叠大小大于等于分块大小
        """
        if self.overlap >= self.size:
            raise ValueError(
                f"分块重叠 ({self.overlap}) 必须小于分块大小 ({self.size})"
            )

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True

