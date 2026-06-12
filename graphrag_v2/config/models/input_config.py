"""输入配置模型。

参考: graphrag/config/models/input_config.py
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from graphrag_v2.config.defaults import input_defaults
from graphrag_v2.config.enums import InputFileType
from graphrag_v2.config.models.storage_config import StorageConfig


class InputConfig(BaseModel):
    """输入配置类。
    
    定义输入数据的来源和格式。
    """

    file_type: InputFileType = Field(
        description="输入文件类型",
        default=input_defaults.type,
    )
    
    file_pattern: str = Field(
        description="文件匹配模式（正则表达式）",
        default=input_defaults.file_pattern,
    )
    
    base_dir: str = Field(
        description="输入文件基础目录",
        default=input_defaults.base_dir,
    )
    
    encoding: str = Field(
        description="文件编码",
        default=input_defaults.encoding,
    )

    unsupported_file_policy: Literal["ignore", "warn", "fail"] = Field(
        description="不支持文件类型的处理策略：ignore、warn 或 fail",
        default="ignore",
    )

    max_file_size_bytes: int | None = Field(
        description="单个输入文件的最大字节数；None 表示不限制",
        default=None,
    )

    max_document_count: int | None = Field(
        description="一次索引最多纳入的文档数；None 表示不限制",
        default=None,
    )
    
    storage: StorageConfig = Field(
        description="输入存储配置",
        default_factory=lambda: StorageConfig(base_dir=input_defaults.base_dir),
    )

    @model_validator(mode="after")
    def _validate_model(self):
        if self.max_file_size_bytes is not None and self.max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be at least 1")
        if self.max_document_count is not None and self.max_document_count < 1:
            raise ValueError("max_document_count must be at least 1")
        return self

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True
