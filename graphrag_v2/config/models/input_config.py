"""输入配置模型。

参考: graphrag/config/models/input_config.py
"""

from pydantic import BaseModel, Field

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
    
    storage: StorageConfig = Field(
        description="输入存储配置",
        default_factory=lambda: StorageConfig(base_dir=input_defaults.base_dir),
    )

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True

