"""存储配置模型。

参考: graphrag/config/models/storage_config.py
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from graphrag_v2.config.defaults import storage_defaults
from graphrag_v2.config.enums import StorageType


class StorageConfig(BaseModel):
    """存储配置类。
    
    定义数据存储的配置，支持本地文件、内存、Blob 等多种存储类型。
    """

    type: StorageType | str = Field(
        description="存储类型",
        default=storage_defaults.type,
    )
    
    base_dir: str = Field(
        description="基础目录路径",
        default=storage_defaults.base_dir,
    )

    @field_validator("base_dir", mode="before")
    @classmethod
    def validate_base_dir(cls, value, info):
        """验证基础目录路径。
        
        当使用本地文件存储时，确保路径格式正确。
        """
        # 如果不是文件存储类型，直接返回
        if info.data.get("type") != StorageType.file:
            return value
        # 转换为标准路径格式
        return str(Path(value))

    connection_string: str | None = Field(
        description="存储连接字符串（用于 Blob 存储）",
        default=storage_defaults.connection_string,
    )
    
    container_name: str | None = Field(
        description="容器名称（用于 Blob 存储）",
        default=storage_defaults.container_name,
    )

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True

