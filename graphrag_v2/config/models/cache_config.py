"""缓存配置模型。

参考: graphrag/config/models/cache_config.py
"""

from pydantic import BaseModel, Field

from graphrag_v2.config.defaults import cache_defaults
from graphrag_v2.config.enums import CacheType


class CacheConfig(BaseModel):
    """缓存配置类。
    
    定义 LLM 调用结果的缓存配置，用于节省成本和提高性能。
    """

    type: CacheType | str = Field(
        description="缓存类型",
        default=cache_defaults.type,
    )
    
    base_dir: str = Field(
        description="缓存基础目录（用于文件缓存）",
        default=cache_defaults.base_dir,
    )
    
    connection_string: str | None = Field(
        description="缓存连接字符串（用于 Blob 缓存）",
        default=cache_defaults.connection_string,
    )
    
    container_name: str | None = Field(
        description="容器名称（用于 Blob 缓存）",
        default=cache_defaults.container_name,
    )

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True

