"""GraphRAG 主配置模型。

参考: graphrag/config/models/graph_rag_config.py
"""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from graphrag_v2.config.defaults import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODEL_ID,
    DEFAULT_CHAT_MODEL_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_TYPE,
    DEFAULT_MODEL_PROVIDER,
    graphrag_config_defaults,
)
from graphrag_v2.config.enums import AuthType
from graphrag_v2.config.models.cache_config import CacheConfig
from graphrag_v2.config.models.chunking_config import ChunkingConfig
from graphrag_v2.config.models.input_config import InputConfig
from graphrag_v2.config.models.language_model_config import LanguageModelConfig
from graphrag_v2.config.models.storage_config import StorageConfig


class GraphRagConfig(BaseModel):
    """GraphRAG 主配置类。
    
    这是整个 GraphRAG 系统的顶层配置，包含所有子模块的配置。
    """

    def __str__(self):
        """获取字符串表示。"""
        return self.model_dump_json(indent=4)

    def model_dump(self, **kwargs):
        """导出为字典，包含别名字段以保持向后兼容。"""
        data = super().model_dump(**kwargs)
        # 添加别名字段
        data["storage"] = data.get("output")
        data["llm"] = {"models": data.get("models", {})}
        data["embeddings"] = {"models": data.get("models", {})}
        data["entity_extraction"] = {"max_gleanings": 1}
        return data

    # ========== 基础配置 ==========
    root_dir: str = Field(
        description="项目根目录",
        default=graphrag_config_defaults.root_dir,
    )

    # ========== 语言模型配置 ==========
    models: dict[str, LanguageModelConfig] = Field(
        description="语言模型配置字典",
        default_factory=lambda: {
            DEFAULT_CHAT_MODEL_ID: LanguageModelConfig(
                type=DEFAULT_CHAT_MODEL_TYPE,
                model=DEFAULT_CHAT_MODEL,
                model_provider=DEFAULT_MODEL_PROVIDER,
                auth_type=AuthType.APIKey,
            ),
            DEFAULT_EMBEDDING_MODEL_ID: LanguageModelConfig(
                type=DEFAULT_EMBEDDING_MODEL_TYPE,
                model=DEFAULT_EMBEDDING_MODEL,
                model_provider=DEFAULT_MODEL_PROVIDER,
                auth_type=AuthType.APIKey,
            ),
        },
    )

    # ========== 输入配置 ==========
    input: InputConfig = Field(
        description="输入配置",
        default_factory=InputConfig,
    )

    # ========== 输出配置 ==========
    output: StorageConfig = Field(
        description="输出配置",
        default_factory=StorageConfig,
    )

    # ========== 缓存配置 ==========
    cache: CacheConfig = Field(
        description="缓存配置",
        default_factory=CacheConfig,
    )

    # ========== 文本分块配置 ==========
    chunks: ChunkingConfig = Field(
        description="文本分块配置",
        default_factory=ChunkingConfig,
    )

    # ========== Pipeline 配置 ==========
    workflows: list[str] | None = Field(
        description="自定义工作流列表（可选）。如果为 None，则使用默认的工作流。",
        default=None,
    )

    # ========== 验证方法 ==========
    def _validate_root_dir(self) -> None:
        """验证根目录。
        
        确保根目录存在且是一个有效的目录。
        """
        if self.root_dir.strip() == "":
            self.root_dir = str(Path.cwd())

        root_dir = Path(self.root_dir).resolve()
        if not root_dir.is_dir():
            raise FileNotFoundError(
                f"无效的根目录: {self.root_dir} 不是一个目录"
            )
        self.root_dir = str(root_dir)

    def _validate_models(self) -> None:
        """验证模型配置。
        
        确保至少定义了默认的聊天模型和嵌入模型。
        """
        if DEFAULT_CHAT_MODEL_ID not in self.models:
            raise ValueError(
                f"必须定义默认聊天模型: {DEFAULT_CHAT_MODEL_ID}"
            )
        
        if DEFAULT_EMBEDDING_MODEL_ID not in self.models:
            raise ValueError(
                f"必须定义默认嵌入模型: {DEFAULT_EMBEDDING_MODEL_ID}"
            )

    def _validate_input_base_dir(self) -> None:
        """验证输入基础目录。"""
        if self.input.storage.base_dir.strip() == "":
            raise ValueError("输入存储基础目录不能为空")
        
        # 将相对路径转换为绝对路径
        self.input.storage.base_dir = str(
            (Path(self.root_dir) / self.input.storage.base_dir).resolve()
        )

    def _validate_output_base_dir(self) -> None:
        """验证输出基础目录。"""
        if self.output.base_dir.strip() == "":
            raise ValueError("输出基础目录不能为空")
        
        # 将相对路径转换为绝对路径
        self.output.base_dir = str(
            (Path(self.root_dir) / self.output.base_dir).resolve()
        )

    def _validate_cache_base_dir(self) -> None:
        """验证缓存基础目录。"""
        if self.cache.base_dir.strip() == "":
            self.cache.base_dir = "cache"
        
        # 将相对路径转换为绝对路径
        self.cache.base_dir = str(
            (Path(self.root_dir) / self.cache.base_dir).resolve()
        )

    @model_validator(mode="after")
    def _validate_model(self):
        """模型级别的验证。
        
        在所有字段赋值后执行验证。
        """
        self._validate_root_dir()
        self._validate_models()
        self._validate_input_base_dir()
        self._validate_output_base_dir()
        self._validate_cache_base_dir()
        return self

    # ========== 辅助方法 ==========
    def get_language_model_config(self, model_id: str) -> LanguageModelConfig:
        """获取指定 ID 的语言模型配置。

        Args:
            model_id: 模型 ID

        Returns:
            语言模型配置

        Raises:
            ValueError: 如果模型 ID 不存在
        """
        if model_id not in self.models:
            raise ValueError(
                f"模型 ID {model_id} 不存在于配置中。"
                f"可用的模型: {', '.join(self.models.keys())}"
            )
        return self.models[model_id]

    def get_embedding_model_config(self, model_id: str) -> LanguageModelConfig:
        """获取指定 ID 的嵌入模型配置。

        这是 get_language_model_config 的别名，用于向后兼容。

        Args:
            model_id: 模型 ID

        Returns:
            语言模型配置
        """
        return self.get_language_model_config(model_id)

    # ========== 属性别名（向后兼容）==========
    @property
    def storage(self) -> StorageConfig:
        """存储配置的别名（向后兼容）。"""
        return self.output

    @storage.setter
    def storage(self, value: StorageConfig):
        """设置存储配置。"""
        self.output = value

    @property
    def llm(self) -> "LLMConfigWrapper":
        """LLM配置的包装器（向后兼容）。"""
        return LLMConfigWrapper(self.models)

    @property
    def embeddings(self) -> "EmbeddingsConfigWrapper":
        """嵌入配置的包装器（向后兼容）。"""
        return EmbeddingsConfigWrapper(self.models)

    @property
    def entity_extraction(self) -> "EntityExtractionConfigWrapper":
        """实体提取配置的包装器（向后兼容）。"""
        return EntityExtractionConfigWrapper()

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True


# ========== 配置包装器类（向后兼容）==========
class LLMConfigWrapper:
    """LLM配置包装器，提供向后兼容的接口。"""

    def __init__(self, models: dict[str, LanguageModelConfig]):
        self.models = models


class EmbeddingsConfigWrapper:
    """嵌入配置包装器，提供向后兼容的接口。"""

    def __init__(self, models: dict[str, LanguageModelConfig]):
        self.models = models


class EntityExtractionConfigWrapper:
    """实体提取配置包装器，提供向后兼容的接口。"""

    def __init__(self):
        self.max_gleanings = 1  # 默认值

