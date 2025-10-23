"""默认配置值定义。

参考微软 GraphRAG 的 defaults.py，使用 dataclass 定义默认配置值。
"""

from dataclasses import dataclass, field
from typing import ClassVar

from graphrag_v2.config.enums import (
    AuthType,
    CacheType,
    ChunkStrategyType,
    InputFileType,
    ModelType,
    StorageType,
)

# 常量定义
DEFAULT_OUTPUT_BASE_DIR = "output"
DEFAULT_CHAT_MODEL_ID = "default_chat_model"
DEFAULT_CHAT_MODEL_TYPE = ModelType.Chat
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHAT_MODEL_AUTH_TYPE = AuthType.APIKey
DEFAULT_EMBEDDING_MODEL_ID = "default_embedding_model"
DEFAULT_EMBEDDING_MODEL_TYPE = ModelType.Embedding
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL_AUTH_TYPE = AuthType.APIKey
DEFAULT_MODEL_PROVIDER = "openai"

ENCODING_MODEL = "cl100k_base"


@dataclass
class CacheDefaults:
    """缓存默认配置。"""

    type: ClassVar[CacheType] = CacheType.file
    base_dir: str = "cache"
    connection_string: None = None
    container_name: None = None


@dataclass
class ChunksDefaults:
    """文本分块默认配置。"""

    size: int = 300
    overlap: int = 100
    group_by_columns: list[str] = field(default_factory=lambda: ["id"])
    strategy: ClassVar[ChunkStrategyType] = ChunkStrategyType.tokens
    encoding_model: str = "cl100k_base"


@dataclass
class StorageDefaults:
    """存储默认配置。"""

    type: ClassVar[StorageType] = StorageType.file
    base_dir: str = "output"
    connection_string: None = None
    container_name: None = None


@dataclass
class InputDefaults:
    """输入默认配置。"""

    type: ClassVar[InputFileType] = InputFileType.text
    file_pattern: str = r".*\.txt$"
    base_dir: str = "input"
    encoding: str = "utf-8"


@dataclass
class LanguageModelDefaults:
    """语言模型默认配置。"""

    api_key: None = None
    auth_type: ClassVar[AuthType] = AuthType.APIKey
    model_provider: str = DEFAULT_MODEL_PROVIDER
    encoding_model: str = ENCODING_MODEL
    api_base: None = None
    api_version: None = None
    deployment_name: None = None
    organization: None = None
    proxy: None = None
    model_supports_json: bool = True
    request_timeout: float = 60.0
    tokens_per_minute: int = 150_000
    requests_per_minute: int = 10_000
    max_retries: int = 10
    max_retry_wait: float = 10.0
    concurrent_requests: int = 25
    max_tokens: int = 4000
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1


@dataclass
class ExtractGraphDefaults:
    """实体提取默认配置。"""

    enabled: bool = True
    max_gleanings: int = 1
    entity_types: list[str] = field(
        default_factory=lambda: ["organization", "person", "geo", "event"]
    )
    model_id: str = DEFAULT_CHAT_MODEL_ID


@dataclass
class CommunityReportDefaults:
    """社区报告默认配置。"""

    max_length: int = 2000
    max_input_length: int = 8000
    model_id: str = DEFAULT_CHAT_MODEL_ID


@dataclass
class GlobalSearchDefaults:
    """全局搜索默认配置。"""

    max_tokens: int = 12_000
    data_max_tokens: int = 12_000
    map_max_tokens: int = 1000
    reduce_max_tokens: int = 2000
    concurrency: int = 32
    model_id: str = DEFAULT_CHAT_MODEL_ID


@dataclass
class LocalSearchDefaults:
    """局部搜索默认配置。"""

    max_tokens: int = 12_000
    text_unit_prop: float = 0.5
    community_prop: float = 0.1
    top_k_entities: int = 10
    top_k_relationships: int = 10
    model_id: str = DEFAULT_CHAT_MODEL_ID


@dataclass
class GraphRagConfigDefaults:
    """GraphRAG 主配置默认值。"""

    root_dir: str = "."
    cache: CacheDefaults = field(default_factory=CacheDefaults)
    chunks: ChunksDefaults = field(default_factory=ChunksDefaults)
    storage: StorageDefaults = field(default_factory=StorageDefaults)
    input: InputDefaults = field(default_factory=InputDefaults)
    extract_graph: ExtractGraphDefaults = field(default_factory=ExtractGraphDefaults)
    community_reports: CommunityReportDefaults = field(
        default_factory=CommunityReportDefaults
    )
    global_search: GlobalSearchDefaults = field(default_factory=GlobalSearchDefaults)
    local_search: LocalSearchDefaults = field(default_factory=LocalSearchDefaults)


# 全局默认配置实例
graphrag_config_defaults = GraphRagConfigDefaults()
language_model_defaults = LanguageModelDefaults()
cache_defaults = CacheDefaults()
chunks_defaults = ChunksDefaults()
storage_defaults = StorageDefaults()
input_defaults = InputDefaults()


def create_default_config():
    """创建默认配置。

    Returns:
        GraphRagConfig: 默认配置实例
    """
    from graphrag_v2.config.models.graph_rag_config import GraphRagConfig

    # 使用默认值创建配置（Pydantic 会自动使用 Field 中的默认值）
    config = GraphRagConfig()

    return config

