"""配置枚举类型定义。

参考微软 GraphRAG 的 enums.py，定义配置中使用的枚举类型。
"""

from __future__ import annotations

from enum import Enum


class CacheType(str, Enum):
    """缓存类型枚举。"""

    file = "file"
    """文件缓存"""
    memory = "memory"
    """内存缓存"""
    none = "none"
    """不使用缓存"""
    blob = "blob"
    """Blob 存储缓存"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class InputFileType(str, Enum):
    """输入文件类型枚举。"""

    csv = "csv"
    """CSV 文件"""
    text = "txt"  # 改为 txt 以匹配文件扩展名
    """文本文件"""
    json = "json"
    """JSON 文件"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class StorageType(str, Enum):
    """存储类型枚举。"""

    file = "file"
    """文件存储"""
    memory = "memory"
    """内存存储"""
    blob = "blob"
    """Blob 存储"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class ModelType(str, Enum):
    """语言模型类型枚举。"""

    # Embeddings
    OpenAIEmbedding = "openai_embedding"
    """OpenAI 嵌入模型"""
    AzureOpenAIEmbedding = "azure_openai_embedding"
    """Azure OpenAI 嵌入模型"""
    Embedding = "embedding"
    """通用嵌入模型（LiteLLM）"""

    # Chat Completion
    OpenAIChat = "openai_chat"
    """OpenAI 聊天模型"""
    AzureOpenAIChat = "azure_openai_chat"
    """Azure OpenAI 聊天模型"""
    Chat = "chat"
    """通用聊天模型（LiteLLM）"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class AuthType(str, Enum):
    """认证类型枚举。"""

    APIKey = "api_key"
    """API Key 认证"""
    AzureManagedIdentity = "azure_managed_identity"
    """Azure 托管身份认证"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class ChunkStrategyType(str, Enum):
    """文本分块策略枚举。"""

    tokens = "tokens"
    """基于 token 的分块"""
    sentence = "sentence"
    """基于句子的分块"""

    def __repr__(self):
        """获取字符串表示。"""
        return f'"{self.value}"'


class SearchMethod(str, Enum):
    """搜索方法枚举。"""

    LOCAL = "local"
    """局部搜索"""
    GLOBAL = "global"
    """全局搜索"""
    DRIFT = "drift"
    """DRIFT 混合搜索"""

    def __str__(self):
        """返回枚举值的字符串表示。"""
        return self.value


class IndexingMethod(str, Enum):
    """索引方法枚举。"""

    Standard = "standard"
    """标准索引"""
    Fast = "fast"
    """快速索引（使用 NLP 而非 LLM）"""
    StandardUpdate = "standard_update"
    """标准增量更新"""
    FastUpdate = "fast_update"
    """快速增量更新"""

    def __str__(self):
        """返回枚举值的字符串表示。"""
        return self.value

