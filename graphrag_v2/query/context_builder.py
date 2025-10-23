"""上下文构建器。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ContextBuilderResult:
    """上下文构建结果。"""
    
    # 上下文文本块（用于 LLM 输入）
    context_chunks: str | list[str]
    # 上下文记录（DataFrame 格式，用于追踪）
    context_records: dict[str, pd.DataFrame]
    # LLM 调用次数（如果构建上下文需要 LLM）
    llm_calls: int = 0
    # Prompt tokens 数量
    prompt_tokens: int = 0
    # 输出 tokens 数量
    output_tokens: int = 0


class ContextBuilder(ABC):
    """上下文构建器基类。"""
    
    @abstractmethod
    def build_context(
        self,
        query: str,
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """构建上下文。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            ContextBuilderResult: 上下文构建结果
        """
        raise NotImplementedError("子类必须实现 build_context 方法")


class GlobalContextBuilder(ContextBuilder):
    """Global Search 上下文构建器基类。
    
    Global Search 使用 Map-Reduce 模式：
    1. 将社区报告分批
    2. 对每批并行调用 LLM 生成中间答案（Map）
    3. 合并中间答案生成最终答案（Reduce）
    """
    
    @abstractmethod
    async def build_context(
        self,
        query: str,
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """构建 Global Search 上下文。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            ContextBuilderResult: 上下文构建结果
        """
        raise NotImplementedError("子类必须实现 build_context 方法")


class LocalContextBuilder(ContextBuilder):
    """Local Search 上下文构建器基类。
    
    Local Search 基于向量相似度检索：
    1. 使用查询嵌入找到最相关的实体
    2. 获取这些实体的关系和社区
    3. 构建包含实体、关系、社区报告的上下文
    """
    
    @abstractmethod
    def build_context(
        self,
        query: str,
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """构建 Local Search 上下文。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            ContextBuilderResult: 上下文构建结果
        """
        raise NotImplementedError("子类必须实现 build_context 方法")

