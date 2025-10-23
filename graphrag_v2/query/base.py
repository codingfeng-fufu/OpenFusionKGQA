"""查询引擎基础类。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SearchResult:
    """搜索结果。"""
    
    # 响应文本
    response: str
    # 上下文数据（DataFrame 格式）
    context_data: dict[str, pd.DataFrame]
    # 上下文文本（实际在上下文窗口中的文本）
    context_text: str | list[str]
    # 完成时间（秒）
    completion_time: float
    # LLM 调用次数
    llm_calls: int
    # Prompt tokens 数量
    prompt_tokens: int
    # 输出 tokens 数量
    output_tokens: int
    # LLM 调用次数分类统计
    llm_calls_categories: dict[str, int] | None = None
    # Prompt tokens 分类统计
    prompt_tokens_categories: dict[str, int] | None = None
    # 输出 tokens 分类统计
    output_tokens_categories: dict[str, int] | None = None


class BaseSearch(ABC):
    """搜索引擎基类。
    
    所有搜索引擎（Global、Local、DRIFT 等）都继承自这个基类。
    """
    
    def __init__(
        self,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ):
        """初始化搜索引擎。
        
        Args:
            llm_params: LLM 参数
            context_builder_params: 上下文构建器参数
        """
        self.llm_params = llm_params or {}
        self.context_builder_params = context_builder_params or {}
    
    @abstractmethod
    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> SearchResult:
        """执行搜索。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            SearchResult: 搜索结果
        """
        raise NotImplementedError("子类必须实现 search 方法")
    
    @abstractmethod
    async def stream_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """流式搜索。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Yields:
            str: 响应文本片段
        """
        yield ""  # 使其成为异步生成器
        raise NotImplementedError("子类必须实现 stream_search 方法")

