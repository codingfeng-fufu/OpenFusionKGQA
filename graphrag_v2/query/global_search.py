"""Global Search 实现。"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pandas as pd

from graphrag_v2.query.base import BaseSearch, SearchResult
from graphrag_v2.query.context_builder import ContextBuilderResult, GlobalContextBuilder

logger = logging.getLogger(__name__)


@dataclass
class GlobalSearchResult(SearchResult):
    """Global Search 结果。"""

    # Map 阶段的响应
    map_responses: list[str] = None
    # Reduce 阶段的上下文数据
    reduce_context_data: dict[str, pd.DataFrame] = None
    # Reduce 阶段的上下文文本
    reduce_context_text: str = None

    def __post_init__(self):
        """初始化后处理。"""
        if self.map_responses is None:
            self.map_responses = []
        if self.reduce_context_data is None:
            self.reduce_context_data = {}
        if self.reduce_context_text is None:
            self.reduce_context_text = ""


class GlobalSearch(BaseSearch):
    """Global Search 实现。
    
    Global Search 使用 Map-Reduce 模式：
    1. Map 阶段：将社区报告分批，对每批并行生成中间答案
    2. Reduce 阶段：合并中间答案生成最终答案
    
    这种方式适合回答需要全局视角的问题，例如：
    - "总结整个数据集的主要主题"
    - "比较不同社区的特点"
    - "找出最重要的实体"
    """
    
    def __init__(
        self,
        context_builder: GlobalContextBuilder,
        map_system_prompt: str | None = None,
        reduce_system_prompt: str | None = None,
        response_type: str = "multiple paragraphs",
        max_data_tokens: int = 8000,
        map_max_length: int = 1000,
        reduce_max_length: int = 2000,
        concurrent_coroutines: int = 32,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ):
        """初始化 Global Search。
        
        Args:
            context_builder: 上下文构建器
            map_system_prompt: Map 阶段的系统提示
            reduce_system_prompt: Reduce 阶段的系统提示
            response_type: 响应类型
            max_data_tokens: 最大数据 tokens 数
            map_max_length: Map 阶段最大响应长度
            reduce_max_length: Reduce 阶段最大响应长度
            concurrent_coroutines: 并发协程数
            llm_params: LLM 参数
            context_builder_params: 上下文构建器参数
        """
        super().__init__(llm_params, context_builder_params)
        self.context_builder = context_builder
        self.map_system_prompt = map_system_prompt or self._default_map_prompt()
        self.reduce_system_prompt = reduce_system_prompt or self._default_reduce_prompt()
        self.response_type = response_type
        self.max_data_tokens = max_data_tokens
        self.map_max_length = map_max_length
        self.reduce_max_length = reduce_max_length
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
    
    def _default_map_prompt(self) -> str:
        """默认的 Map 阶段提示。"""
        return """---角色---

你是一个有帮助的助手，负责回答关于提供的数据的问题。

---目标---

根据下面的数据表格，生成一个全面的回答。
如果你不知道答案，就说你不知道。不要编造信息。

---数据---

{context_data}

---目标---

根据上述数据，回答以下问题。
以 {response_type} 的形式回答。
"""
    
    def _default_reduce_prompt(self) -> str:
        """默认的 Reduce 阶段提示。"""
        return """---角色---

你是一个有帮助的助手，负责综合多个分析结果来回答问题。

---目标---

根据下面的多个分析结果，生成一个全面、连贯的最终回答。
确保回答涵盖所有重要信息，并且逻辑清晰。

---分析结果---

{context_data}

---目标---

综合上述分析结果，回答以下问题。
以 {response_type} 的形式回答。
"""
    
    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """执行 Global Search。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            GlobalSearchResult: 搜索结果
        """
        start_time = time.time()
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        
        # 步骤 1: 构建上下文（获取社区报告并分批）
        logger.info(f"Global Search: 构建上下文，查询: {query}")
        context_result = await self.context_builder.build_context(
            query=query,
            **self.context_builder_params,
        )
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens
        
        # 步骤 2: Map 阶段 - 对每批数据并行生成中间答案
        logger.info(f"Global Search: Map 阶段，批次数: {len(context_result.context_chunks)}")
        map_responses = await self._map_responses(
            context_chunks=context_result.context_chunks,
            query=query,
        )
        llm_calls["map"] = len(map_responses)
        # 简化版本：假设每个 map 调用使用相同的 tokens
        prompt_tokens["map"] = len(map_responses) * 500  # 估计值
        output_tokens["map"] = sum(len(r) for r in map_responses)
        
        # 步骤 3: Reduce 阶段 - 合并中间答案生成最终答案
        logger.info(f"Global Search: Reduce 阶段")
        final_response = await self._reduce_responses(
            map_responses=map_responses,
            query=query,
        )
        llm_calls["reduce"] = 1
        prompt_tokens["reduce"] = len("\n\n".join(map_responses)) + 200  # 估计值
        output_tokens["reduce"] = len(final_response)
        
        return GlobalSearchResult(
            response=final_response,
            context_data=context_result.context_records,
            context_text=context_result.context_chunks,
            completion_time=time.time() - start_time,
            llm_calls=sum(llm_calls.values()),
            prompt_tokens=sum(prompt_tokens.values()),
            output_tokens=sum(output_tokens.values()),
            llm_calls_categories=llm_calls,
            prompt_tokens_categories=prompt_tokens,
            output_tokens_categories=output_tokens,
            map_responses=map_responses,
            reduce_context_data=context_result.context_records,
            reduce_context_text="\n\n".join(map_responses),
        )
    
    async def _map_responses(
        self,
        context_chunks: list[str],
        query: str,
    ) -> list[str]:
        """Map 阶段：对每批数据生成中间答案。
        
        Args:
            context_chunks: 上下文数据块
            query: 查询文本
            
        Returns:
            list[str]: 中间答案列表
        """
        # 简化版本：使用模拟响应
        # 在生产环境中，这里应该调用 LLM
        responses = []
        for i, chunk in enumerate(context_chunks):
            response = f"[Map 响应 {i+1}] 基于提供的数据，这是对查询 '{query}' 的分析结果。数据包含 {len(chunk)} 个字符的信息。"
            responses.append(response)
        return responses
    
    async def _reduce_responses(
        self,
        map_responses: list[str],
        query: str,
    ) -> str:
        """Reduce 阶段：合并中间答案生成最终答案。
        
        Args:
            map_responses: Map 阶段的响应列表
            query: 查询文本
            
        Returns:
            str: 最终答案
        """
        # 简化版本：使用模拟响应
        # 在生产环境中，这里应该调用 LLM
        combined = "\n\n".join(map_responses)
        final_response = f"""基于对 {len(map_responses)} 个数据批次的分析，以下是对查询 "{query}" 的综合回答：

{combined}

总结：这是一个综合了多个数据源的全局视角回答。"""
        return final_response
    
    async def stream_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """流式 Global Search。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Yields:
            str: 响应文本片段
        """
        # 简化版本：直接返回完整结果
        result = await self.search(query, **kwargs)
        yield result.response

