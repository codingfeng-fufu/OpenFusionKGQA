"""Local Search 实现。"""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from graphrag_v2.query.base import BaseSearch, SearchResult
from graphrag_v2.query.context_builder import LocalContextBuilder

logger = logging.getLogger(__name__)


class LocalSearch(BaseSearch):
    """Local Search 实现。
    
    Local Search 基于向量相似度检索：
    1. 使用查询嵌入找到最相关的实体
    2. 获取这些实体的关系和社区
    3. 构建包含实体、关系、社区报告的上下文
    4. 使用 LLM 生成答案
    
    这种方式适合回答需要局部细节的问题，例如：
    - "GraphRAG 是什么？"
    - "微软和 OpenAI 的关系是什么？"
    - "Leiden 算法的作用是什么？"
    """
    
    def __init__(
        self,
        context_builder: LocalContextBuilder,
        system_prompt: str | None = None,
        response_type: str = "multiple paragraphs",
        max_tokens: int = 4000,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ):
        """初始化 Local Search。
        
        Args:
            context_builder: 上下文构建器
            system_prompt: 系统提示
            response_type: 响应类型
            max_tokens: 最大 tokens 数
            llm_params: LLM 参数
            context_builder_params: 上下文构建器参数
        """
        super().__init__(llm_params, context_builder_params)
        self.context_builder = context_builder
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.response_type = response_type
        self.max_tokens = max_tokens
    
    def _default_system_prompt(self) -> str:
        """默认的系统提示。"""
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
    
    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> SearchResult:
        """执行 Local Search。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            SearchResult: 搜索结果
        """
        start_time = time.time()
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        
        # 步骤 1: 构建上下文（检索相关实体、关系、社区）
        logger.info(f"Local Search: 构建上下文，查询: {query}")
        context_result = self.context_builder.build_context(
            query=query,
            **self.context_builder_params,
        )
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens
        
        # 步骤 2: 使用 LLM 生成答案
        logger.info(f"Local Search: 生成答案")
        
        # 格式化系统提示
        search_prompt = self.system_prompt.format(
            context_data=context_result.context_chunks,
            response_type=self.response_type,
        )
        
        # 简化版本：使用模拟响应
        # 在生产环境中，这里应该调用 LLM
        response = self._generate_mock_response(query, context_result.context_chunks)
        
        llm_calls["response"] = 1
        prompt_tokens["response"] = len(search_prompt) // 4  # 估计值
        output_tokens["response"] = len(response) // 4  # 估计值
        
        return SearchResult(
            response=response,
            context_data=context_result.context_records,
            context_text=context_result.context_chunks,
            completion_time=time.time() - start_time,
            llm_calls=sum(llm_calls.values()),
            prompt_tokens=sum(prompt_tokens.values()),
            output_tokens=sum(output_tokens.values()),
            llm_calls_categories=llm_calls,
            prompt_tokens_categories=prompt_tokens,
            output_tokens_categories=output_tokens,
        )
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """生成模拟响应。
        
        Args:
            query: 查询文本
            context: 上下文文本
            
        Returns:
            str: 模拟响应
        """
        return f"""基于提供的上下文数据，以下是对查询 "{query}" 的回答：

{context[:500]}...

这是一个基于局部上下文的详细回答，包含了与查询最相关的实体、关系和社区信息。
"""
    
    async def stream_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """流式 Local Search。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Yields:
            str: 响应文本片段
        """
        # 简化版本：直接返回完整结果
        result = await self.search(query, **kwargs)
        yield result.response

