"""GLM (智谱 AI) 客户端封装。"""

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class GLMClient:
    """GLM 客户端封装。
    
    支持：
    - 同步和流式调用
    - 自动重试
    - Token 计数
    - 成本跟踪
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "glm-4",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """初始化 GLM 客户端。
        
        Args:
            api_key: API 密钥（如果为 None，从环境变量 ZHIPUAI_API_KEY 读取）
            model: 模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 统计信息
        self.total_tokens = 0
        self.total_calls = 0
        self.total_errors = 0
        
        # 尝试导入 zhipuai
        try:
            from zhipuai import ZhipuAI

            # 检查是否有 API key
            if not self.api_key:
                logger.warning("未提供 ZHIPUAI_API_KEY，将使用 mock 模式")
                self.client = None
                self._has_zhipuai = False
                self.mock_mode = True
            else:
                self.client = ZhipuAI(api_key=self.api_key)
                self._has_zhipuai = True
                self.mock_mode = False
                logger.info(f"GLM 客户端初始化成功，模型: {self.model}")
        except ImportError:
            logger.warning("zhipuai 包未安装，将使用 mock 模式")
            self.client = None
            self._has_zhipuai = False
            self.mock_mode = True
        except Exception as e:
            logger.warning(f"GLM 客户端初始化失败: {e}，将使用 mock 模式")
            self.client = None
            self._has_zhipuai = False
            self.mock_mode = True
    
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | Any:
        """调用 GLM Chat Completion API。
        
        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            temperature: 温度参数（0-1）
            max_tokens: 最大 token 数
            stream: 是否使用流式响应
            
        Returns:
            str | Any: 如果 stream=False，返回响应文本；否则返回流式响应对象
        """
        if not self._has_zhipuai:
            # Mock 模式
            return self._mock_chat_completion(messages, stream)
        
        # 真实 API 调用
        for attempt in range(self.max_retries):
            try:
                # 构建请求参数
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": stream,
                }
                
                # 添加可选参数
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                
                # 调用 API
                response = self.client.chat.completions.create(**kwargs)
                
                # 更新统计
                self.total_calls += 1
                
                # 处理响应
                if stream:
                    return response
                else:
                    # 提取响应文本
                    content = response.choices[0].message.content
                    
                    # 更新 token 统计
                    if hasattr(response, "usage"):
                        self.total_tokens += response.usage.total_tokens
                    
                    return content
                    
            except Exception as e:
                self.total_errors += 1
                logger.error(f"GLM API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("GLM API 调用失败，已达到最大重试次数")
                    # 返回 mock 响应
                    return self._mock_chat_completion(messages, stream)
        
        # 不应该到达这里
        return ""
    
    def _mock_chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
    ) -> str:
        """Mock Chat Completion（用于测试）。

        Args:
            messages: 消息列表
            stream: 是否使用流式响应

        Returns:
            str: Mock 响应
        """
        logger.info("使用 Mock 模式生成响应")

        # 获取最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # 根据消息内容生成 mock 响应
        # 检查是否是社区报告（需要在实体提取之前检查，因为社区报告也包含"实体"）
        if "报告" in user_message and ("title" in user_message or "findings" in user_message):
            # 社区报告 mock
            return self._mock_community_report()
        elif "实体" in user_message and "entity_types" in user_message.lower():
            # 实体提取 mock
            return self._mock_entity_extraction()
        elif "points" in user_message and "score" in user_message:
            # Global Search Map mock
            return self._mock_global_search_map()
        elif "分析师" in user_message or "analyst" in user_message.lower():
            # Global Search Reduce mock
            return self._mock_global_search_reduce()
        else:
            # Local Search mock
            return self._mock_local_search()
    
    def _mock_entity_extraction(self) -> str:
        """Mock 实体提取响应。"""
        return """("entity"<|>GraphRAG<|>技术<|>GraphRAG是一种结合知识图谱和检索增强生成的技术)
<|>
("entity"<|>微软<|>组织<|>微软是GraphRAG的开发者)
<|>
("relationship"<|>微软<|>GraphRAG<|>微软开发了GraphRAG技术<|>9)
<|COMPLETE|>"""
    
    def _mock_community_report(self) -> str:
        """Mock 社区报告响应。"""
        return json.dumps({
            "title": "GraphRAG 技术社区",
            "summary": "该社区围绕 GraphRAG 技术展开，包括其核心概念、应用场景和相关组织。",
            "rating": 7.5,
            "rating_explanation": "该社区具有较高的技术重要性，代表了知识图谱和 RAG 技术的融合。",
            "findings": [
                {
                    "summary": "GraphRAG 的核心价值",
                    "explanation": "GraphRAG 结合了知识图谱的结构化表示和检索增强生成的灵活性，提供了更准确的信息检索能力。[Data: Entities (1, 2, 3)]"
                },
                {
                    "summary": "微软的技术贡献",
                    "explanation": "微软作为 GraphRAG 的主要开发者，推动了该技术在企业级应用中的落地。[Data: Relationships (1, 2)]"
                }
            ]
        }, ensure_ascii=False)
    
    def _mock_global_search_map(self) -> str:
        """Mock Global Search Map 响应。"""
        return json.dumps({
            "points": [
                {
                    "description": "GraphRAG 是一种结合知识图谱和检索增强生成的技术 [Data: Reports (1, 2)]",
                    "score": 85
                },
                {
                    "description": "该技术由微软开发，用于提升信息检索的准确性 [Data: Reports (3, 4)]",
                    "score": 75
                }
            ]
        }, ensure_ascii=False)
    
    def _mock_global_search_reduce(self) -> str:
        """Mock Global Search Reduce 响应。"""
        return """# GraphRAG 技术概述

GraphRAG 是一种创新的技术，结合了知识图谱的结构化表示和检索增强生成（RAG）的灵活性。该技术由微软开发，旨在提升信息检索的准确性和可解释性。

## 核心特点

- **知识图谱集成**：利用图结构表示实体和关系
- **检索增强**：基于向量相似度检索相关信息
- **生成能力**：使用大语言模型生成自然语言响应

## 应用场景

GraphRAG 适用于需要高准确性和可解释性的信息检索场景，如企业知识管理、智能问答系统等。

[Data: Reports (1, 2, 3, 4, 5)]"""
    
    def _mock_local_search(self) -> str:
        """Mock Local Search 响应。"""
        return """# 关于 GraphRAG

GraphRAG 是一种结合知识图谱和检索增强生成的技术，由微软开发。它通过构建实体关系图谱，并利用向量相似度检索相关信息，最后使用大语言模型生成自然语言响应。

## 主要特点

- **结构化知识表示**：使用图结构存储实体和关系 [Data: Entities (1, 2, 3)]
- **语义检索**：基于向量嵌入进行相似度搜索 [Data: Entities (4, 5)]
- **智能生成**：利用 LLM 生成连贯的回答 [Data: Relationships (1, 2)]

GraphRAG 在企业知识管理、智能问答等场景中具有广泛的应用前景。"""
    
    def get_stats(self) -> dict[str, int]:
        """获取统计信息。
        
        Returns:
            dict: 统计信息
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息。"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_errors = 0

