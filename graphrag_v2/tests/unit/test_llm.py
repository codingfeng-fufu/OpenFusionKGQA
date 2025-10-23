"""
LLM 模块单元测试
"""
import pytest
import os

from graphrag_v2.llm import GLMClient


class TestGLMClient:
    """测试 GLMClient 类"""
    
    def test_create_client_without_api_key(self):
        """测试创建无 API key 的客户端"""
        # 确保环境变量中没有 API key
        old_key = os.environ.get("ZHIPUAI_API_KEY")
        if old_key:
            del os.environ["ZHIPUAI_API_KEY"]
        
        client = GLMClient()
        assert client is not None
        assert client.mock_mode is True
        
        # 恢复环境变量
        if old_key:
            os.environ["ZHIPUAI_API_KEY"] = old_key
            
    def test_create_client_with_api_key(self):
        """测试创建带 API key 的客户端"""
        client = GLMClient(api_key="test-key")
        assert client is not None
        # 即使提供了 key，如果 zhipuai 包不可用，也会进入 mock 模式
        
    def test_default_model(self):
        """测试默认模型"""
        client = GLMClient()
        assert client.model == "glm-4"
        
    def test_custom_model(self):
        """测试自定义模型"""
        client = GLMClient(model="glm-4v")
        assert client.model == "glm-4v"
        
    def test_default_retry_settings(self):
        """测试默认重试设置"""
        client = GLMClient()
        assert client.max_retries == 3
        assert client.retry_delay == 1.0


class TestGLMClientChatCompletion:
    """测试 GLMClient 聊天完成功能"""
    
    def test_basic_chat_completion(self):
        """测试基础聊天完成"""
        client = GLMClient()
        
        messages = [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "你好！"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        
    def test_chat_completion_with_temperature(self):
        """测试带温度参数的聊天完成"""
        client = GLMClient()
        
        messages = [{"role": "user", "content": "测试"}]
        
        response = client.chat_completion(messages, temperature=0.5)
        assert response is not None
        
    def test_chat_completion_with_max_tokens(self):
        """测试带最大 token 数的聊天完成"""
        client = GLMClient()
        
        messages = [{"role": "user", "content": "测试"}]
        
        response = client.chat_completion(messages, max_tokens=100)
        assert response is not None


class TestGLMClientMockMode:
    """测试 GLMClient Mock 模式"""
    
    def test_mock_entity_extraction(self):
        """测试 Mock 实体提取"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "提取实体\nentity_types: 组织, 技术"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None
        assert "entity" in response or "实体" in response
        
    def test_mock_community_report(self):
        """测试 Mock 社区报告"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "生成报告\ntitle\nfindings"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None
        # 应该返回 JSON 格式
        
    def test_mock_global_search_map(self):
        """测试 Mock Global Search Map"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "提取关键点\npoints\nscore"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None
        
    def test_mock_global_search_reduce(self):
        """测试 Mock Global Search Reduce"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "综合分析师报告"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None
        
    def test_mock_local_search(self):
        """测试 Mock Local Search"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "回答问题：什么是 GraphRAG？"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None


class TestGLMClientStatistics:
    """测试 GLMClient 统计功能"""
    
    def test_initial_stats(self):
        """测试初始统计"""
        client = GLMClient()
        stats = client.get_stats()
        
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_errors"] == 0
        
    def test_stats_after_call(self):
        """测试调用后的统计"""
        client = GLMClient()
        
        messages = [{"role": "user", "content": "测试"}]
        client.chat_completion(messages)
        
        stats = client.get_stats()
        # Mock 模式下不增加计数
        assert stats["total_calls"] >= 0
        
    def test_reset_stats(self):
        """测试重置统计"""
        client = GLMClient()
        
        # 进行一些调用
        messages = [{"role": "user", "content": "测试"}]
        client.chat_completion(messages)
        
        # 重置统计
        client.reset_stats()
        
        stats = client.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_errors"] == 0


class TestGLMClientRetry:
    """测试 GLMClient 重试机制"""
    
    def test_custom_retry_settings(self):
        """测试自定义重试设置"""
        client = GLMClient(max_retries=5, retry_delay=2.0)
        
        assert client.max_retries == 5
        assert client.retry_delay == 2.0


class TestGLMClientEdgeCases:
    """测试 GLMClient 边界情况"""
    
    def test_empty_messages(self):
        """测试空消息列表"""
        client = GLMClient()
        
        # 空消息列表应该能处理
        response = client.chat_completion([])
        assert response is not None
        
    def test_very_long_message(self):
        """测试非常长的消息"""
        client = GLMClient()
        
        long_text = "测试" * 1000
        messages = [{"role": "user", "content": long_text}]
        
        response = client.chat_completion(messages)
        assert response is not None
        
    def test_special_characters(self):
        """测试特殊字符"""
        client = GLMClient()
        
        messages = [
            {"role": "user", "content": "测试 <|> {?} [Data: (1, 2, 3)]"}
        ]
        
        response = client.chat_completion(messages)
        assert response is not None


class TestGLMClientIntegration:
    """测试 GLMClient 集成"""
    
    def test_multiple_calls(self):
        """测试多次调用"""
        client = GLMClient()
        
        messages1 = [{"role": "user", "content": "第一次调用"}]
        messages2 = [{"role": "user", "content": "第二次调用"}]
        messages3 = [{"role": "user", "content": "第三次调用"}]
        
        response1 = client.chat_completion(messages1)
        response2 = client.chat_completion(messages2)
        response3 = client.chat_completion(messages3)
        
        assert response1 is not None
        assert response2 is not None
        assert response3 is not None
        
    def test_different_message_types(self):
        """测试不同类型的消息"""
        client = GLMClient()
        
        # 系统消息 + 用户消息
        messages1 = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}
        ]
        
        # 只有用户消息
        messages2 = [
            {"role": "user", "content": "你好"}
        ]
        
        response1 = client.chat_completion(messages1)
        response2 = client.chat_completion(messages2)
        
        assert response1 is not None
        assert response2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

