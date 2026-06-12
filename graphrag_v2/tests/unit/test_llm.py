"""
LLM 模块单元测试
"""
import pytest
import os
import time
from types import SimpleNamespace

from graphrag_v2.llm import (
    GLMClient,
    LLMProviderError,
    OpenAICompatibleClient,
    create_chat_provider,
)


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

    def test_strict_real_call_does_not_fallback_to_mock(self):
        """测试严格真实调用失败时不会回退到 mock。"""
        client = GLMClient(
            api_key="test-key",
            max_retries=1,
            fallback_to_mock_on_error=False,
        )
        client._has_zhipuai = True
        client.mock_mode = False
        client.client = _FailingZhipuClient()

        with pytest.raises(RuntimeError, match="api unavailable"):
            client.chat_completion([{"role": "user", "content": "测试"}])

        assert client.total_errors == 1


class TestLLMProviderRegistry:
    """测试 LLM provider registry。"""

    def test_create_glm_provider_allows_mock_when_not_required(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        provider = create_chat_provider(
            "zhipu",
            SimpleNamespace(
                api_key=None,
                model="glm-4",
                max_retries=1,
                max_retry_wait=1.0,
                prompt_token_cost_per_1k=None,
                completion_token_cost_per_1k=None,
            ),
        )

        assert provider.provider_name == "glm"
        assert provider.mock_mode is True

    def test_create_glm_provider_requires_real_client(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)

        with pytest.raises(LLMProviderError, match="requires a configured real client"):
            create_chat_provider(
                "glm",
                SimpleNamespace(
                    api_key=None,
                    model="glm-4",
                    max_retries=1,
                    max_retry_wait=1.0,
                    prompt_token_cost_per_1k=None,
                    completion_token_cost_per_1k=None,
                ),
                require_real=True,
            )

    def test_unknown_provider_fails(self):
        with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
            create_chat_provider("unknown-provider", SimpleNamespace())

    def test_openai_compatible_provider_requires_endpoint(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("LOCAL_LLM_API_BASE", raising=False)

        with pytest.raises(LLMProviderError, match="requires api_base"):
            create_chat_provider(
                "openai",
                SimpleNamespace(
                    api_base=None,
                    api_key=None,
                    model="local-model",
                    max_retries=1,
                    max_retry_wait=1.0,
                    request_timeout=1.0,
                    max_tokens=None,
                    prompt_token_cost_per_1k=None,
                    completion_token_cost_per_1k=None,
                ),
                require_real=True,
            )

    def test_deepseek_provider_uses_openai_compatible_defaults(self, monkeypatch):
        monkeypatch.delenv("KGQA_REAL_LLM_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
        provider = create_chat_provider(
            "deepseek",
            SimpleNamespace(
                api_base=None,
                api_key=None,
                model="",
                max_retries=1,
                max_retry_wait=1.0,
                request_timeout=1.0,
                max_tokens=None,
                supports_guided_json=None,
                prompt_token_cost_per_1k=None,
                completion_token_cost_per_1k=None,
            ),
            require_real=True,
        )

        assert provider.provider_name == "deepseek"
        assert provider.api_base == "https://api.deepseek.com"
        assert provider.api_key == "deepseek-test-key"
        assert provider.model == "deepseek-v4-flash"
        assert provider.supports_guided_json is False
        assert provider.supports_response_format_json is True

    def test_deepseek_json_mode_can_be_disabled(self, monkeypatch):
        monkeypatch.delenv("KGQA_REAL_LLM_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
        provider = create_chat_provider(
            "deepseek",
            SimpleNamespace(
                api_base=None,
                api_key=None,
                model="",
                max_retries=1,
                max_retry_wait=1.0,
                request_timeout=1.0,
                max_tokens=None,
                supports_guided_json=None,
                model_supports_json=False,
                prompt_token_cost_per_1k=None,
                completion_token_cost_per_1k=None,
            ),
            require_real=True,
        )

        assert provider.supports_response_format_json is False

    def test_deepseek_provider_requires_api_key_when_real(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("KGQA_REAL_LLM_API_KEY", raising=False)

        with pytest.raises(LLMProviderError, match="requires an API key"):
            create_chat_provider(
                "deepseek",
                SimpleNamespace(
                    api_base=None,
                    api_key=None,
                    model="",
                    max_retries=1,
                    max_retry_wait=1.0,
                    request_timeout=1.0,
                    max_tokens=None,
                    supports_guided_json=None,
                    prompt_token_cost_per_1k=None,
                    completion_token_cost_per_1k=None,
                ),
                require_real=True,
            )

    def test_generic_real_llm_api_key_env_is_supported(self, monkeypatch):
        monkeypatch.setenv("KGQA_REAL_LLM_API_BASE", "https://llm.example/v1")
        monkeypatch.setenv("KGQA_REAL_LLM_API_KEY", "generic-test-key")
        provider = create_chat_provider(
            "openai-compatible",
            SimpleNamespace(
                api_base=None,
                api_key=None,
                model="provider-model",
                max_retries=1,
                max_retry_wait=1.0,
                request_timeout=1.0,
                max_tokens=None,
                supports_guided_json=None,
                prompt_token_cost_per_1k=None,
                completion_token_cost_per_1k=None,
            ),
            require_real=True,
        )

        assert provider.provider_name == "openai-compatible"
        assert provider.api_base == "https://llm.example/v1"
        assert provider.api_key == "generic-test-key"
        assert provider.model == "provider-model"

    def test_openai_compatible_provider_defaults_guided_json_off(self):
        provider = create_chat_provider(
            "openai-compatible",
            _openai_compatible_config(supports_guided_json=None),
            require_real=True,
        )

        assert provider.provider_name == "openai-compatible"
        assert provider.supports_guided_json is False

    def test_vllm_provider_defaults_guided_json_on(self):
        provider = create_chat_provider(
            "vllm",
            _openai_compatible_config(supports_guided_json=None),
            require_real=True,
        )

        assert provider.provider_name == "vllm"
        assert provider.supports_guided_json is True

    def test_guided_json_support_can_be_overridden(self):
        openai_provider = create_chat_provider(
            "openai-compatible",
            _openai_compatible_config(supports_guided_json=True),
            require_real=True,
        )
        vllm_provider = create_chat_provider(
            "vllm",
            _openai_compatible_config(supports_guided_json=False),
            require_real=True,
        )

        assert openai_provider.supports_guided_json is True
        assert vllm_provider.supports_guided_json is False

    def test_openai_compatible_provider_exposes_json_mode_support(self):
        provider = create_chat_provider(
            "openai-compatible",
            _openai_compatible_config(model_supports_json=True),
            require_real=True,
        )

        assert provider.supports_response_format_json is True

    def test_openai_compatible_client_adds_json_mode_only_when_requested(
        self,
        monkeypatch,
    ):
        client = OpenAICompatibleClient(
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            model="local-model",
            supports_response_format_json=True,
        )
        payloads = []

        def fake_post_json(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": "{}"}}]}

        monkeypatch.setattr(client, "_post_json", fake_post_json)

        client.chat_completion([{"role": "user", "content": "hello"}])
        client.chat_completion(
            [{"role": "user", "content": "return JSON"}],
            response_format_json=True,
        )

        assert "response_format" not in payloads[0]
        assert payloads[1]["response_format"] == {"type": "json_object"}

    def test_openai_compatible_client_enforces_body_read_timeout(self, monkeypatch):
        class SlowResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                time.sleep(1.0)
                return b'{"choices":[{"message":{"content":"late"}}]}'

        def slow_urlopen(request, timeout):
            return SlowResponse()

        monkeypatch.setattr("urllib.request.urlopen", slow_urlopen)
        client = OpenAICompatibleClient(
            api_base="https://llm.example/v1",
            api_key="test-key",
            model="model",
            max_retries=1,
            request_timeout=0.01,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            client.chat_completion([{"role": "user", "content": "hello"}])

    def test_openai_compatible_client_sets_body_socket_timeout(self, monkeypatch):
        class FakeSocket:
            def __init__(self):
                self.timeout = None

            def settimeout(self, timeout):
                self.timeout = timeout

        sock = FakeSocket()

        class Response:
            fp = SimpleNamespace(raw=SimpleNamespace(_sock=sock))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
        client = OpenAICompatibleClient(
            api_base="https://llm.example/v1",
            api_key="test-key",
            model="model",
            max_retries=1,
            request_timeout=7.0,
        )

        assert client.chat_completion([{"role": "user", "content": "hello"}]) == "ok"
        assert sock.timeout == 7.0


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
        assert "entities" in response or "实体" in response
        
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
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["total_errors"] == 0
        assert stats["total_latency_seconds"] == 0.0
        assert stats["estimated_cost"] is None
        
    def test_stats_after_call(self):
        """测试调用后的统计"""
        client = GLMClient()
        
        messages = [{"role": "user", "content": "测试"}]
        client.chat_completion(messages)
        
        stats = client.get_stats()
        # Mock 模式下不增加计数
        assert stats["total_calls"] >= 0

    def test_real_call_stats_include_usage_latency_and_cost(self):
        """测试真实调用统计包含 usage、latency 和成本估算"""
        client = GLMClient(
            api_key="test-key",
            prompt_token_cost_per_1k=0.1,
            completion_token_cost_per_1k=0.2,
        )
        client._has_zhipuai = True
        client.mock_mode = False
        client.client = _FakeZhipuClient(
            content="ok",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

        response = client.chat_completion([{"role": "user", "content": "测试"}])
        stats = client.get_stats()

        assert response == "ok"
        assert stats["total_calls"] == 1
        assert stats["prompt_tokens"] == 10
        assert stats["completion_tokens"] == 5
        assert stats["total_tokens"] == 15
        assert stats["total_latency_seconds"] >= 0
        assert stats["max_latency_seconds"] >= 0
        assert stats["average_latency_seconds"] >= 0
        assert stats["estimated_cost"] == 0.002
        
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
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["total_errors"] == 0
        assert stats["total_latency_seconds"] == 0.0


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


def _openai_compatible_config(
    supports_guided_json=None,
    model_supports_json=True,
):
    return SimpleNamespace(
        api_base="http://localhost:8000/v1",
        api_key="test-key",
        model="local-model",
        max_retries=1,
        max_retry_wait=1.0,
        request_timeout=1.0,
        max_tokens=None,
        model_supports_json=model_supports_json,
        supports_guided_json=supports_guided_json,
        prompt_token_cost_per_1k=None,
        completion_token_cost_per_1k=None,
    )


class _FakeZhipuClient:
    def __init__(self, content: str, usage: dict):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content),
                        )
                    ],
                    usage=usage,
                )
            )
        )


class _FailingZhipuClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        raise RuntimeError("api unavailable")
