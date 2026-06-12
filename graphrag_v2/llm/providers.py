"""LLM provider registry for chat-completion clients."""

from __future__ import annotations

import os
from typing import Any, Protocol

from graphrag_v2.llm.glm_client import GLMClient
from graphrag_v2.llm.openai_compatible_client import OpenAICompatibleClient
from graphrag_v2.llm.real_llm_config import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_DEFAULT_MODEL,
    SUPPORTED_REAL_LLM_PROVIDERS,
    normalize_real_llm_provider,
)


class ChatProvider(Protocol):
    """Minimal chat provider contract used by extraction and QA."""

    provider_name: str
    mock_mode: bool
    total_errors: int
    supports_guided_json: bool

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | Any:
        """Return a chat-completion response."""

    def get_stats(self) -> dict[str, Any]:
        """Return provider runtime stats."""


class LLMProviderError(ValueError):
    """Raised when an LLM provider cannot be created safely."""


SUPPORTED_LLM_PROVIDERS = SUPPORTED_REAL_LLM_PROVIDERS


def create_chat_provider(
    provider: str,
    model_config: Any,
    require_real: bool = False,
) -> ChatProvider:
    """Create a chat provider from model config."""
    provider_key = _normalize_provider(provider)
    if provider_key not in SUPPORTED_LLM_PROVIDERS:
        raise LLMProviderError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}"
        )

    if provider_key == "glm":
        client = GLMClient(
            api_key=getattr(model_config, "api_key", None),
            model=getattr(model_config, "model", "glm-4"),
            max_retries=getattr(model_config, "max_retries", 3),
            retry_delay=min(float(getattr(model_config, "max_retry_wait", 5.0)), 5.0),
            prompt_token_cost_per_1k=getattr(
                model_config,
                "prompt_token_cost_per_1k",
                None,
            ),
            completion_token_cost_per_1k=getattr(
                model_config,
                "completion_token_cost_per_1k",
                None,
            ),
            fallback_to_mock_on_error=not require_real,
        )
        client.provider_name = provider_key
    else:
        api_base = getattr(model_config, "api_base", None)
        model = getattr(model_config, "model", "")
        api_base_env_names = None
        api_key_env_names = None
        if provider_key == "deepseek":
            api_base = (
                api_base
                or os.getenv("KGQA_REAL_LLM_API_BASE")
                or DEEPSEEK_API_BASE
            )
            model = model or DEEPSEEK_DEFAULT_MODEL
            api_base_env_names = ("KGQA_REAL_LLM_API_BASE",)
            api_key_env_names = ("KGQA_REAL_LLM_API_KEY", "DEEPSEEK_API_KEY")
        client = OpenAICompatibleClient(
            api_base=api_base,
            api_key=getattr(model_config, "api_key", None),
            model=model,
            max_retries=getattr(model_config, "max_retries", 3),
            retry_delay=min(float(getattr(model_config, "max_retry_wait", 5.0)), 5.0),
            request_timeout=float(getattr(model_config, "request_timeout", 60.0)),
            default_max_tokens=getattr(model_config, "max_tokens", None),
            supports_guided_json=_supports_guided_json(provider_key, model_config),
            supports_response_format_json=_supports_response_format_json(
                provider_key,
                model_config,
            ),
            prompt_token_cost_per_1k=getattr(
                model_config,
                "prompt_token_cost_per_1k",
                None,
            ),
            completion_token_cost_per_1k=getattr(
                model_config,
                "completion_token_cost_per_1k",
                None,
            ),
            api_base_env_names=api_base_env_names,
            api_key_env_names=api_key_env_names,
        )
        client.provider_name = provider_key

    if require_real and bool(getattr(client, "mock_mode", False)):
        raise LLMProviderError(
            f"LLM provider '{provider_key}' requires a configured real client. "
            "Set the provider credentials or endpoint before using a real LLM path."
        )
    if require_real and provider_key != "glm" and not getattr(client, "api_base", None):
        raise LLMProviderError(
            f"LLM provider '{provider_key}' requires api_base or a compatible "
            "OPENAI_BASE_URL/OPENAI_API_BASE/LOCAL_LLM_API_BASE environment variable."
        )
    if require_real and provider_key in {"deepseek", "openai-compatible"} and not getattr(
        client,
        "api_key",
        None,
    ):
        raise LLMProviderError(
            f"LLM provider '{provider_key}' requires an API key. "
            "Set the provider-specific key or KGQA_REAL_LLM_API_KEY."
        )
    return client


def _normalize_provider(provider: str) -> str:
    return normalize_real_llm_provider(provider)


def _supports_guided_json(provider_key: str, model_config: Any) -> bool:
    configured = getattr(model_config, "supports_guided_json", None)
    if configured is not None:
        return bool(configured)
    return provider_key in {"local", "vllm"}


def _supports_response_format_json(provider_key: str, model_config: Any) -> bool:
    configured = getattr(model_config, "model_supports_json", None)
    if configured is not None:
        return bool(configured)
    return provider_key == "deepseek"
