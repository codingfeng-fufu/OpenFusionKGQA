"""Shared settings for opt-in real LLM gates and smoke tests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_REAL_LLM_PROVIDER = "deepseek"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
GLM_DEFAULT_MODEL = "glm-4"

SUPPORTED_REAL_LLM_PROVIDERS = frozenset(
    {"deepseek", "glm", "openai-compatible", "vllm", "local"}
)


@dataclass(frozen=True)
class RealLLMSettings:
    """Resolved real LLM smoke settings without exposing the API key in metadata."""

    provider: str
    model: str
    api_base: str
    api_key_env: str
    api_key_value: str | None = field(default=None, repr=False)
    key_required: bool = True
    api_base_required: bool = False
    supported: bool = True
    config_path: str = ""
    config_error: str = ""

    @property
    def api_key_set(self) -> bool:
        return bool(self.api_key_value)

    def metadata(self) -> dict[str, str]:
        metadata = {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "api_key": "set" if self.api_key_set else "unset",
        }
        if self.config_path:
            metadata["config_path"] = self.config_path
        return metadata

    def blocker_reason(self) -> str:
        if self.config_error:
            return self.config_error
        if not self.supported:
            return f"unsupported real LLM provider: {self.provider}"
        if not self.model:
            return "KGQA_REAL_LLM_MODEL is not set"
        if self.api_base_required and not self.api_base:
            return "KGQA_REAL_LLM_API_BASE is not set"
        if self.key_required and not self.api_key_set:
            if self.config_path:
                return "KGQA_REAL_LLM_CONFIG does not define an API key"
            key_names = provider_key_env_names(self.provider)
            if len(key_names) == 1:
                return f"{key_names[0]} is not set"
            return f"{key_names[0]} or {key_names[1]} is not set"
        return ""

    def safe_display_env(self) -> dict[str, str]:
        env = {
            "KGQA_REAL_LLM_SMOKE": "1",
            "KGQA_REAL_LLM_PROVIDER": self.provider,
            "KGQA_REAL_LLM_MODEL": self.model,
        }
        if self.api_base:
            env["KGQA_REAL_LLM_API_BASE"] = self.api_base
        if self.config_path:
            env["KGQA_REAL_LLM_CONFIG"] = self.config_path
        return env

    def runtime_env(self) -> dict[str, str]:
        if not self.api_key_value:
            return {}
        if self.provider == "glm" and self.api_key_env in {
            "KGQA_REAL_LLM_API_KEY",
            "KGQA_REAL_LLM_CONFIG",
        }:
            return {"ZHIPUAI_API_KEY": self.api_key_value}
        if self.api_key_env == "KGQA_REAL_LLM_CONFIG":
            return {"KGQA_REAL_LLM_API_KEY": self.api_key_value}
        return {}


def resolve_real_llm_settings(
    env: Mapping[str, str] | None = None,
) -> RealLLMSettings:
    source = os.environ if env is None else env
    config_path = (source.get("KGQA_REAL_LLM_CONFIG") or "").strip()
    config_values, config_error = _load_config_values(config_path)
    provider = normalize_real_llm_provider(
        source.get("KGQA_REAL_LLM_PROVIDER")
        or config_values.get("provider")
        or DEFAULT_REAL_LLM_PROVIDER
    )
    supported = provider in SUPPORTED_REAL_LLM_PROVIDERS
    model = (
        source.get("KGQA_REAL_LLM_MODEL")
        or config_values.get("model")
        or _default_model(provider)
    ).strip()
    api_base = _api_base_for_provider(
        provider,
        source,
        configured_base=config_values.get("api_base", ""),
    )
    api_key_env, api_key_value = _api_key_for_provider(
        provider,
        source,
        configured_api_key=config_values.get("api_key", ""),
        config_path=config_path,
    )
    return RealLLMSettings(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        api_key_value=api_key_value,
        key_required=_key_required(provider),
        api_base_required=_api_base_required(provider),
        supported=supported,
        config_path=config_path,
        config_error=config_error,
    )


def normalize_real_llm_provider(provider: str) -> str:
    provider_key = str(provider or "").strip().lower()
    if provider_key in {"zhipu", "zhipuai"}:
        return "glm"
    if provider_key in {"openai", "openai_compatible", "openai-compatible"}:
        return "openai-compatible"
    if provider_key in {"local", "local-openai", "local_openai", "llamacpp"}:
        return "local"
    return provider_key


def provider_key_env_names(provider: str) -> tuple[str, ...]:
    provider = normalize_real_llm_provider(provider)
    if provider == "deepseek":
        return ("DEEPSEEK_API_KEY", "KGQA_REAL_LLM_API_KEY")
    if provider == "glm":
        return ("ZHIPUAI_API_KEY", "KGQA_REAL_LLM_API_KEY")
    if provider == "openai-compatible":
        return ("OPENAI_API_KEY", "KGQA_REAL_LLM_API_KEY")
    if provider in {"local", "vllm"}:
        return ("KGQA_REAL_LLM_API_KEY", "LOCAL_LLM_API_KEY")
    return ("KGQA_REAL_LLM_API_KEY",)


def _default_model(provider: str) -> str:
    if provider == "deepseek":
        return DEEPSEEK_DEFAULT_MODEL
    if provider == "glm":
        return GLM_DEFAULT_MODEL
    return ""


def _api_base_for_provider(
    provider: str,
    env: Mapping[str, str],
    *,
    configured_base: str = "",
) -> str:
    generic_base = env.get("KGQA_REAL_LLM_API_BASE") or ""
    if provider == "deepseek":
        return generic_base or configured_base or DEEPSEEK_API_BASE
    if provider == "openai-compatible":
        return (
            generic_base
            or configured_base
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or ""
        )
    if provider in {"local", "vllm"}:
        return (
            generic_base
            or configured_base
            or env.get("LOCAL_LLM_API_BASE")
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or ""
        )
    return generic_base or configured_base


def _api_key_for_provider(
    provider: str,
    env: Mapping[str, str],
    *,
    configured_api_key: str = "",
    config_path: str = "",
) -> tuple[str, str | None]:
    provider = normalize_real_llm_provider(provider)
    generic_value = env.get("KGQA_REAL_LLM_API_KEY")
    if generic_value:
        return "KGQA_REAL_LLM_API_KEY", generic_value
    for env_name in provider_key_env_names(provider):
        value = env.get(env_name)
        if value:
            return env_name, value
    if configured_api_key:
        return "KGQA_REAL_LLM_CONFIG", configured_api_key
    if config_path:
        return "KGQA_REAL_LLM_CONFIG", None
    key_names = provider_key_env_names(provider)
    return key_names[0], None


def _key_required(provider: str) -> bool:
    return provider in {"deepseek", "glm", "openai-compatible"}


def _api_base_required(provider: str) -> bool:
    return provider in {"openai-compatible", "local", "vllm"}


def _load_config_values(config_path: str) -> tuple[dict[str, str], str]:
    if not config_path:
        return {}, ""
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}, f"KGQA_REAL_LLM_CONFIG does not exist: {config_path}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {}, f"KGQA_REAL_LLM_CONFIG is not valid YAML: {exc}"
    except OSError as exc:
        return {}, f"KGQA_REAL_LLM_CONFIG cannot be read: {exc}"
    if not isinstance(data, dict):
        return {}, "KGQA_REAL_LLM_CONFIG must contain a YAML mapping"
    return _config_values_from_mapping(data), ""


def _config_values_from_mapping(data: Mapping) -> dict[str, str]:
    extraction = data.get("extraction") if isinstance(data.get("extraction"), dict) else {}
    model_id = str(extraction.get("llm_model_id") or "default_chat_model")
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    model_config = models.get(model_id)
    if not isinstance(model_config, dict):
        model_config = models.get("default_chat_model")
    if not isinstance(model_config, dict):
        model_config = {}

    provider = (
        extraction.get("llm_provider")
        or model_config.get("model_provider")
        or ""
    )
    return {
        "provider": normalize_real_llm_provider(str(provider)),
        "model": str(model_config.get("model") or ""),
        "api_base": str(model_config.get("api_base") or ""),
        "api_key": str(model_config.get("api_key") or ""),
    }
