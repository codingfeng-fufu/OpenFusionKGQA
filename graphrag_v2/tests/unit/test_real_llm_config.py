"""Tests for real LLM smoke configuration resolution."""

from __future__ import annotations

import json
from pathlib import Path

from graphrag_v2.llm.real_llm_config import resolve_real_llm_settings


def test_real_llm_settings_loads_deepseek_key_from_local_config(tmp_path: Path):
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: deepseek-v4-flash
    model_provider: deepseek
    api_base: https://api.deepseek.com
    api_key: dummy-config-key
extraction:
  llm_provider: deepseek
  llm_model_id: default_chat_model
""",
        encoding="utf-8",
    )

    settings = resolve_real_llm_settings(
        {
            "KGQA_REAL_LLM_CONFIG": str(config_path),
        }
    )

    assert settings.provider == "deepseek"
    assert settings.model == "deepseek-v4-flash"
    assert settings.api_base == "https://api.deepseek.com"
    assert settings.api_key_value == "dummy-config-key"
    assert settings.blocker_reason() == ""
    assert settings.runtime_env() == {"KGQA_REAL_LLM_API_KEY": "dummy-config-key"}
    assert settings.metadata() == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "KGQA_REAL_LLM_CONFIG",
        "api_key": "set",
        "config_path": str(config_path),
    }
    assert "dummy-config-key" not in json.dumps(settings.metadata())


def test_real_llm_settings_uses_configured_model_id(tmp_path: Path):
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  smoke_model:
    type: chat
    model: provider-model
    model_provider: openai-compatible
    api_base: https://provider.example/v1
    api_key: dummy-provider-key
extraction:
  llm_provider: openai-compatible
  llm_model_id: smoke_model
""",
        encoding="utf-8",
    )

    settings = resolve_real_llm_settings({"KGQA_REAL_LLM_CONFIG": str(config_path)})

    assert settings.provider == "openai-compatible"
    assert settings.model == "provider-model"
    assert settings.api_base == "https://provider.example/v1"
    assert settings.api_key_value == "dummy-provider-key"
    assert settings.runtime_env() == {"KGQA_REAL_LLM_API_KEY": "dummy-provider-key"}


def test_real_llm_settings_maps_glm_config_key_to_zhipu_env(tmp_path: Path):
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: glm-4
    model_provider: zhipu
    api_key: dummy-zhipu-config-key
extraction:
  llm_provider: glm
  llm_model_id: default_chat_model
""",
        encoding="utf-8",
    )

    settings = resolve_real_llm_settings({"KGQA_REAL_LLM_CONFIG": str(config_path)})

    assert settings.provider == "glm"
    assert settings.model == "glm-4"
    assert settings.api_key_value == "dummy-zhipu-config-key"
    assert settings.runtime_env() == {"ZHIPUAI_API_KEY": "dummy-zhipu-config-key"}


def test_real_llm_settings_reports_missing_config_file(tmp_path: Path):
    config_path = tmp_path / "missing.local.real-llm.yaml"

    settings = resolve_real_llm_settings({"KGQA_REAL_LLM_CONFIG": str(config_path)})

    assert settings.metadata()["config_path"] == str(config_path)
    assert settings.metadata()["api_key"] == "unset"
    assert settings.blocker_reason() == f"KGQA_REAL_LLM_CONFIG does not exist: {config_path}"


def test_real_llm_settings_reports_config_without_api_key(tmp_path: Path):
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: deepseek-v4-flash
    model_provider: deepseek
    api_base: https://api.deepseek.com
    api_key: null
extraction:
  llm_provider: deepseek
  llm_model_id: default_chat_model
""",
        encoding="utf-8",
    )

    settings = resolve_real_llm_settings({"KGQA_REAL_LLM_CONFIG": str(config_path)})

    assert settings.metadata()["api_key_env"] == "KGQA_REAL_LLM_CONFIG"
    assert settings.metadata()["api_key"] == "unset"
    assert settings.blocker_reason() == "KGQA_REAL_LLM_CONFIG does not define an API key"
