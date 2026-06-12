"""配置加载器。

提供从 YAML/JSON 文件加载配置的功能，支持环境变量覆盖。
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from graphrag_v2.config.defaults import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODEL_ID,
    DEFAULT_CHAT_MODEL_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_TYPE,
    DEFAULT_MODEL_PROVIDER,
)
from graphrag_v2.config.enums import AuthType
from graphrag_v2.config.models.graph_rag_config import GraphRagConfig


def load_config(
    config_path: str | Path,
    env_file: str | Path | None = None,
) -> GraphRagConfig:
    """从配置文件加载 GraphRAG 配置。
    
    Args:
        config_path: 配置文件路径（YAML 或 JSON）
        env_file: 环境变量文件路径（可选）
        
    Returns:
        GraphRagConfig 实例
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        ValueError: 如果配置文件格式错误
    """
    # 加载环境变量
    if env_file:
        load_dotenv(env_file)
    else:
        # 尝试加载默认的 .env 文件
        load_dotenv()
    
    # 读取配置文件
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 解析配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            import json
            config_dict = json.load(f)
        else:
            raise ValueError(
                f"不支持的配置文件格式: {config_path.suffix}。"
                "请使用 .yaml, .yml 或 .json 文件。"
            )
    
    # 应用环境变量覆盖
    config_dict = _apply_env_overrides(config_dict)
    
    # 创建配置对象
    try:
        config = GraphRagConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"配置文件格式错误: {e}") from e
    
    return config


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """应用环境变量覆盖。
    
    支持的环境变量格式:
    - GRAPHRAG_API_KEY: 覆盖默认聊天模型的 API 密钥
    - GRAPHRAG_EMBEDDING_API_KEY: 覆盖默认嵌入模型的 API 密钥
    - GRAPHRAG_ROOT_DIR: 覆盖根目录
    
    Args:
        config_dict: 配置字典
        
    Returns:
        应用环境变量后的配置字典
    """
    # 兼容旧配置键名
    if "storage" in config_dict and "output" not in config_dict:
        config_dict["output"] = config_dict["storage"]

    if "llm" in config_dict and "models" not in config_dict:
        config_dict["models"] = {
            DEFAULT_CHAT_MODEL_ID: {
                "type": DEFAULT_CHAT_MODEL_TYPE,
                "model": DEFAULT_CHAT_MODEL,
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "auth_type": AuthType.APIKey,
            },
            DEFAULT_EMBEDDING_MODEL_ID: {
                "type": DEFAULT_EMBEDDING_MODEL_TYPE,
                "model": DEFAULT_EMBEDDING_MODEL,
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "auth_type": AuthType.APIKey,
            },
            **config_dict["llm"].get("models", {}),
        }

    if "embeddings" in config_dict and "models" not in config_dict:
        config_dict["models"] = config_dict["embeddings"].get("models", {})

    entity_extraction = config_dict.get("entity_extraction")
    if isinstance(entity_extraction, dict) and "max_gleanings" in entity_extraction:
        config_dict.setdefault("extraction", {})
        config_dict["extraction"].setdefault(
            "max_gleanings",
            entity_extraction["max_gleanings"],
        )

    # 覆盖根目录
    if "GRAPHRAG_ROOT_DIR" in os.environ:
        config_dict["root_dir"] = os.environ["GRAPHRAG_ROOT_DIR"]
    
    # 覆盖 API 密钥
    if "GRAPHRAG_API_KEY" in os.environ:
        if "models" not in config_dict:
            config_dict["models"] = {}
        
        # 覆盖默认聊天模型的 API 密钥
        if "default_chat_model" in config_dict["models"]:
            config_dict["models"]["default_chat_model"]["api_key"] = os.environ[
                "GRAPHRAG_API_KEY"
            ]
    
    # 覆盖嵌入模型 API 密钥
    if "GRAPHRAG_EMBEDDING_API_KEY" in os.environ:
        if "models" not in config_dict:
            config_dict["models"] = {}
        
        # 覆盖默认嵌入模型的 API 密钥
        if "default_embedding_model" in config_dict["models"]:
            config_dict["models"]["default_embedding_model"]["api_key"] = os.environ[
                "GRAPHRAG_EMBEDDING_API_KEY"
            ]

    graph_store_env = {
        "NEO4J_URI": "uri",
        "NEO4J_USERNAME": "username",
        "NEO4J_DATABASE": "database",
        "NEO4J_PASSWORD_ENV": "password_env",
    }
    if any(name in os.environ for name in graph_store_env):
        config_dict.setdefault("graph_store", {})
        for env_name, field_name in graph_store_env.items():
            value = os.environ.get(env_name)
            if value:
                config_dict["graph_store"][field_name] = value
    
    return config_dict


def create_default_config(output_path: str | Path | None = None) -> GraphRagConfig:
    """创建默认配置。
    
    Args:
        output_path: 输出配置文件路径（可选）
        
    Returns:
        默认的 GraphRagConfig 实例
    """
    config = GraphRagConfig()
    
    # 如果指定了输出路径，保存配置文件
    if output_path:
        output_path = Path(output_path)
        config_dict = config.model_dump(mode="json", exclude_none=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if output_path.suffix in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif output_path.suffix == ".json":
                import json
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(
                    f"不支持的配置文件格式: {output_path.suffix}。"
                    "请使用 .yaml, .yml 或 .json 文件。"
                )
    
    return config
