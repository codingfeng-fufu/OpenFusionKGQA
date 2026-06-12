"""
配置模块单元测试
"""
import pytest
from pathlib import Path

from graphrag_v2.config import (
    GraphRagConfig,
    create_default_config,
    load_config,
    StorageType,
    ChunkStrategyType,
)
from graphrag_v2.config.models.extraction_config import ExtractionConfig
from graphrag_v2.config.models.input_config import InputConfig


class TestGraphRagConfig:
    """测试 GraphRagConfig 类"""
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        config = create_default_config()
        
        assert config is not None
        assert isinstance(config, GraphRagConfig)
        assert config.root_dir is not None
        assert config.storage is not None
        assert config.llm is not None
        assert config.embeddings is not None
        assert config.chunks is not None
        assert config.entity_extraction is not None
        
    def test_config_validation(self):
        """测试配置验证"""
        config = create_default_config()
        
        # 验证 LLM 配置
        assert len(config.llm.models) > 0
        assert "default_chat_model" in config.llm.models
        
        # 验证嵌入配置
        assert len(config.embeddings.models) > 0
        assert "default_embedding_model" in config.embeddings.models
        
        # 验证分块配置
        assert config.chunks.size > 0
        assert config.chunks.overlap >= 0
        assert config.chunks.overlap < config.chunks.size
        
    def test_get_language_model_config(self):
        """测试获取语言模型配置"""
        config = create_default_config()
        
        # 获取默认聊天模型
        chat_model = config.get_language_model_config("default_chat_model")
        assert chat_model is not None
        assert chat_model.model is not None
        
    def test_get_embedding_model_config(self):
        """测试获取嵌入模型配置"""
        config = create_default_config()
        
        # 获取默认嵌入模型
        embedding_model = config.get_embedding_model_config("default_embedding_model")
        assert embedding_model is not None
        assert embedding_model.model is not None
        
    def test_load_config_from_yaml(self, temp_dir: Path):
        """测试从 YAML 文件加载配置"""
        # 创建测试配置文件
        config_file = temp_dir / "test_config.yaml"
        config_content = """
root_dir: "./test_data"

storage:
  type: "file"
  base_dir: "./test_output"

llm:
  models:
    test_model:
      type: "openai_chat"
      model: "gpt-4"
      api_key: "test-key"

chunks:
  size: 500
  overlap: 50

graph_store:
  provider: "json"
  fallback: "json"
  transaction_timeout_seconds: 12.5
  max_transaction_retries: 3
  transaction_retry_backoff_seconds: 0.25

community:
  enabled: true
  algorithm: "louvain"
  min_community_size: 2

fusion:
  relation_schema_mode: "closed"
"""
        config_file.write_text(config_content, encoding="utf-8")
        
        # 加载配置
        config = load_config(str(config_file))
        
        assert config.root_dir == str(Path("./test_data").resolve())
        assert config.storage.base_dir == str(
            (Path(config.root_dir) / "./test_output").resolve()
        )
        assert config.chunks.size == 500
        assert config.chunks.overlap == 50
        assert config.graph_store.provider == "json"
        assert config.graph_store.fallback == "json"
        assert config.graph_store.transaction_timeout_seconds == 12.5
        assert config.graph_store.max_transaction_retries == 3
        assert config.graph_store.transaction_retry_backoff_seconds == 0.25
        assert config.community.enabled is True
        assert config.community.algorithm == "louvain"
        assert config.community.min_community_size == 2
        assert config.fusion.relation_schema_mode == "closed"
        assert "test_model" in config.llm.models

    def test_neo4j_environment_overrides_graph_store(self, temp_dir: Path, monkeypatch):
        """测试 Neo4j 环境变量覆盖图存储连接配置。"""
        config_file = temp_dir / "neo4j_config.yaml"
        config_file.write_text(
            """
graph_store:
  provider: "neo4j"
  uri: "bolt://127.0.0.1:7690"
  username: "neo4j"
  password_env: "NEO4J_PASSWORD"
  database: "neo4j"
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:7694")
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j_test")
        monkeypatch.setenv("NEO4J_DATABASE", "neo4j_test_db")
        monkeypatch.setenv("NEO4J_PASSWORD_ENV", "KGQA_TEST_NEO4J_PASSWORD")

        config = load_config(config_file)

        assert config.graph_store.uri == "bolt://127.0.0.1:7694"
        assert config.graph_store.username == "neo4j_test"
        assert config.graph_store.database == "neo4j_test_db"
        assert config.graph_store.password_env == "KGQA_TEST_NEO4J_PASSWORD"
        
    def test_config_serialization(self):
        """测试配置序列化"""
        config = create_default_config()
        
        # 转换为字典
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert "root_dir" in config_dict
        assert "storage" in config_dict
        assert "llm" in config_dict
        
        # 从字典重建
        new_config = GraphRagConfig(**config_dict)
        assert new_config.root_dir == config.root_dir


class TestConfigEnums:
    """测试配置枚举类"""

    def test_storage_type_enum(self):
        """测试存储类型枚举"""
        assert StorageType.file == "file"
        assert StorageType.memory == "memory"
        assert StorageType.blob == "blob"

    def test_chunk_strategy_enum(self):
        """测试分块策略枚举"""
        assert ChunkStrategyType.tokens == "tokens"
        assert ChunkStrategyType.sentence == "sentence"


class TestConfigDefaults:
    """测试默认配置值"""
    
    def test_default_chunk_size(self):
        """测试默认分块大小"""
        config = create_default_config()
        assert config.chunks.size == 300
        
    def test_default_chunk_overlap(self):
        """测试默认分块重叠"""
        config = create_default_config()
        assert config.chunks.overlap == 100
        
    def test_default_storage_type(self):
        """测试默认存储类型"""
        config = create_default_config()
        assert config.storage.type == "file"

    def test_default_graph_store(self):
        """测试默认图存储配置"""
        config = create_default_config()
        assert config.graph_store.provider == "json"
        assert config.graph_store.uri == "bolt://localhost:7687"
        assert config.graph_store.password_env == "NEO4J_PASSWORD"
        assert config.graph_store.batch_size == 500
        assert config.graph_store.replace_index_on_write is True
        assert config.graph_store.staged_replace_on_write is True
        assert config.graph_store.connection_timeout_seconds == 10.0
        assert config.graph_store.transaction_timeout_seconds == 30.0
        assert config.graph_store.max_transaction_retries == 2
        assert config.graph_store.transaction_retry_backoff_seconds == 0.5

    def test_default_community(self):
        """测试默认社区配置"""
        config = create_default_config()
        assert config.community.enabled is False
        assert config.community.algorithm == "louvain"
        assert config.community.min_community_size == 2
        assert config.community.reporter == "mock"

    def test_default_extraction(self):
        """测试默认知识抽取配置"""
        config = create_default_config()
        assert config.extraction.extractor_provider == "mock"
        assert config.extraction.llm_model_id == "default_chat_model"
        assert config.extraction.llm_provider == "deepseek"
        assert config.extraction.max_retries == 2
        assert config.extraction.max_gleanings == 1
        assert config.extraction.fail_on_invalid_chunk is True
        assert config.extraction.default_confidence == 0.7
        assert config.extraction.requests_per_minute is None
        assert config.extraction.concurrent_requests == 1
        assert config.extraction.max_prompt_tokens_per_chunk is None
        assert config.extraction.max_total_tokens is None
        assert config.extraction.max_estimated_cost is None
        assert config.extraction.salvage_on_parse_failure is True
        assert config.extraction.cache_enabled is False
        assert config.extraction.cache_dir is None

    def test_default_fusion(self):
        """测试默认图融合配置"""
        config = create_default_config()
        assert config.fusion.relation_schema_mode == "open"

    def test_deepseek_extraction_provider_is_supported(self):
        """测试 DeepSeek 可以作为真实 LLM 抽取 provider。"""
        from graphrag_v2.config.models.extraction_config import ExtractionConfig

        config = ExtractionConfig(llm_provider="deepseek")

        assert config.llm_provider == "deepseek"

    def test_default_input_ingestion_limits(self):
        """测试默认输入摄入策略。"""
        config = create_default_config()
        assert config.input.unsupported_file_policy == "ignore"
        assert config.input.max_file_size_bytes is None
        assert config.input.max_document_count is None
        
    def test_default_entity_extraction_max_gleanings(self):
        """测试默认实体提取最大迭代次数"""
        config = create_default_config()
        assert config.entity_extraction.max_gleanings == 1

    def test_entity_extraction_alias_tracks_extraction_config(self):
        """测试旧实体提取别名跟随新的 extraction 配置。"""
        config = GraphRagConfig(extraction=ExtractionConfig(max_gleanings=3))

        assert config.entity_extraction.max_gleanings == 3
        assert config.model_dump()["entity_extraction"]["max_gleanings"] == 3

    def test_load_extraction_config_from_yaml(self, temp_dir: Path):
        """测试从 YAML 文件加载知识抽取配置"""
        config_file = temp_dir / "test_extraction_config.yaml"
        config_file.write_text(
            """
root_dir: "."
extraction:
  extractor_provider: "llm"
  llm_model_id: "default_chat_model"
  llm_provider: "glm"
  max_retries: 3
  fail_on_invalid_chunk: false
  default_confidence: 0.61
  requests_per_minute: 5
  concurrent_requests: 2
  max_prompt_tokens_per_chunk: 1000
  max_total_tokens: 2000
  max_estimated_cost: 0.5
  salvage_on_parse_failure: false
  cache_enabled: true
  cache_dir: "./cache/extraction"
input:
  unsupported_file_policy: "fail"
  max_file_size_bytes: 1024
  max_document_count: 3
models:
  default_chat_model:
    type: chat
    model: glm-4
    prompt_token_cost_per_1k: 0.1
    completion_token_cost_per_1k: 0.2
  default_embedding_model:
    type: embedding
    model: text-embedding-3-small
""",
            encoding="utf-8",
        )

        config = load_config(str(config_file))

        assert config.extraction.extractor_provider == "llm"
        assert config.extraction.llm_provider == "glm"
        assert config.extraction.max_retries == 3
        assert config.extraction.fail_on_invalid_chunk is False
        assert config.extraction.default_confidence == 0.61
        assert config.extraction.requests_per_minute == 5
        assert config.extraction.concurrent_requests == 2
        assert config.extraction.max_prompt_tokens_per_chunk == 1000
        assert config.extraction.max_total_tokens == 2000
        assert config.extraction.max_estimated_cost == 0.5
        assert config.extraction.salvage_on_parse_failure is False
        assert config.extraction.cache_enabled is True
        assert config.extraction.cache_dir == "./cache/extraction"
        assert config.input.unsupported_file_policy == "fail"
        assert config.input.max_file_size_bytes == 1024
        assert config.input.max_document_count == 3
        model_config = config.get_language_model_config("default_chat_model")
        assert model_config.prompt_token_cost_per_1k == 0.1
        assert model_config.completion_token_cost_per_1k == 0.2

    def test_load_legacy_entity_extraction_max_gleanings(self, temp_dir: Path):
        """测试旧 entity_extraction 配置会迁移到 extraction。"""
        config_file = temp_dir / "test_legacy_extraction_config.yaml"
        config_file.write_text(
            """
root_dir: "."
entity_extraction:
  max_gleanings: 2
""",
            encoding="utf-8",
        )

        config = load_config(str(config_file))

        assert config.extraction.max_gleanings == 2
        assert config.entity_extraction.max_gleanings == 2


class TestConfigModification:
    """测试配置修改"""
    
    def test_modify_chunk_size(self):
        """测试修改分块大小"""
        config = create_default_config()
        original_size = config.chunks.size
        
        config.chunks.size = 500
        assert config.chunks.size == 500
        assert config.chunks.size != original_size
        
    def test_modify_storage_base_dir(self):
        """测试修改存储基础目录"""
        config = create_default_config()
        
        config.storage.base_dir = "/new/path"
        assert config.storage.base_dir == "/new/path"
        
    def test_add_new_llm_model(self):
        """测试添加新的 LLM 模型"""
        config = create_default_config()
        
        from graphrag_v2.config.models.llm_config import LanguageModelConfig
        
        new_model = LanguageModelConfig(
            type="openai_chat",
            model="gpt-4-turbo",
            api_key="test-key",
        )
        
        config.llm.models["new_model"] = new_model
        assert "new_model" in config.llm.models
        assert config.llm.models["new_model"].model == "gpt-4-turbo"


class TestConfigValidation:
    """测试配置验证"""
    
    def test_invalid_chunk_overlap(self):
        """测试无效的分块重叠"""
        config = create_default_config()
        
        # 重叠不能大于等于大小
        with pytest.raises(ValueError):
            config.chunks.overlap = config.chunks.size
            config.chunks.validate_overlap()
            
    def test_empty_model_name(self):
        """测试空模型名称"""
        from graphrag_v2.config.models.llm_config import LanguageModelConfig
        
        with pytest.raises(ValueError):
            LanguageModelConfig(
                type="openai_chat",
                model="",  # 空模型名称
            )

    def test_invalid_extraction_provider_budget(self):
        """测试无效 extraction provider 和预算配置。"""
        with pytest.raises(ValueError, match="llm_provider"):
            ExtractionConfig(llm_provider="unknown-provider")
        assert ExtractionConfig(llm_provider="openai").llm_provider == "openai-compatible"
        assert ExtractionConfig(llm_provider="vllm").llm_provider == "vllm"
        with pytest.raises(ValueError, match="requests_per_minute"):
            ExtractionConfig(requests_per_minute=0)
        with pytest.raises(ValueError, match="concurrent_requests"):
            ExtractionConfig(concurrent_requests=0)
        with pytest.raises(ValueError, match="max_estimated_cost"):
            ExtractionConfig(max_estimated_cost=-0.1)
        with pytest.raises(ValueError, match="cache_dir"):
            ExtractionConfig(cache_dir=" ")
        with pytest.raises(ValueError, match="max_file_size_bytes"):
            InputConfig(max_file_size_bytes=0)
        with pytest.raises(ValueError, match="max_document_count"):
            InputConfig(max_document_count=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
