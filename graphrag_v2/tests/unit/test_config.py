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
"""
        config_file.write_text(config_content, encoding="utf-8")
        
        # 加载配置
        config = load_config(str(config_file))
        
        assert config.root_dir == "./test_data"
        assert config.storage.base_dir == "./test_output"
        assert config.chunks.size == 500
        assert config.chunks.overlap == 50
        assert "test_model" in config.llm.models
        
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
        
    def test_default_entity_extraction_max_gleanings(self):
        """测试默认实体提取最大迭代次数"""
        config = create_default_config()
        assert config.entity_extraction.max_gleanings == 1


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

