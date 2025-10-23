"""测试配置系统。

验证配置加载、验证和序列化功能。
"""

import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag_v2.config.enums import AuthType, ModelType
from graphrag_v2.config.loader import create_default_config, load_config
from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.config.models.language_model_config import LanguageModelConfig


def test_create_default_config():
    """测试创建默认配置。"""
    print("=" * 60)
    print("测试 1: 创建默认配置")
    print("=" * 60)
    
    config = create_default_config()
    print(f"\n✓ 成功创建默认配置")
    print(f"  - 根目录: {config.root_dir}")
    print(f"  - 模型数量: {len(config.models)}")
    print(f"  - 输入目录: {config.input.base_dir}")
    print(f"  - 输出目录: {config.output.base_dir}")
    print(f"  - 缓存目录: {config.cache.base_dir}")
    print(f"  - 分块大小: {config.chunks.size}")
    
    return config


def test_model_config_validation():
    """测试模型配置验证。"""
    print("\n" + "=" * 60)
    print("测试 2: 模型配置验证")
    print("=" * 60)
    
    # 测试有效配置
    try:
        config = LanguageModelConfig(
            type=ModelType.Chat,
            model="gpt-4o-mini",
            model_provider="openai",
            auth_type=AuthType.APIKey,
            api_key="test-key",
        )
        print(f"\n✓ 有效配置通过验证")
        print(f"  - 模型类型: {config.type}")
        print(f"  - 模型名称: {config.model}")
        print(f"  - 认证类型: {config.auth_type}")
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
    
    # 测试缺少 API 密钥
    try:
        config = LanguageModelConfig(
            type=ModelType.Chat,
            model="gpt-4o-mini",
            model_provider="openai",
            auth_type=AuthType.APIKey,
            api_key="",  # 空 API 密钥
        )
        print(f"\n✗ 应该抛出异常但没有")
    except ValueError as e:
        print(f"\n✓ 正确捕获到缺少 API 密钥的错误")
        print(f"  - 错误信息: {e}")


def test_config_serialization():
    """测试配置序列化。"""
    print("\n" + "=" * 60)
    print("测试 3: 配置序列化")
    print("=" * 60)
    
    config = create_default_config()
    
    # 测试 JSON 序列化
    json_str = config.model_dump_json(indent=2)
    print(f"\n✓ JSON 序列化成功")
    print(f"  - JSON 长度: {len(json_str)} 字符")
    
    # 测试字典序列化
    config_dict = config.model_dump()
    print(f"\n✓ 字典序列化成功")
    print(f"  - 字典键数量: {len(config_dict)}")
    
    return config_dict


def test_save_and_load_config():
    """测试保存和加载配置。"""
    print("\n" + "=" * 60)
    print("测试 4: 保存和加载配置")
    print("=" * 60)
    
    # 创建临时配置文件
    temp_config_path = Path("temp_settings.yaml")
    
    try:
        # 保存配置
        config = create_default_config(output_path=temp_config_path)
        print(f"\n✓ 配置已保存到: {temp_config_path}")
        
        # 加载配置
        loaded_config = load_config(temp_config_path)
        print(f"✓ 配置已从文件加载")
        
        # 验证配置一致性
        assert loaded_config.root_dir == config.root_dir
        assert len(loaded_config.models) == len(config.models)
        print(f"✓ 加载的配置与原配置一致")
        
    finally:
        # 清理临时文件
        if temp_config_path.exists():
            temp_config_path.unlink()
            print(f"✓ 临时文件已清理")


def test_get_model_config():
    """测试获取模型配置。"""
    print("\n" + "=" * 60)
    print("测试 5: 获取模型配置")
    print("=" * 60)
    
    config = create_default_config()
    
    # 获取默认聊天模型
    chat_model = config.get_language_model_config("default_chat_model")
    print(f"\n✓ 成功获取默认聊天模型配置")
    print(f"  - 模型: {chat_model.model}")
    print(f"  - 类型: {chat_model.type}")
    
    # 获取默认嵌入模型
    embedding_model = config.get_language_model_config("default_embedding_model")
    print(f"\n✓ 成功获取默认嵌入模型配置")
    print(f"  - 模型: {embedding_model.model}")
    print(f"  - 类型: {embedding_model.type}")
    
    # 尝试获取不存在的模型
    try:
        config.get_language_model_config("non_existent_model")
        print(f"\n✗ 应该抛出异常但没有")
    except ValueError as e:
        print(f"\n✓ 正确捕获到模型不存在的错误")
        print(f"  - 错误信息: {e}")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("GraphRAG v2 配置系统测试")
    print("=" * 60)
    
    try:
        test_create_default_config()
        test_model_config_validation()
        test_config_serialization()
        test_save_and_load_config()
        test_get_model_config()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

