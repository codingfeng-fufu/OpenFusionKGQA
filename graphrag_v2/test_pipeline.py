"""测试 Pipeline 功能。

验证 Pipeline 的创建和运行。
"""

import asyncio
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag_v2.config.enums import IndexingMethod
from graphrag_v2.config.loader import create_default_config
from graphrag_v2.pipeline.context import PipelineStorage
from graphrag_v2.pipeline.factory import PipelineFactory
from graphrag_v2.pipeline.runner import create_run_context, run_pipeline


def print_section(title: str) -> None:
    """打印分节标题。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


async def test_pipeline_creation():
    """测试 Pipeline 创建。"""
    print_section("测试 1: Pipeline 创建")
    
    try:
        # 创建默认配置
        config = create_default_config()
        
        # 创建 Pipeline
        pipeline = PipelineFactory.create_pipeline(config, IndexingMethod.Standard)
        
        print(f"\n✓ 成功创建 Pipeline")
        print(f"  - 工作流数量: {len(pipeline)}")
        print(f"  - 工作流列表: {pipeline.names()}")
        print(f"  - Pipeline 表示: {pipeline}")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_pipeline_run():
    """测试 Pipeline 运行。"""
    print_section("测试 2: Pipeline 运行")
    
    try:
        # 创建默认配置
        config = create_default_config()
        
        # 创建测试输入目录和文件
        test_dir = Path("test_input")
        test_dir.mkdir(exist_ok=True)
        
        # 创建测试文档
        test_file = test_dir / "test_doc.txt"
        test_content = """这是一个测试文档。
它包含多行文本。
用于测试文本分块功能。
GraphRAG 是一个基于知识图谱的 RAG 系统。
它可以从文本中提取实体和关系。
然后使用社区检测算法发现实体社区。
最后生成社区摘要报告。
这些报告可以用于回答高层次的问题。
"""
        test_file.write_text(test_content, encoding="utf-8")
        
        print(f"\n✓ 创建测试文档: {test_file}")
        
        # 更新配置
        config.input.base_dir = str(test_dir)
        config.input.file_type = "txt"
        config.chunks.size = 100  # 小块用于测试
        config.chunks.overlap = 20
        
        # 创建 Pipeline
        pipeline = PipelineFactory.create_pipeline(config, IndexingMethod.Standard)
        
        # 创建存储
        input_storage = PipelineStorage(base_dir=str(test_dir))
        output_storage = PipelineStorage(base_dir="test_output")
        
        # 创建运行上下文
        context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
        )
        
        print(f"\n✓ 创建运行上下文")
        
        # 运行 Pipeline
        print(f"\n开始运行 Pipeline...")
        
        async for result in run_pipeline(pipeline, config, context):
            print(f"\n工作流: {result.workflow_name}")
            print(f"  - 运行时间: {result.runtime:.2f}秒")
            if result.errors:
                print(f"  - 错误: {result.errors}")
            else:
                print(f"  - 状态: 成功")
                if result.result is not None:
                    print(f"  - 结果类型: {type(result.result).__name__}")
                    if hasattr(result.result, '__len__'):
                        print(f"  - 结果数量: {len(result.result)}")
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"  - 总运行时间: {context.stats.total_runtime:.2f}秒")
        print(f"  - 文档数量: {context.stats.num_documents}")
        print(f"  - 文本单元数量: {context.stats.num_text_units}")
        
        # 检查输出
        documents = await output_storage.get("documents")
        text_units = await output_storage.get("text_units")
        
        if documents is not None:
            print(f"\n✓ 文档数据:")
            print(f"  - 行数: {len(documents)}")
            print(f"  - 列: {list(documents.columns)}")
            print(f"\n前几行:")
            print(documents[["id", "title"]].head())
        
        if text_units is not None:
            print(f"\n✓ 文本单元数据:")
            print(f"  - 行数: {len(text_units)}")
            print(f"  - 列: {list(text_units.columns)}")
            print(f"\n前几行:")
            print(text_units[["id", "n_tokens", "chunk_index"]].head())
            print(f"\n第一个文本单元的内容:")
            print(text_units.iloc[0]["text"][:200] + "...")
        
        print(f"\n✓ Pipeline 运行成功")
        
        # 清理测试文件
        test_file.unlink()
        test_dir.rmdir()
        print(f"\n✓ 清理测试文件")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_pipeline_operations():
    """测试 Pipeline 操作。"""
    print_section("测试 3: Pipeline 操作")
    
    try:
        config = create_default_config()
        pipeline = PipelineFactory.create_pipeline(config, IndexingMethod.Standard)
        
        # 测试 remove
        original_count = len(pipeline)
        pipeline.remove("load_documents")
        print(f"\n✓ 移除工作流后: {original_count} -> {len(pipeline)}")
        
        # 测试 add
        from graphrag_v2.workflows import load_documents
        pipeline.add(("load_documents", load_documents))
        print(f"✓ 添加工作流后: {len(pipeline)}")
        
        # 测试 insert
        pipeline.insert(0, ("test_workflow", load_documents))
        print(f"✓ 插入工作流后: {len(pipeline)}")
        print(f"  - 工作流列表: {pipeline.names()}")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


async def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("GraphRAG v2 Pipeline 测试")
    print("=" * 60)
    
    await test_pipeline_creation()
    await test_pipeline_run()
    await test_pipeline_operations()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())

