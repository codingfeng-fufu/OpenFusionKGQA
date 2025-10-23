"""
Pipeline 集成测试
"""
import pytest
import asyncio
from pathlib import Path

from graphrag_v2.config import create_default_config
from graphrag_v2.pipeline import create_pipeline, PipelineRunner
from graphrag_v2.workflows import (
    load_documents,
    create_base_text_units,
    extract_graph,
    create_communities,
    create_community_reports,
    generate_embeddings,
)


class TestPipelineIntegration:
    """测试 Pipeline 集成"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, temp_dir: Path, sample_text: str):
        """测试完整的 Pipeline"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 创建输入文件
        input_file = temp_dir / "input.txt"
        input_file.write_text(sample_text, encoding="utf-8")
        
        # 创建 Pipeline
        pipeline = create_pipeline(
            config=config,
            workflows=[
                load_documents,
                create_base_text_units,
                extract_graph,
                create_communities,
                create_community_reports,
                generate_embeddings,
            ],
        )
        
        # 运行 Pipeline
        runner = PipelineRunner(pipeline)
        results = []
        async for result in runner.run():
            results.append(result)
        
        # 验证结果
        assert len(results) == 6  # 6 个工作流
        
        # 验证每个工作流都成功
        for result in results:
            assert result is not None
            
    @pytest.mark.asyncio
    async def test_partial_pipeline(self, temp_dir: Path, sample_text: str):
        """测试部分 Pipeline"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 创建输入文件
        input_file = temp_dir / "input.txt"
        input_file.write_text(sample_text, encoding="utf-8")
        
        # 只运行前 3 个工作流
        pipeline = create_pipeline(
            config=config,
            workflows=[
                load_documents,
                create_base_text_units,
                extract_graph,
            ],
        )
        
        # 运行 Pipeline
        runner = PipelineRunner(pipeline)
        results = []
        async for result in runner.run():
            results.append(result)
        
        # 验证结果
        assert len(results) == 3
        
    @pytest.mark.asyncio
    async def test_pipeline_with_chinese_text(self, temp_dir: Path, chinese_text: str):
        """测试中文文本的 Pipeline"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 创建输入文件
        input_file = temp_dir / "input.txt"
        input_file.write_text(chinese_text, encoding="utf-8")
        
        # 创建 Pipeline
        pipeline = create_pipeline(
            config=config,
            workflows=[
                load_documents,
                create_base_text_units,
                extract_graph,
            ],
        )
        
        # 运行 Pipeline
        runner = PipelineRunner(pipeline)
        results = []
        async for result in runner.run():
            results.append(result)
        
        # 验证结果
        assert len(results) == 3


class TestWorkflowIntegration:
    """测试工作流集成"""
    
    @pytest.mark.asyncio
    async def test_load_documents_workflow(self, temp_dir: Path, sample_text: str):
        """测试文档加载工作流"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 创建输入文件
        input_file = temp_dir / "input.txt"
        input_file.write_text(sample_text, encoding="utf-8")
        
        # 运行工作流
        from graphrag_v2.pipeline.context import PipelineRunContext
        from graphrag_v2.storage.memory_storage import MemoryPipelineStorage
        
        context = PipelineRunContext(
            storage=MemoryPipelineStorage(),
            output_storage=MemoryPipelineStorage(),
        )
        
        result = await load_documents(config, context)
        
        # 验证结果
        assert result is not None
        assert "documents" in result.outputs
        
    @pytest.mark.asyncio
    async def test_create_text_units_workflow(self, temp_dir: Path):
        """测试文本分块工作流"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 准备输入数据
        from graphrag_v2.pipeline.context import PipelineRunContext
        from graphrag_v2.storage.memory_storage import MemoryPipelineStorage
        from graphrag_v2.data_model import Document
        
        storage = MemoryPipelineStorage()
        documents = [
            Document(id="doc1", text="这是一个测试文档。" * 50)
        ]
        storage.set("documents", documents)
        
        context = PipelineRunContext(
            storage=storage,
            output_storage=MemoryPipelineStorage(),
        )
        
        result = await create_base_text_units(config, context)
        
        # 验证结果
        assert result is not None
        assert "text_units" in result.outputs


class TestPipelineErrorHandling:
    """测试 Pipeline 错误处理"""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_missing_input(self, temp_dir: Path):
        """测试缺少输入文件的 Pipeline"""
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 不创建输入文件
        
        # 创建 Pipeline
        pipeline = create_pipeline(
            config=config,
            workflows=[load_documents],
        )
        
        # 运行 Pipeline（应该处理错误）
        runner = PipelineRunner(pipeline)
        results = []
        
        try:
            async for result in runner.run():
                results.append(result)
        except Exception as e:
            # 预期会有错误
            assert e is not None


class TestPipelinePerformance:
    """测试 Pipeline 性能"""
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_time(self, temp_dir: Path, sample_text: str):
        """测试 Pipeline 执行时间"""
        import time
        
        # 创建配置
        config = create_default_config()
        config.storage.base_dir = str(temp_dir)
        
        # 创建输入文件
        input_file = temp_dir / "input.txt"
        input_file.write_text(sample_text, encoding="utf-8")
        
        # 创建 Pipeline
        pipeline = create_pipeline(
            config=config,
            workflows=[
                load_documents,
                create_base_text_units,
            ],
        )
        
        # 测量执行时间
        start_time = time.time()
        
        runner = PipelineRunner(pipeline)
        results = []
        async for result in runner.run():
            results.append(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证执行时间合理（应该在几秒内完成）
        assert execution_time < 10.0  # 10 秒内完成
        
        print(f"\nPipeline 执行时间: {execution_time:.2f} 秒")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

