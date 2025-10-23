"""
查询引擎集成测试
"""
import pytest
import asyncio

from graphrag_v2.query import GlobalSearch, LocalSearch
from graphrag_v2.query.global_context_builder import CommunityContextBuilder
from graphrag_v2.query.local_context_builder import EntityRelationshipContextBuilder
from graphrag_v2.data_model import CommunityReport, Entity, Relationship, Community
from graphrag_v2.llm import GLMClient


class TestGlobalSearchIntegration:
    """测试 Global Search 集成"""
    
    @pytest.mark.asyncio
    async def test_global_search_basic(self):
        """测试基础 Global Search"""
        # 准备数据
        reports = [
            CommunityReport(
                id="cr1",
                community_id="c1",
                title="GraphRAG 技术社区",
                summary="GraphRAG 是一种结合知识图谱和 RAG 的技术",
                full_content="详细内容...",
                rank=8.5,
            ),
            CommunityReport(
                id="cr2",
                community_id="c2",
                title="社区检测算法",
                summary="Leiden 算法用于检测社区结构",
                full_content="详细内容...",
                rank=7.5,
            ),
        ]
        
        # 创建上下文构建器
        context_builder = CommunityContextBuilder(
            community_reports=reports,
            max_tokens=2000,
        )
        
        # 创建 LLM 客户端
        llm_client = GLMClient()
        
        # 创建搜索引擎
        search = GlobalSearch(
            llm_client=llm_client,
            context_builder=context_builder,
        )
        
        # 执行搜索
        result = await search.search("GraphRAG 的主要特点是什么？")
        
        # 验证结果
        assert result is not None
        assert result.response is not None
        assert len(result.response) > 0
        assert result.context_data is not None
        
    @pytest.mark.asyncio
    async def test_global_search_with_multiple_reports(self):
        """测试多个报告的 Global Search"""
        # 创建更多报告
        reports = [
            CommunityReport(
                id=f"cr{i}",
                community_id=f"c{i}",
                title=f"社区 {i}",
                summary=f"社区 {i} 的摘要",
                full_content=f"社区 {i} 的详细内容",
                rank=8.0 - i * 0.5,
            )
            for i in range(5)
        ]
        
        context_builder = CommunityContextBuilder(
            community_reports=reports,
            max_tokens=3000,
        )
        
        llm_client = GLMClient()
        search = GlobalSearch(
            llm_client=llm_client,
            context_builder=context_builder,
        )
        
        result = await search.search("总结所有社区的主要内容")
        
        assert result is not None
        assert result.response is not None


class TestLocalSearchIntegration:
    """测试 Local Search 集成"""
    
    @pytest.mark.asyncio
    async def test_local_search_basic(self, sample_entities, sample_relationships):
        """测试基础 Local Search"""
        # 创建社区
        communities = [
            Community(
                id="c1",
                title="GraphRAG 技术生态",
                level=0,
                entities=["GraphRAG", "微软"],
                relationships=["r1"],
            ),
        ]
        
        # 创建社区报告
        reports = [
            CommunityReport(
                id="cr1",
                community_id="c1",
                title="GraphRAG 技术生态",
                summary="GraphRAG 相关技术和组织",
                full_content="详细内容...",
                rank=8.0,
            ),
        ]
        
        # 创建上下文构建器
        context_builder = EntityRelationshipContextBuilder(
            entities=sample_entities,
            relationships=sample_relationships,
            communities=communities,
            community_reports=reports,
            max_tokens=2000,
        )
        
        # 创建 LLM 客户端
        llm_client = GLMClient()
        
        # 创建搜索引擎
        search = LocalSearch(
            llm_client=llm_client,
            context_builder=context_builder,
        )
        
        # 执行搜索
        result = await search.search("GraphRAG 是什么？")
        
        # 验证结果
        assert result is not None
        assert result.response is not None
        assert len(result.response) > 0
        assert result.context_data is not None
        
    @pytest.mark.asyncio
    async def test_local_search_with_embeddings(self, sample_entities, sample_relationships):
        """测试带嵌入的 Local Search"""
        # 为实体添加嵌入
        import numpy as np
        
        for entity in sample_entities:
            entity.description_embedding = np.random.rand(1536).tolist()
        
        communities = [
            Community(
                id="c1",
                title="测试社区",
                level=0,
                entities=[e.name for e in sample_entities],
            ),
        ]
        
        reports = [
            CommunityReport(
                id="cr1",
                community_id="c1",
                title="测试社区",
                summary="测试摘要",
                full_content="测试内容",
                rank=7.0,
            ),
        ]
        
        context_builder = EntityRelationshipContextBuilder(
            entities=sample_entities,
            relationships=sample_relationships,
            communities=communities,
            community_reports=reports,
            max_tokens=2000,
        )
        
        llm_client = GLMClient()
        search = LocalSearch(
            llm_client=llm_client,
            context_builder=context_builder,
        )
        
        result = await search.search("微软开发了什么技术？")
        
        assert result is not None
        assert result.response is not None


class TestSearchComparison:
    """测试搜索引擎对比"""
    
    @pytest.mark.asyncio
    async def test_global_vs_local_search(self, sample_entities, sample_relationships):
        """测试 Global Search vs Local Search"""
        # 准备数据
        communities = [
            Community(
                id="c1",
                title="技术社区",
                level=0,
                entities=[e.name for e in sample_entities],
            ),
        ]
        
        reports = [
            CommunityReport(
                id="cr1",
                community_id="c1",
                title="技术社区报告",
                summary="技术社区的摘要",
                full_content="技术社区的详细内容",
                rank=8.0,
            ),
        ]
        
        llm_client = GLMClient()
        
        # Global Search
        global_context = CommunityContextBuilder(
            community_reports=reports,
            max_tokens=2000,
        )
        global_search = GlobalSearch(
            llm_client=llm_client,
            context_builder=global_context,
        )
        
        # Local Search
        local_context = EntityRelationshipContextBuilder(
            entities=sample_entities,
            relationships=sample_relationships,
            communities=communities,
            community_reports=reports,
            max_tokens=2000,
        )
        local_search = LocalSearch(
            llm_client=llm_client,
            context_builder=local_context,
        )
        
        # 同一个问题
        question = "总结主要内容"
        
        global_result = await global_search.search(question)
        local_result = await local_search.search(question)
        
        # 两种搜索都应该返回结果
        assert global_result is not None
        assert local_result is not None
        assert global_result.response is not None
        assert local_result.response is not None


class TestSearchPerformance:
    """测试搜索性能"""
    
    @pytest.mark.asyncio
    async def test_search_execution_time(self):
        """测试搜索执行时间"""
        import time
        
        # 准备数据
        reports = [
            CommunityReport(
                id=f"cr{i}",
                community_id=f"c{i}",
                title=f"社区 {i}",
                summary=f"摘要 {i}",
                full_content=f"内容 {i}",
                rank=8.0,
            )
            for i in range(10)
        ]
        
        context_builder = CommunityContextBuilder(
            community_reports=reports,
            max_tokens=3000,
        )
        
        llm_client = GLMClient()
        search = GlobalSearch(
            llm_client=llm_client,
            context_builder=context_builder,
        )
        
        # 测量执行时间
        start_time = time.time()
        result = await search.search("总结所有社区")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 验证执行时间合理
        assert execution_time < 5.0  # 5 秒内完成
        
        print(f"\n搜索执行时间: {execution_time:.2f} 秒")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

