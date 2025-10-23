"""测试查询引擎。"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Windows 控制台 UTF-8 编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from graphrag_v2.query import (
    CommunityContextBuilder,
    EntityRelationshipContextBuilder,
    GlobalSearch,
    LocalSearch,
)


async def test_query_engines():
    """测试查询引擎。"""
    print("=" * 60)
    print("GraphRAG v2 查询引擎测试")
    print("=" * 60)
    print()
    
    # 步骤 1: 加载索引数据
    print("步骤 1: 加载索引数据")
    print()
    
    # 从之前的 Pipeline 运行结果加载数据
    # 这里使用模拟数据
    
    # 社区报告
    community_reports = pd.DataFrame([
        {
            "id": "report_0",
            "community_id": 0,
            "title": "知识图谱和问答系统",
            "summary": "This community contains 7 entities and 21 relationships...",
            "full_content": "## 实体\n- 知识图谱\n- 问答系统\n...",
            "rank": 0.641,
            "findings": "最重要的实体: 知识图谱",
        },
        {
            "id": "report_1",
            "community_id": 1,
            "title": "RAG 和微软",
            "summary": "This community contains 14 entities and 31 relationships...",
            "full_content": "## 实体\n- RAG\n- 微软\n...",
            "rank": 0.554,
            "findings": "最重要的实体: RAG",
        },
        {
            "id": "report_2",
            "community_id": 2,
            "title": "Leiden 算法和社区检测",
            "summary": "This community contains 11 entities and 29 relationships...",
            "full_content": "## 实体\n- Leiden\n- 社区检测\n...",
            "rank": 0.564,
            "findings": "最重要的实体: Leiden",
        },
    ])
    
    # 实体
    entities = pd.DataFrame([
        {"name": "GraphRAG", "type": "CONCEPT", "description": "知识图谱增强检索系统", "id": "entity_0"},
        {"name": "微软", "type": "ORGANIZATION", "description": "微软公司", "id": "entity_1"},
        {"name": "OpenAI", "type": "ORGANIZATION", "description": "OpenAI 公司", "id": "entity_2"},
        {"name": "Leiden", "type": "CONCEPT", "description": "社区检测算法", "id": "entity_3"},
        {"name": "知识图谱", "type": "CONCEPT", "description": "知识图谱", "id": "entity_4"},
    ])
    
    # 关系
    relationships = pd.DataFrame([
        {"id": "rel_0", "source": "GraphRAG", "target": "微软", "description": "开发", "weight": 2.0},
        {"id": "rel_1", "source": "微软", "target": "OpenAI", "description": "合作", "weight": 1.5},
        {"id": "rel_2", "source": "GraphRAG", "target": "知识图谱", "description": "使用", "weight": 2.0},
        {"id": "rel_3", "source": "GraphRAG", "target": "Leiden", "description": "使用", "weight": 1.5},
    ])
    
    # 实体嵌入
    import hashlib
    import numpy as np
    
    def generate_embedding(text: str) -> list[float]:
        """生成确定性嵌入。"""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        dimension = 1536
        
        embedding = []
        for i in range(dimension):
            byte_idx = (i * 2) % len(hash_bytes)
            value = int.from_bytes(hash_bytes[byte_idx:byte_idx+2], 'big')
            normalized = (value / 65535.0) * 2 - 1
            embedding.append(normalized)
        
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    entity_embeddings = pd.DataFrame([
        {"name": entity["name"], "embedding": generate_embedding(entity["name"])}
        for _, entity in entities.iterrows()
    ])
    
    print(f"  - 社区报告: {len(community_reports)} 个")
    print(f"  - 实体: {len(entities)} 个")
    print(f"  - 关系: {len(relationships)} 个")
    print(f"  - 实体嵌入: {len(entity_embeddings)} 个")
    print()
    
    # 步骤 2: 测试 Global Search
    print("步骤 2: 测试 Global Search")
    print()
    
    # 创建 Global Context Builder
    global_context_builder = CommunityContextBuilder(
        community_reports=community_reports,
        max_tokens=8000,
        batch_size=2,
    )
    
    # 创建 Global Search
    global_search = GlobalSearch(
        context_builder=global_context_builder,
        response_type="multiple paragraphs",
    )
    
    # 执行搜索
    query1 = "总结整个数据集的主要主题"
    print(f"查询: {query1}")
    print()
    
    result1 = await global_search.search(query1)
    
    print(f"✓ Global Search 完成")
    print(f"  - 响应长度: {len(result1.response)} 字符")
    print(f"  - 完成时间: {result1.completion_time:.2f} 秒")
    print(f"  - LLM 调用次数: {result1.llm_calls}")
    print(f"  - Map 响应数: {len(result1.map_responses)}")
    print()
    
    print("响应预览:")
    print(result1.response[:300] + "...")
    print()
    
    # 步骤 3: 测试 Local Search
    print("步骤 3: 测试 Local Search")
    print()
    
    # 创建 Local Context Builder
    local_context_builder = EntityRelationshipContextBuilder(
        entities=entities,
        relationships=relationships,
        community_reports=community_reports,
        entity_embeddings=entity_embeddings,
        top_k_entities=3,
        top_k_relationships=5,
    )
    
    # 创建 Local Search
    local_search = LocalSearch(
        context_builder=local_context_builder,
        response_type="multiple paragraphs",
    )
    
    # 执行搜索
    query2 = "GraphRAG 是什么？"
    print(f"查询: {query2}")
    print()
    
    result2 = await local_search.search(query2)
    
    print(f"✓ Local Search 完成")
    print(f"  - 响应长度: {len(result2.response)} 字符")
    print(f"  - 完成时间: {result2.completion_time:.2f} 秒")
    print(f"  - LLM 调用次数: {result2.llm_calls}")
    print(f"  - 相关实体数: {len(result2.context_data['entities'])}")
    print(f"  - 相关关系数: {len(result2.context_data['relationships'])}")
    print()
    
    print("响应预览:")
    print(result2.response[:300] + "...")
    print()
    
    # 步骤 4: 测试另一个 Local Search 查询
    print("步骤 4: 测试另一个 Local Search 查询")
    print()
    
    query3 = "Leiden 算法的作用是什么？"
    print(f"查询: {query3}")
    print()
    
    result3 = await local_search.search(query3)
    
    print(f"✓ Local Search 完成")
    print(f"  - 响应长度: {len(result3.response)} 字符")
    print(f"  - 完成时间: {result3.completion_time:.2f} 秒")
    print(f"  - 相关实体数: {len(result3.context_data['entities'])}")
    print()
    
    print("响应预览:")
    print(result3.response[:300] + "...")
    print()
    
    # 总结
    print("=" * 60)
    print("查询引擎测试完成！")
    print("=" * 60)
    print()
    print("✓ Global Search 测试通过")
    print("✓ Local Search 测试通过")
    print()
    print("下一步:")
    print("  - 集成真实的 LLM API（OpenAI GPT-4）")
    print("  - 优化上下文构建策略")
    print("  - 添加更多的搜索模式（DRIFT Search）")
    print("  - 实现流式响应")


if __name__ == "__main__":
    asyncio.run(test_query_engines())

