"""
完整 Pipeline 测试。

测试所有 6 个工作流的端到端执行。
"""

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.config.models.input_config import InputConfig
from graphrag_v2.pipeline.context import (
    PipelineRunContext,
    PipelineRunStats,
    PipelineStorage,
)
from graphrag_v2.pipeline.factory import PipelineFactory
from graphrag_v2.pipeline.runner import run_pipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_full_pipeline():
    """测试完整的 Pipeline。"""
    print("\n" + "=" * 60)
    print("GraphRAG v2 完整 Pipeline 测试")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = "test_input"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试文档（更长的文本，用于测试分块）
    test_content = """GraphRAG 是微软开发的一个基于知识图谱的 RAG 系统。
它可以从文本中提取实体和关系，构建知识图谱。
GraphRAG 使用 Leiden 算法进行社区检测。
社区检测可以帮助理解大规模知识图谱的结构。

微软公司在自然语言处理领域有深厚的技术积累。
微软与 OpenAI 合作，共同推动人工智能技术的发展。
OpenAI 开发的 GPT 系列模型在语言理解和生成方面表现出色。
GPT-4 是目前最先进的大型语言模型之一。

知识图谱可以用于问答系统、推荐系统和信息检索。
实体提取是知识图谱构建的第一步。
关系抽取可以发现实体之间的语义联系。
Leiden 算法是一种高效的社区检测算法。

自然语言处理技术不断发展和进步。
深度学习推动了 NLP 领域的重大突破。
Transformer 架构彻底改变了语言模型的设计。
注意力机制是 Transformer 的核心创新。

大型语言模型在知识图谱构建中发挥重要作用。
它们可以理解复杂的语义关系。
提示工程是优化 LLM 性能的关键技术。
Few-shot 学习可以提高模型在特定任务上的表现。

人工智能正在改变我们的生活方式。
机器学习算法在各个领域都有广泛应用。
数据是训练高质量模型的基础。
算力的提升使得更大规模的模型训练成为可能。

GraphRAG 结合了检索增强生成和知识图谱的优势。
它可以提供更准确、更可解释的答案。
社区报告可以总结大规模图谱的关键信息。
向量嵌入使得语义搜索成为可能。"""
    
    with open(f"{test_dir}/test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"\n✓ 创建测试文档: {test_dir}/test_doc.txt")
    
    # 创建配置（指定输入目录）
    config = GraphRagConfig(
        input=InputConfig(base_dir=test_dir)
    )
    
    # 创建 Pipeline
    pipeline = PipelineFactory.create_pipeline(config, "standard")
    print(f"\n✓ 创建 Pipeline")
    print(f"  - 工作流数量: {len(pipeline.workflows)}")
    print(f"  - 工作流列表: {[name for name, _ in pipeline.workflows]}")
    
    # 创建运行上下文
    input_storage = PipelineStorage(base_dir=test_dir)
    output_storage = PipelineStorage(base_dir="output")
    stats = PipelineRunStats()
    context = PipelineRunContext(
        input_storage=input_storage,
        output_storage=output_storage,
        stats=stats,
    )
    print(f"\n✓ 创建运行上下文")
    
    # 运行 Pipeline
    print(f"\n开始运行 Pipeline...")
    print()

    async for _ in run_pipeline(pipeline, config, context):
        pass  # 等待所有工作流完成
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  - 总运行时间: {stats.total_runtime:.2f}秒")
    print(f"  - 文档数量: {stats.num_documents}")
    print(f"  - 文本单元数量: {stats.num_text_units}")
    print(f"  - 实体数量: {stats.num_entities}")
    print(f"  - 关系数量: {stats.num_relationships}")
    print(f"  - 社区数量: {stats.num_communities}")
    
    # 检查输出数据
    print(f"\n" + "=" * 60)
    print("输出数据检查")
    print("=" * 60)
    
    # 检查文档
    documents = output_storage.data.get("documents")
    print(f"\n✓ 文档数据:")
    print(f"  - 行数: {len(documents)}")
    print(f"  - 列: {list(documents.columns)}")

    # 检查文本单元
    text_units = output_storage.data.get("text_units")
    print(f"\n✓ 文本单元数据:")
    print(f"  - 行数: {len(text_units)}")
    print(f"  - 列: {list(text_units.columns)}")
    if len(text_units) > 0:
        print(f"  - 总 tokens: {text_units['n_tokens'].sum()}")

    # 检查实体
    entities = output_storage.data.get("entities")
    print(f"\n✓ 实体数据:")
    print(f"  - 行数: {len(entities)}")
    print(f"  - 列: {list(entities.columns)}")
    
    if len(entities) > 0:
        print(f"\n前 10 个实体:")
        for entity in entities.head(10)['name']:
            entity_type = entities[entities['name'] == entity]['type'].iloc[0]
            print(f"  - {entity} ({entity_type})")
    
    # 检查关系
    relationships = output_storage.data.get("relationships")
    print(f"\n✓ 关系数据:")
    print(f"  - 行数: {len(relationships)}")
    print(f"  - 列: {list(relationships.columns)}")

    if len(relationships) > 0:
        print(f"\n前 10 个关系:")
        for _, rel in relationships.head(10).iterrows():
            print(f"  - {rel['source']} -> {rel['target']} (权重: {rel['weight']})")

    # 检查社区
    communities = output_storage.data.get("communities")
    print(f"\n✓ 社区数据:")
    print(f"  - 行数: {len(communities)}")
    print(f"  - 列: {list(communities.columns)}")

    if len(communities) > 0:
        print(f"\n社区统计:")
        print(f"  - 平均社区大小: {communities['size'].mean():.2f}")
        print(f"  - 最大社区大小: {communities['size'].max()}")
        print(f"  - 最小社区大小: {communities['size'].min()}")

    # 检查社区报告
    community_reports = output_storage.data.get("community_reports")
    print(f"\n✓ 社区报告数据:")
    print(f"  - 行数: {len(community_reports)}")
    if len(community_reports) > 0:
        print(f"  - 列: {list(community_reports.columns)}")
        
        print(f"\n前 3 个社区报告:")
        for idx, report in community_reports.head(3).iterrows():
            print(f"\n  社区 {idx}:")
            print(f"    标题: {report['title']}")
            print(f"    摘要: {report['summary'][:100]}...")
            print(f"    排名: {report['rank']:.3f}")
    
    # 检查嵌入
    text_unit_embeddings = output_storage.data.get("text_unit_embeddings")
    entity_embeddings = output_storage.data.get("entity_embeddings")
    
    print(f"\n✓ 嵌入数据:")
    if text_unit_embeddings is not None:
        print(f"  - 文本单元嵌入: {len(text_unit_embeddings)} 个")
        if len(text_unit_embeddings) > 0:
            embedding_dim = len(text_unit_embeddings.iloc[0]['embedding'])
            print(f"  - 嵌入维度: {embedding_dim}")
    
    if entity_embeddings is not None:
        print(f"  - 实体嵌入: {len(entity_embeddings)} 个")
        if len(entity_embeddings) > 0:
            embedding_dim = len(entity_embeddings.iloc[0]['embedding'])
            print(f"  - 嵌入维度: {embedding_dim}")
    
    # 测试嵌入相似度搜索
    if entity_embeddings is not None and len(entity_embeddings) > 0:
        from graphrag_v2.workflows.generate_embeddings import find_similar_texts
        
        # 使用第一个实体作为查询
        query_entity = entity_embeddings.iloc[0]
        query_embedding = query_entity['embedding']
        
        print(f"\n✓ 嵌入相似度搜索测试:")
        print(f"  查询实体: {query_entity['name']}")
        
        similar = find_similar_texts(query_embedding, entity_embeddings, top_k=5)
        print(f"\n  最相似的 5 个实体:")
        for _, row in similar.iterrows():
            print(f"    - {row['text'][:50]}... (相似度: {row['similarity']:.3f})")
    
    # 清理测试文件
    shutil.rmtree(test_dir)
    print(f"\n✓ 清理测试文件")
    
    print(f"\n" + "=" * 60)
    print("✓ 完整 Pipeline 测试完成！")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())

