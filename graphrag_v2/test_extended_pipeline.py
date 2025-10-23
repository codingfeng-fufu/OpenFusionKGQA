"""测试扩展的 Pipeline。

测试包含实体提取和社区检测的完整 Pipeline。
"""

import asyncio
import logging
import os
import shutil
import sys

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineStorage
from graphrag_v2.pipeline.factory import PipelineFactory
from graphrag_v2.pipeline.runner import create_run_context, run_pipeline


async def test_extended_pipeline():
    """测试扩展的 Pipeline。"""
    print("\n" + "=" * 60)
    print("GraphRAG v2 扩展 Pipeline 测试")
    print("=" * 60)
    
    # 创建测试文档
    test_dir = "test_input"
    os.makedirs(test_dir, exist_ok=True)
    
    test_content = """GraphRAG 是微软开发的一个基于知识图谱的 RAG 系统。
它可以从文本中提取实体和关系，构建知识图谱。
GraphRAG 使用 Leiden 算法进行社区检测。
社区检测可以发现实体之间的聚类结构。
微软公司在自然语言处理领域有深厚的技术积累。
OpenAI 与微软公司有密切的合作关系。
GPT 模型是 OpenAI 开发的大型语言模型。
大型语言模型在知识图谱构建中发挥重要作用。
Leiden 算法是一种高效的社区检测算法。
社区检测算法可以帮助理解复杂网络的结构。
知识图谱可以用于问答系统和推荐系统。
推荐系统需要理解用户和物品之间的关系。
自然语言处理技术不断发展和进步。
技术进步推动了人工智能的发展。
人工智能正在改变我们的生活方式。"""
    
    with open(f"{test_dir}/test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"\n✓ 创建测试文档: {test_dir}/test_doc.txt")

    # 创建配置（指定输入目录）
    from graphrag_v2.config.models.input_config import InputConfig
    config = GraphRagConfig(
        input=InputConfig(base_dir=test_dir)
    )
    
    # 创建 Pipeline（使用标准方法，包含所有工作流）
    pipeline = PipelineFactory.create_pipeline(config, "standard")
    
    print(f"\n✓ 创建 Pipeline")
    print(f"  - 工作流数量: {len(pipeline.workflows)}")
    print(f"  - 工作流列表: {pipeline.names()}")

    # 创建存储
    input_storage = PipelineStorage(base_dir=test_dir)
    output_storage = PipelineStorage(base_dir="output")

    # 创建运行上下文
    context = create_run_context(input_storage, output_storage)

    print(f"\n✓ 创建运行上下文")
    
    # 运行 Pipeline
    print(f"\n开始运行 Pipeline...\n")
    
    results = []
    async for result in run_pipeline(pipeline, config, context):
        print(f"{'✓' if not result.errors else '✗'} 工作流: {result.workflow_name}")
        print(f"  - 运行时间: {result.runtime:.2f}秒")
        
        if result.errors:
            print(f"  - 状态: 失败")
            for error in result.errors:
                print(f"  - 错误: {error}")
        else:
            print(f"  - 状态: 成功")
            if result.result is not None:
                if isinstance(result.result, dict):
                    for key, value in result.result.items():
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            print(f"  - {key}: {len(value)} 条记录")
                        else:
                            print(f"  - {key}: {value}")
        
        print()
        results.append(result)
    
    # 打印统计信息
    print("统计信息:")
    print(f"  - 总运行时间: {context.stats.total_runtime:.2f}秒")
    print(f"  - 文档数量: {context.stats.num_documents}")
    print(f"  - 文本单元数量: {context.stats.num_text_units}")
    print(f"  - 实体数量: {context.stats.num_entities}")
    print(f"  - 关系数量: {context.stats.num_relationships}")
    print(f"  - 社区数量: {context.stats.num_communities}")
    
    # 检查输出数据
    print("\n" + "=" * 60)
    print("输出数据检查")
    print("=" * 60)
    
    # 检查文档
    documents = await context.output_storage.get("documents")
    if documents is not None:
        print(f"\n✓ 文档数据:")
        print(f"  - 行数: {len(documents)}")
        print(f"  - 列: {list(documents.columns)}")
    
    # 检查文本单元
    text_units = await context.output_storage.get("text_units")
    if text_units is not None:
        print(f"\n✓ 文本单元数据:")
        print(f"  - 行数: {len(text_units)}")
        print(f"  - 列: {list(text_units.columns)}")
    
    # 检查实体
    entities = await context.output_storage.get("entities")
    if entities is not None:
        print(f"\n✓ 实体数据:")
        print(f"  - 行数: {len(entities)}")
        print(f"  - 列: {list(entities.columns)}")
        
        # 显示前几个实体
        print(f"\n前 10 个实体:")
        for idx, row in entities.head(10).iterrows():
            print(f"  - {row['name']} ({row['type']})")
    
    # 检查关系
    relationships = await context.output_storage.get("relationships")
    if relationships is not None:
        print(f"\n✓ 关系数据:")
        print(f"  - 行数: {len(relationships)}")
        print(f"  - 列: {list(relationships.columns)}")
        
        # 显示前几个关系
        print(f"\n前 10 个关系:")
        for idx, row in relationships.head(10).iterrows():
            print(f"  - {row['source']} -> {row['target']} (权重: {row['weight']})")
    
    # 检查社区
    communities = await context.output_storage.get("communities")
    if communities is not None:
        print(f"\n✓ 社区数据:")
        print(f"  - 行数: {len(communities)}")
        print(f"  - 列: {list(communities.columns)}")

        if len(communities) > 0:
            # 显示社区统计
            print(f"\n社区统计:")
            print(f"  - 平均社区大小: {communities['size'].mean():.2f}")
            print(f"  - 最大社区大小: {communities['size'].max()}")
            print(f"  - 最小社区大小: {communities['size'].min()}")

            # 显示每个社区
            print(f"\n社区详情:")
            for idx, row in communities.iterrows():
                print(f"  - {row['title']}: {row['size']} 个节点")
                print(f"    节点: {', '.join(row['nodes'][:5])}" +
                      (f" ... (共 {len(row['nodes'])} 个)" if len(row['nodes']) > 5 else ""))
    
    # 检查图
    graph = await context.output_storage.get("graph")
    if graph is not None:
        print(f"\n✓ 图数据:")
        print(f"  - 节点数: {len(graph.nodes)}")
        print(f"  - 边数: {len(graph.edges)}")
        if len(graph.nodes) > 0:
            print(f"  - 平均度: {sum(dict(graph.degree()).values()) / len(graph.nodes):.2f}")
    
    # 清理测试文件
    shutil.rmtree(test_dir)
    print(f"\n✓ 清理测试文件")
    
    print("\n" + "=" * 60)
    print("✓ 扩展 Pipeline 测试完成！")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(test_extended_pipeline())

