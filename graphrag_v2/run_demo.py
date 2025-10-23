"""
GraphRAG v2 完整演示
展示从文档处理到查询的完整流程
"""

import asyncio
import os
import sys
from pathlib import Path
import pandas as pd

# 设置UTF-8编码（Windows兼容）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量（如果需要）
# os.environ["ZHIPUAI_API_KEY"] = "your_api_key_here"

from graphrag_v2.config import GraphRagConfig, load_config
from graphrag_v2.llm import GLMClient
from graphrag_v2.data_model import Document, Entity, Relationship, Community, CommunityReport
from graphrag_v2.prompts import (
    get_entity_extraction_prompt,
    get_community_report_prompt,
    get_global_search_map_prompt,
    get_local_search_prompt,
)


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


async def demo_step1_config():
    """步骤1: 加载配置"""
    print_section("步骤1: 加载配置")

    config = load_config("test_data/config.yaml")
    
    print(f"✓ 配置加载成功")
    print(f"  根目录: {config.root_dir}")
    print(f"  输入目录: {config.input.base_dir}")
    print(f"  输出目录: {config.output.base_dir}")
    print(f"  分块大小: {config.chunks.size}")
    print(f"  分块重叠: {config.chunks.overlap}")
    
    return config


async def demo_step2_load_document():
    """步骤2: 加载文档"""
    print_section("步骤2: 加载文档")
    
    doc_path = Path("test_data/sample_document.txt")
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    document = Document(
        id="doc_001",
        title="GraphRAG技术概述",
        text=text,
    )
    
    print(f"✓ 文档加载成功")
    print(f"  文档ID: {document.id}")
    print(f"  文档标题: {document.title}")
    print(f"  文档长度: {len(document.text)} 字符")
    print(f"\n文档内容预览:")
    print(f"  {document.text[:200]}...")
    
    return document


async def demo_step3_chunk_document(document: Document, config: GraphRagConfig):
    """步骤3: 文档分块"""
    print_section("步骤3: 文档分块")
    
    # 简单分块（实际应使用tiktoken）
    chunk_size = config.chunks.size
    overlap = config.chunks.overlap
    
    chunks = []
    text = document.text
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        chunks.append({
            "id": f"chunk_{chunk_id:03d}",
            "document_id": document.id,
            "text": chunk_text,
            "start": start,
            "end": end,
        })
        
        chunk_id += 1
        start = end - overlap if end < len(text) else end
    
    print(f"✓ 文档分块完成")
    print(f"  总块数: {len(chunks)}")
    print(f"  平均块大小: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} 字符")
    print(f"\n第一个块预览:")
    print(f"  {chunks[0]['text'][:150]}...")
    
    return chunks


async def demo_step4_extract_entities(chunks: list, config: GraphRagConfig):
    """步骤4: 提取实体和关系"""
    print_section("步骤4: 提取实体和关系（模拟）")
    
    # 创建LLM客户端
    llm_client = GLMClient()
    
    # 生成实体提取Prompt
    sample_chunk = chunks[0]['text']
    entity_types = ["人物", "组织", "技术", "产品", "概念"]
    prompt = get_entity_extraction_prompt(
        entity_types=entity_types,
        input_text=sample_chunk,
    )
    
    print(f"✓ 实体提取Prompt生成成功")
    print(f"  Prompt长度: {len(prompt)} 字符")
    print(f"\nPrompt预览:")
    print(f"  {prompt[:300]}...")
    
    # 模拟提取的实体和关系
    entities = [
        Entity(id="e1", title="GraphRAG", type="技术", description="微软研究院开发的检索增强生成技术"),
        Entity(id="e2", title="微软研究院", type="组织", description="开发GraphRAG的研究机构"),
        Entity(id="e3", title="Jonathan Larson", type="人物", description="GraphRAG项目负责人"),
        Entity(id="e4", title="Steven Truitt", type="人物", description="GraphRAG团队成员"),
        Entity(id="e5", title="Louvain算法", type="技术", description="用于社区检测的算法"),
        Entity(id="e6", title="GLM-4", type="产品", description="语言模型"),
        Entity(id="e7", title="Sarah Chen", type="人物", description="负责算法优化"),
        Entity(id="e8", title="David Kim", type="人物", description="负责系统架构"),
    ]
    
    relationships = [
        Relationship(id="r1", source="微软研究院", target="GraphRAG", description="开发: 微软研究院开发了GraphRAG"),
        Relationship(id="r2", source="Jonathan Larson", target="GraphRAG", description="领导: Jonathan Larson领导GraphRAG项目"),
        Relationship(id="r3", source="Steven Truitt", target="GraphRAG", description="参与: Steven Truitt参与GraphRAG开发"),
        Relationship(id="r4", source="GraphRAG", target="Louvain算法", description="使用: GraphRAG使用Louvain算法进行社区检测"),
        Relationship(id="r5", source="Sarah Chen", target="微软研究院", description="任职: Sarah Chen在微软研究院工作"),
        Relationship(id="r6", source="David Kim", target="微软研究院", description="任职: David Kim在微软研究院工作"),
    ]
    
    print(f"\n✓ 实体提取完成（模拟）")
    print(f"  提取实体数: {len(entities)}")
    print(f"  提取关系数: {len(relationships)}")
    
    print(f"\n实体示例:")
    for i, entity in enumerate(entities[:3], 1):
        print(f"  {i}. {entity.title} ({entity.type}): {entity.description}")
    
    print(f"\n关系示例:")
    for i, rel in enumerate(relationships[:3], 1):
        rel_type = rel.description.split(":")[0] if ":" in rel.description else "关系"
        print(f"  {i}. {rel.source} --[{rel_type}]--> {rel.target}")
    
    return entities, relationships


async def demo_step5_detect_communities(entities: list, relationships: list):
    """步骤5: 社区检测"""
    print_section("步骤5: 社区检测（模拟）")
    
    # 模拟社区检测结果
    communities = [
        Community(
            id="c1",
            title="GraphRAG核心团队",
            level="0",
            entity_ids=["e1", "e2", "e3", "e4"],
            relationship_ids=["r1", "r2", "r3"],
        ),
        Community(
            id="c2",
            title="技术栈",
            level="0",
            entity_ids=["e1", "e5", "e6"],
            relationship_ids=["r4"],
        ),
        Community(
            id="c3",
            title="研发团队",
            level="0",
            entity_ids=["e2", "e7", "e8"],
            relationship_ids=["r5", "r6"],
        ),
    ]
    
    print(f"✓ 社区检测完成（模拟）")
    print(f"  检测到社区数: {len(communities)}")
    
    print(f"\n社区详情:")
    for i, comm in enumerate(communities, 1):
        print(f"  {i}. {comm.title}")
        print(f"     实体数: {len(comm.entity_ids or [])}")
        print(f"     关系数: {len(comm.relationship_ids or [])}")
    
    return communities


async def demo_step6_generate_reports(communities: list, entities: list, relationships: list):
    """步骤6: 生成社区报告"""
    print_section("步骤6: 生成社区报告（模拟）")
    
    # 创建LLM客户端
    llm_client = GLMClient()
    
    # 为第一个社区生成报告Prompt
    comm = communities[0]
    
    # 构建实体和关系文本
    entity_texts = []
    for eid in comm.entity_ids or []:
        entity = next((e for e in entities if e.id == eid), None)
        if entity:
            entity_texts.append(f"- {entity.title} ({entity.type}): {entity.description}")
    
    rel_texts = []
    for rid in comm.relationship_ids or []:
        rel = next((r for r in relationships if r.id == rid), None)
        if rel:
            rel_type = rel.description.split(":")[0] if ":" in rel.description else "关系"
            rel_texts.append(f"- {rel.source} --[{rel_type}]--> {rel.target}: {rel.description}")
    
    entities_text = "\n".join(entity_texts)
    relationships_text = "\n".join(rel_texts)
    
    prompt = get_community_report_prompt(
        entities=entities_text,
        relationships=relationships_text,
        role="数据分析师",
        report_length="300-500字",
    )
    
    print(f"✓ 社区报告Prompt生成成功")
    print(f"  社区: {comm.title}")
    print(f"  Prompt长度: {len(prompt)} 字符")
    
    # 模拟生成的报告
    reports = [
        CommunityReport(
            id="cr1",
            community_id="c1",
            title="GraphRAG核心团队分析",
            summary="GraphRAG是由微软研究院开发的创新技术，由Jonathan Larson和Steven Truitt领导。该团队专注于结合知识图谱和大语言模型的优势。",
            full_content="GraphRAG核心团队由微软研究院的精英成员组成。Jonathan Larson作为项目负责人，带领团队开发了这一创新的检索增强生成技术。Steven Truitt作为核心成员，在技术实现方面做出了重要贡献。",
            rank=8.5,
        ),
        CommunityReport(
            id="cr2",
            community_id="c2",
            title="GraphRAG技术栈分析",
            summary="GraphRAG采用Louvain算法进行社区检测，并支持GLM-4等多种语言模型。",
            full_content="GraphRAG的技术栈包括多个关键组件。Louvain算法用于社区检测，能够有效地将相关实体聚类。系统支持GLM-4等先进的语言模型，提供强大的文本理解和生成能力。",
            rank=7.8,
        ),
    ]
    
    print(f"\n✓ 社区报告生成完成（模拟）")
    print(f"  生成报告数: {len(reports)}")
    
    print(f"\n报告示例:")
    for i, report in enumerate(reports[:2], 1):
        print(f"  {i}. {report.title}")
        print(f"     重要性: {report.rank}/10")
        print(f"     摘要: {report.summary}")
    
    return reports


async def demo_step7_global_search(reports: list):
    """步骤7: 全局搜索"""
    print_section("步骤7: 全局搜索（模拟）")
    
    query = "GraphRAG的主要特点是什么？"
    
    # 构建上下文数据
    context_data = "\n\n".join([
        f"报告 {i+1}: {report.title}\n{report.full_content}"
        for i, report in enumerate(reports)
    ])
    
    # 生成Global Search Map Prompt
    map_prompt = get_global_search_map_prompt(
        context_data=context_data,
        query=query,
        max_length=500,
    )
    
    print(f"✓ 全局搜索Prompt生成成功")
    print(f"  查询: {query}")
    print(f"  上下文长度: {len(context_data)} 字符")
    print(f"  Prompt长度: {len(map_prompt)} 字符")
    
    # 模拟搜索结果
    result = """
    基于社区报告分析，GraphRAG的主要特点包括：
    
    1. **创新的技术架构** [Data: Reports (1)]
       - 结合知识图谱和大语言模型的优势
       - 由微软研究院的精英团队开发
    
    2. **先进的社区检测** [Data: Reports (2)]
       - 采用Louvain算法进行实体聚类
       - 能够发现隐藏的关联模式
    
    3. **灵活的模型支持** [Data: Reports (2)]
       - 支持GLM-4等多种语言模型
       - 具有良好的扩展性
    """
    
    print(f"\n✓ 全局搜索完成（模拟）")
    print(f"\n搜索结果:")
    print(result)
    
    return result


async def demo_step8_local_search(entities: list, relationships: list):
    """步骤8: 局部搜索"""
    print_section("步骤8: 局部搜索（模拟）")
    
    query = "Jonathan Larson在GraphRAG项目中的角色是什么？"
    
    # 构建上下文数据（相关实体和关系）
    context_parts = []
    context_parts.append("实体:")
    for entity in entities[:4]:
        context_parts.append(f"- {entity.title} ({entity.type}): {entity.description}")
    
    context_parts.append("\n关系:")
    for rel in relationships[:3]:
        rel_type = rel.description.split(":")[0] if ":" in rel.description else "关系"
        context_parts.append(f"- {rel.source} --[{rel_type}]--> {rel.target}: {rel.description}")
    
    context_data = "\n".join(context_parts)
    
    # 生成Local Search Prompt
    local_prompt = get_local_search_prompt(
        context_data=context_data,
        query=query,
        response_type="简短段落",
    )
    
    print(f"✓ 局部搜索Prompt生成成功")
    print(f"  查询: {query}")
    print(f"  上下文长度: {len(context_data)} 字符")
    print(f"  Prompt长度: {len(local_prompt)} 字符")
    
    # 模拟搜索结果
    result = """
    根据知识图谱数据，Jonathan Larson是GraphRAG项目的负责人 [Data: Entities (e3); Relationships (r2)]。
    他领导微软研究院的团队开发了这一创新的检索增强生成技术。
    """
    
    print(f"\n✓ 局部搜索完成（模拟）")
    print(f"\n搜索结果:")
    print(result)
    
    return result


async def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("  GraphRAG v2 完整流程演示")
    print("🚀" * 40)
    
    try:
        # 步骤1: 加载配置
        config = await demo_step1_config()
        
        # 步骤2: 加载文档
        document = await demo_step2_load_document()
        
        # 步骤3: 文档分块
        chunks = await demo_step3_chunk_document(document, config)
        
        # 步骤4: 提取实体和关系
        entities, relationships = await demo_step4_extract_entities(chunks, config)
        
        # 步骤5: 社区检测
        communities = await demo_step5_detect_communities(entities, relationships)
        
        # 步骤6: 生成社区报告
        reports = await demo_step6_generate_reports(communities, entities, relationships)
        
        # 步骤7: 全局搜索
        global_result = await demo_step7_global_search(reports)
        
        # 步骤8: 局部搜索
        local_result = await demo_step8_local_search(entities, relationships)
        
        # 总结
        print_section("演示完成")
        print("✅ 所有步骤执行成功！")
        print("\n流程总结:")
        print(f"  1. ✓ 配置加载")
        print(f"  2. ✓ 文档加载 (1个文档)")
        print(f"  3. ✓ 文档分块 ({len(chunks)}个块)")
        print(f"  4. ✓ 实体提取 ({len(entities)}个实体, {len(relationships)}个关系)")
        print(f"  5. ✓ 社区检测 ({len(communities)}个社区)")
        print(f"  6. ✓ 报告生成 ({len(reports)}个报告)")
        print(f"  7. ✓ 全局搜索")
        print(f"  8. ✓ 局部搜索")
        
        print("\n" + "🎉" * 40)
        print("  GraphRAG v2 演示成功完成！")
        print("🎉" * 40 + "\n")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

