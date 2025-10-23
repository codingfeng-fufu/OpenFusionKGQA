"""
社区报告生成工作流。

这个工作流为每个社区生成摘要报告。
"""

import logging
from typing import Any

import pandas as pd

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext
from graphrag_v2.pipeline.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    运行社区报告生成工作流。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        WorkflowFunctionOutput: 包含社区报告 DataFrame
    """
    logger.info("工作流开始: create_community_reports")

    # 从存储中加载数据
    communities = await context.output_storage.get("communities")
    entities = await context.output_storage.get("entities")
    relationships = await context.output_storage.get("relationships")
    
    if communities is None or len(communities) == 0:
        logger.warning("没有社区数据，跳过报告生成")
        empty_df = pd.DataFrame(columns=[
            'id', 'community_id', 'title', 'summary', 
            'full_content', 'rank', 'findings'
        ])
        context.storage.set("community_reports", empty_df)
        return WorkflowFunctionOutput(result=empty_df)
    
    logger.info(f"开始为 {len(communities)} 个社区生成报告")
    
    # 生成社区报告
    reports = await generate_community_reports(
        communities=communities,
        entities=entities,
        relationships=relationships,
        config=config,
        context=context,
    )
    
    # 保存到存储
    await context.output_storage.set("community_reports", reports)
    
    logger.info("工作流完成: create_community_reports")
    logger.info(f"  - 生成了 {len(reports)} 个社区报告")
    
    # 更新统计信息
    context.stats.num_communities = len(communities)
    
    return WorkflowFunctionOutput(result=reports)


async def generate_community_reports(
    communities: pd.DataFrame,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> pd.DataFrame:
    """
    为每个社区生成报告。
    
    Args:
        communities: 社区 DataFrame
        entities: 实体 DataFrame
        relationships: 关系 DataFrame
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        pd.DataFrame: 社区报告 DataFrame
    """
    reports = []
    
    for idx, community in communities.iterrows():
        community_id = community['id']
        nodes = community['nodes']
        
        # 获取社区中的实体
        community_entities = entities[entities['name'].isin(nodes)]
        
        # 获取社区中的关系
        community_relationships = relationships[
            (relationships['source'].isin(nodes)) & 
            (relationships['target'].isin(nodes))
        ]
        
        # 生成报告（简化版本）
        report = await generate_single_community_report(
            community_id=community_id,
            community=community,
            entities=community_entities,
            relationships=community_relationships,
            config=config,
            context=context,
        )
        
        reports.append(report)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"已生成 {idx + 1}/{len(communities)} 个社区报告")
    
    return pd.DataFrame(reports)


async def generate_single_community_report(
    community_id: str,
    community: pd.Series,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> dict[str, Any]:
    """
    为单个社区生成报告。
    
    这是一个简化版本，使用规则生成报告。
    在生产环境中，应该使用 LLM 生成更高质量的报告。
    
    Args:
        community_id: 社区 ID
        community: 社区数据
        entities: 社区中的实体
        relationships: 社区中的关系
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        dict: 社区报告
    """
    # 生成标题
    title = generate_community_title(community, entities)
    
    # 生成摘要
    summary = generate_community_summary(community, entities, relationships)
    
    # 生成完整内容
    full_content = generate_community_full_content(
        community, entities, relationships
    )
    
    # 生成发现（关键洞察）
    findings = generate_community_findings(entities, relationships)
    
    # 计算排名（基于社区大小和连接密度）
    rank = calculate_community_rank(community, relationships)
    
    report = {
        'id': f"report_{community_id}",
        'community_id': community_id,
        'title': title,
        'summary': summary,
        'full_content': full_content,
        'rank': rank,
        'findings': findings,
    }
    
    return report


def generate_community_title(
    community: pd.Series,
    entities: pd.DataFrame,
) -> str:
    """生成社区标题。"""
    # 使用最重要的实体作为标题
    if len(entities) == 0:
        return f"Community {community['id']}"
    
    # 按排名排序，取前3个实体
    top_entities = entities.nlargest(min(3, len(entities)), 'rank')
    entity_names = top_entities['name'].tolist()
    
    if len(entity_names) == 1:
        return f"{entity_names[0]} Community"
    elif len(entity_names) == 2:
        return f"{entity_names[0]} and {entity_names[1]} Community"
    else:
        return f"{entity_names[0]}, {entity_names[1]} and Others"


def generate_community_summary(
    community: pd.Series,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> str:
    """生成社区摘要。"""
    num_entities = len(entities)
    num_relationships = len(relationships)
    
    # 获取实体类型分布
    entity_types = entities['type'].value_counts().to_dict()
    type_summary = ", ".join([f"{count} {type_}" for type_, count in entity_types.items()])
    
    summary = (
        f"This community contains {num_entities} entities and {num_relationships} relationships. "
        f"The entities include: {type_summary}. "
    )
    
    # 添加关键实体
    if len(entities) > 0:
        top_entities = entities.nlargest(min(3, len(entities)), 'rank')
        entity_names = ", ".join(top_entities['name'].tolist())
        summary += f"Key entities: {entity_names}."
    
    return summary


def generate_community_full_content(
    community: pd.Series,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> str:
    """生成社区完整内容。"""
    content_parts = []
    
    # 添加社区概述
    content_parts.append(f"# Community {community['id']}")
    content_parts.append(f"\nSize: {community['size']} nodes")
    content_parts.append(f"Level: {community['level']}")
    content_parts.append(f"Average Degree: {community['avg_degree']:.2f}")
    
    # 添加实体列表
    content_parts.append("\n## Entities")
    for _, entity in entities.iterrows():
        content_parts.append(f"- **{entity['name']}** ({entity['type']}): {entity['description']}")
    
    # 添加关系列表
    content_parts.append("\n## Relationships")
    for _, rel in relationships.iterrows():
        content_parts.append(
            f"- {rel['source']} -> {rel['target']}: {rel['description']} (weight: {rel['weight']})"
        )
    
    return "\n".join(content_parts)


def generate_community_findings(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> str:
    """生成社区发现（关键洞察）。"""
    findings = []
    
    # 发现1：最重要的实体
    if len(entities) > 0:
        top_entity = entities.nlargest(1, 'rank').iloc[0]
        findings.append(
            f"The most important entity is '{top_entity['name']}' ({top_entity['type']})"
        )
    
    # 发现2：最强的关系
    if len(relationships) > 0:
        top_rel = relationships.nlargest(1, 'weight').iloc[0]
        findings.append(
            f"The strongest relationship is between '{top_rel['source']}' and '{top_rel['target']}' "
            f"with weight {top_rel['weight']}"
        )
    
    # 发现3：实体类型分布
    if len(entities) > 0:
        entity_types = entities['type'].value_counts()
        dominant_type = entity_types.index[0]
        findings.append(
            f"The dominant entity type is '{dominant_type}' with {entity_types.iloc[0]} instances"
        )
    
    return " | ".join(findings)


def calculate_community_rank(
    community: pd.Series,
    relationships: pd.DataFrame,
) -> float:
    """
    计算社区排名。
    
    排名基于：
    1. 社区大小
    2. 连接密度
    3. 平均度
    """
    size = community['size']
    avg_degree = community['avg_degree']
    num_edges = len(relationships)
    
    # 计算密度（实际边数 / 可能的最大边数）
    max_edges = size * (size - 1) / 2 if size > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0
    
    # 综合排名（归一化到 0-1）
    rank = (
        0.4 * min(size / 20, 1.0) +  # 大小权重
        0.3 * density +                # 密度权重
        0.3 * min(avg_degree / 10, 1.0)  # 平均度权重
    )
    
    return rank

