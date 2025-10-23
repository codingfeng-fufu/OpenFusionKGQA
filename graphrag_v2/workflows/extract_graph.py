"""提取图谱工作流。

从文本单元中提取实体和关系。

注意：这是一个简化版本，使用规则提取而非 LLM。
在生产环境中，应该使用 LLM 进行更准确的提取。
"""

import hashlib
import logging
import re

import pandas as pd

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext
from graphrag_v2.pipeline.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


def extract_entities_from_text(text: str, text_unit_id: str) -> list[dict]:
    """从文本中提取实体（简化版本，使用规则）。
    
    在生产环境中，这应该使用 LLM 进行提取。
    这里我们使用简单的规则：
    - 大写开头的连续词作为实体
    - 特定模式（如"XX公司"、"XX系统"）
    
    Args:
        text: 文本内容
        text_unit_id: 文本单元 ID
        
    Returns:
        实体列表
    """
    entities = []
    
    # 规则1：提取大写开头的连续词（2-4个词）
    # 例如：GraphRAG, Microsoft Corporation
    pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
    matches1 = re.findall(pattern1, text)
    
    for match in matches1:
        # 过滤掉常见的非实体词
        if match not in ['The', 'This', 'That', 'These', 'Those', 'A', 'An']:
            entities.append({
                'name': match,
                'type': 'ENTITY',
                'description': f'从文本中提取的实体: {match}',
            })
    
    # 规则2：提取中文实体模式
    # 例如：微软公司、GraphRAG系统、知识图谱
    pattern2 = r'([一-龥]{2,10}(?:公司|系统|平台|技术|算法|模型|方法))'
    matches2 = re.findall(pattern2, text)
    
    for match in matches2:
        entities.append({
            'name': match,
            'type': 'ORGANIZATION' if '公司' in match else 'TECHNOLOGY',
            'description': f'从文本中提取的实体: {match}',
        })
    
    # 规则3：提取专有名词（连续的中文大写或英文）
    pattern3 = r'([A-Z]{2,}|[一-龥]{2,6})'
    matches3 = re.findall(pattern3, text)
    
    for match in matches3:
        if len(match) >= 2 and match not in [e['name'] for e in entities]:
            entities.append({
                'name': match,
                'type': 'CONCEPT',
                'description': f'从文本中提取的概念: {match}',
            })
    
    # 为每个实体添加 ID 和来源信息
    for entity in entities:
        entity_name = entity['name']
        # 使用名称的哈希作为 ID
        entity_id = hashlib.md5(entity_name.encode()).hexdigest()[:16]
        entity['id'] = f"entity_{entity_id}"
        entity['text_unit_ids'] = [text_unit_id]
    
    return entities


def extract_relationships_from_text(
    text: str,
    entities: list[dict],
    text_unit_id: str,
) -> list[dict]:
    """从文本中提取关系（简化版本）。
    
    在生产环境中，这应该使用 LLM 进行提取。
    这里我们使用简单的规则：
    - 如果两个实体在同一个句子中出现，则认为它们有关系
    
    Args:
        text: 文本内容
        entities: 已提取的实体列表
        text_unit_id: 文本单元 ID
        
    Returns:
        关系列表
    """
    relationships = []
    
    # 将文本分成句子
    sentences = re.split(r'[。！？.!?]', text)
    
    # 在每个句子中查找共现的实体
    for sentence in sentences:
        sentence_entities = []
        for entity in entities:
            if entity['name'] in sentence:
                sentence_entities.append(entity)
        
        # 如果一个句子中有多个实体，创建关系
        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):
                source = sentence_entities[i]['name']
                target = sentence_entities[j]['name']
                
                # 生成关系 ID
                rel_content = f"{source}_{target}"
                rel_id = hashlib.md5(rel_content.encode()).hexdigest()[:16]
                
                relationships.append({
                    'id': f"rel_{rel_id}",
                    'source': source,
                    'target': target,
                    'description': f'{source} 与 {target} 在文本中共现',
                    'weight': 1.0,
                    'text_unit_ids': [text_unit_id],
                })
    
    return relationships


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """提取图谱工作流。
    
    从文本单元中提取实体和关系。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        工作流输出，包含实体和关系
    """
    logger.info("工作流开始: extract_graph")
    
    # 从存储加载文本单元
    text_units = await context.output_storage.get("text_units")
    if text_units is None:
        logger.error("未找到文本单元数据")
        return WorkflowFunctionOutput(result=None, stop=True)
    
    logger.info(f"开始从 {len(text_units)} 个文本单元中提取实体和关系")
    
    # 提取实体和关系
    all_entities = []
    all_relationships = []
    
    for idx, row in text_units.iterrows():
        text_unit_id = row["id"]
        text = row["text"]
        
        # 提取实体
        entities = extract_entities_from_text(text, text_unit_id)
        all_entities.extend(entities)
        
        # 提取关系
        relationships = extract_relationships_from_text(text, entities, text_unit_id)
        all_relationships.extend(relationships)
        
        logger.info(f"文本单元 {idx + 1}/{len(text_units)}: 提取了 {len(entities)} 个实体, {len(relationships)} 个关系")
    
    # 合并重复的实体（按名称）
    entity_dict = {}
    for entity in all_entities:
        name = entity['name']
        if name not in entity_dict:
            entity_dict[name] = entity
        else:
            # 合并 text_unit_ids
            existing = entity_dict[name]
            existing['text_unit_ids'] = list(set(
                existing['text_unit_ids'] + entity['text_unit_ids']
            ))
    
    # 合并重复的关系（按 source-target 对）
    relationship_dict = {}
    for rel in all_relationships:
        key = (rel['source'], rel['target'])
        if key not in relationship_dict:
            relationship_dict[key] = rel
        else:
            # 合并 text_unit_ids 并增加权重
            existing = relationship_dict[key]
            existing['text_unit_ids'] = list(set(
                existing['text_unit_ids'] + rel['text_unit_ids']
            ))
            existing['weight'] += 1.0
    
    # 转换为 DataFrame
    entities_df = pd.DataFrame(list(entity_dict.values()))
    relationships_df = pd.DataFrame(list(relationship_dict.values()))
    
    # 添加排名（基于出现次数）
    if len(entities_df) > 0:
        entities_df['rank'] = entities_df['text_unit_ids'].apply(len)
    
    if len(relationships_df) > 0:
        relationships_df['rank'] = relationships_df['weight']
    
    # 保存到输出存储
    await context.output_storage.set("entities", entities_df)
    await context.output_storage.set("relationships", relationships_df)
    
    # 更新统计信息
    context.stats.num_entities = len(entities_df)
    context.stats.num_relationships = len(relationships_df)
    
    logger.info(f"工作流完成: extract_graph")
    logger.info(f"  - 提取了 {len(entities_df)} 个唯一实体")
    logger.info(f"  - 提取了 {len(relationships_df)} 个唯一关系")
    
    return WorkflowFunctionOutput(
        result={
            'entities': entities_df,
            'relationships': relationships_df,
        }
    )

