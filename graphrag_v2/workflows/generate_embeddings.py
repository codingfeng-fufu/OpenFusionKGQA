"""
文本嵌入生成工作流。

这个工作流为文本单元和实体生成向量嵌入。
"""

import hashlib
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
    运行文本嵌入生成工作流。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        WorkflowFunctionOutput: 包含嵌入结果的字典
    """
    logger.info("工作流开始: generate_embeddings")

    # 从存储中加载数据
    text_units = await context.output_storage.get("text_units")
    entities = await context.output_storage.get("entities")
    
    # 生成嵌入
    embeddings_result = await generate_embeddings(
        text_units=text_units,
        entities=entities,
        config=config,
        context=context,
    )
    
    # 保存到存储
    if embeddings_result.get("text_units") is not None:
        await context.output_storage.set("text_unit_embeddings", embeddings_result["text_units"])
    if embeddings_result.get("entities") is not None:
        await context.output_storage.set("entity_embeddings", embeddings_result["entities"])
    
    logger.info("工作流完成: generate_embeddings")
    logger.info(f"  - 文本单元嵌入: {len(embeddings_result.get('text_units', []))} 个")
    logger.info(f"  - 实体嵌入: {len(embeddings_result.get('entities', []))} 个")
    
    return WorkflowFunctionOutput(result=embeddings_result)


async def generate_embeddings(
    text_units: pd.DataFrame | None,
    entities: pd.DataFrame | None,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> dict[str, pd.DataFrame]:
    """
    生成文本嵌入。
    
    Args:
        text_units: 文本单元 DataFrame
        entities: 实体 DataFrame
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        dict: 包含嵌入结果的字典
    """
    result = {}
    
    # 生成文本单元嵌入
    if text_units is not None and len(text_units) > 0:
        logger.info(f"开始为 {len(text_units)} 个文本单元生成嵌入")
        text_unit_embeddings = await embed_text_units(
            text_units, config, context
        )
        result["text_units"] = text_unit_embeddings
        logger.info(f"完成文本单元嵌入生成")
    
    # 生成实体嵌入
    if entities is not None and len(entities) > 0:
        logger.info(f"开始为 {len(entities)} 个实体生成嵌入")
        entity_embeddings = await embed_entities(
            entities, config, context
        )
        result["entities"] = entity_embeddings
        logger.info(f"完成实体嵌入生成")
    
    return result


async def embed_text_units(
    text_units: pd.DataFrame,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> pd.DataFrame:
    """
    为文本单元生成嵌入。
    
    Args:
        text_units: 文本单元 DataFrame
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        pd.DataFrame: 包含嵌入的 DataFrame
    """
    embeddings = []
    
    for idx, text_unit in text_units.iterrows():
        text = text_unit['text']
        text_id = text_unit['id']
        
        # 生成嵌入（使用 mock 版本）
        embedding = await generate_mock_embedding(text, context)
        
        embeddings.append({
            'id': text_id,
            'text': text,
            'embedding': embedding,
        })
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已生成 {idx + 1}/{len(text_units)} 个文本单元嵌入")
    
    return pd.DataFrame(embeddings)


async def embed_entities(
    entities: pd.DataFrame,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> pd.DataFrame:
    """
    为实体生成嵌入。
    
    Args:
        entities: 实体 DataFrame
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        pd.DataFrame: 包含嵌入的 DataFrame
    """
    embeddings = []
    
    for idx, entity in entities.iterrows():
        # 组合实体名称和描述作为嵌入文本
        text = f"{entity['name']}: {entity['description']}"
        entity_id = entity['id']
        
        # 生成嵌入（使用 mock 版本）
        embedding = await generate_mock_embedding(text, context)
        
        embeddings.append({
            'id': entity_id,
            'name': entity['name'],
            'text': text,
            'embedding': embedding,
        })
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已生成 {idx + 1}/{len(entities)} 个实体嵌入")
    
    return pd.DataFrame(embeddings)


async def generate_mock_embedding(
    text: str,
    context: PipelineRunContext,
    dimension: int = 1536,
) -> list[float]:
    """
    生成 mock 嵌入向量。

    这是一个简化版本，使用确定性哈希生成伪随机向量。
    在生产环境中，应该使用真实的嵌入模型（如 OpenAI text-embedding-3-small）。

    Args:
        text: 要嵌入的文本
        context: Pipeline 运行上下文
        dimension: 嵌入维度

    Returns:
        list[float]: 嵌入向量
    """
    # 使用文本的哈希值生成确定性的伪随机向量
    hash_bytes = hashlib.sha256(text.encode()).digest()

    # 将哈希值转换为浮点数向量
    embedding = []
    for i in range(dimension):
        # 使用哈希值的不同部分生成不同的数值
        byte_idx = (i * 2) % len(hash_bytes)
        value = int.from_bytes(hash_bytes[byte_idx:byte_idx+2], 'big')
        # 归一化到 [-1, 1]
        normalized = (value / 65535.0) * 2 - 1
        embedding.append(normalized)

    # 归一化向量（使其长度为1）
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


async def generate_openai_embedding(
    text: str,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> list[float]:
    """
    使用 OpenAI API 生成真实嵌入。
    
    这个函数需要配置 OpenAI API 密钥。
    
    Args:
        text: 要嵌入的文本
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        list[float]: 嵌入向量
    """
    # 检查缓存
    cache_key = f"openai_embedding:{hashlib.md5(text.encode()).hexdigest()}"
    cached = context.cache.get(cache_key)
    if cached is not None:
        return cached
    
    # TODO: 实现 OpenAI API 调用
    # 这里需要：
    # 1. 从配置中获取 API 密钥和模型名称
    # 2. 调用 OpenAI Embeddings API
    # 3. 处理错误和重试
    # 4. 缓存结果
    
    # 示例代码（需要安装 openai 库）:
    # import openai
    # openai.api_key = config.language_model.api_key
    # response = await openai.Embedding.acreate(
    #     input=text,
    #     model="text-embedding-3-small"
    # )
    # embedding = response['data'][0]['embedding']
    # context.cache.set(cache_key, embedding)
    # return embedding
    
    raise NotImplementedError(
        "OpenAI embedding not implemented. "
        "Please use generate_mock_embedding for testing."
    )


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    计算两个向量的余弦相似度。
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        float: 余弦相似度 [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def find_similar_texts(
    query_embedding: list[float],
    embeddings_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    查找与查询最相似的文本。
    
    Args:
        query_embedding: 查询嵌入向量
        embeddings_df: 包含嵌入的 DataFrame
        top_k: 返回前 k 个最相似的结果
        
    Returns:
        pd.DataFrame: 最相似的文本及其相似度
    """
    similarities = []
    
    for idx, row in embeddings_df.iterrows():
        similarity = cosine_similarity(query_embedding, row['embedding'])
        similarities.append({
            'id': row['id'],
            'text': row.get('text', row.get('name', '')),
            'similarity': similarity,
        })
    
    # 按相似度排序
    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.nlargest(top_k, 'similarity')
    
    return similarities_df

