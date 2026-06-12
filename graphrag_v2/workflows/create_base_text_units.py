"""创建基础文本单元工作流。

将文档分块为文本单元。
"""

import hashlib
import logging

import pandas as pd
import tiktoken

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext
from graphrag_v2.pipeline.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_model: str = "cl100k_base",
) -> list[tuple[str, int]]:
    """将文本分块。
    
    Args:
        text: 要分块的文本
        chunk_size: 每个块的最大 token 数
        chunk_overlap: 块之间的重叠 token 数
        encoding_model: 编码模型名称
        
    Returns:
        块列表，每个块是 (文本, token数) 的元组
    """
    # 获取编码器
    try:
        encoding = tiktoken.get_encoding(encoding_model)
    except Exception:
        # 如果编码模型不存在，使用默认的
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # 编码文本
    tokens = encoding.encode(text)
    
    # 分块
    chunks = []
    start = 0
    
    while start < len(tokens):
        # 计算结束位置
        end = min(start + chunk_size, len(tokens))
        
        # 提取块的 tokens
        chunk_tokens = tokens[start:end]
        
        # 解码为文本
        chunk_text = encoding.decode(chunk_tokens)
        
        # 添加到结果
        chunks.append((chunk_text, len(chunk_tokens)))
        
        # 移动到下一个块（考虑重叠）
        if end >= len(tokens):
            break
        start = end - chunk_overlap
    
    return chunks


def generate_text_unit_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    """生成文本单元 ID。
    
    Args:
        document_id: 文档 ID
        chunk_index: 块索引
        chunk_text: 块文本
        
    Returns:
        文本单元 ID
    """
    # 使用文档 ID、索引和文本内容的哈希生成唯一 ID
    content = f"{document_id}_{chunk_index}_{chunk_text[:100]}"
    hash_value = hashlib.sha256(content.encode()).hexdigest()
    return f"text_unit_{hash_value[:16]}"


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """创建基础文本单元。
    
    将文档分块为文本单元。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        工作流输出，包含文本单元
    """
    logger.info("工作流开始: create_base_text_units")
    
    # 从存储加载文档
    documents = await context.output_storage.get("documents")
    if documents is None:
        logger.error("未找到文档数据")
        return WorkflowFunctionOutput(result=None, stop=True)

    if not isinstance(documents, pd.DataFrame):
        documents = pd.DataFrame([vars(document) for document in documents])
    
    # 获取分块配置
    chunk_size = config.chunks.size
    chunk_overlap = config.chunks.overlap
    encoding_model = config.chunks.encoding_model
    
    logger.info(f"分块配置: size={chunk_size}, overlap={chunk_overlap}, encoding={encoding_model}")
    
    # 处理每个文档
    text_units = []
    
    for idx, row in documents.iterrows():
        document_id = row["id"]
        text = row["text"]
        
        # 分块
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap, encoding_model)
        
        # 为每个块创建文本单元
        for chunk_idx, (chunk_text, n_tokens) in enumerate(chunks):
            text_unit_id = generate_text_unit_id(document_id, chunk_idx, chunk_text)
            
            text_units.append({
                "id": text_unit_id,
                "text": chunk_text,
                "n_tokens": n_tokens,
                "document_ids": [document_id],
                "chunk_index": chunk_idx,
            })
        
        logger.info(f"文档 {document_id} 分块完成: {len(chunks)} 个块")
    
    # 转换为 DataFrame
    df = pd.DataFrame(text_units)
    df.attrs["name"] = "text_units"
    
    # 保存到输出存储
    await context.output_storage.set("text_units", df)
    
    # 更新统计信息
    context.stats.num_text_units = len(df)
    
    logger.info(f"工作流完成: create_base_text_units (生成了 {len(df)} 个文本单元)")
    return WorkflowFunctionOutput(result=df)
