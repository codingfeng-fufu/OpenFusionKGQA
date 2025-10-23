"""加载文档工作流。

从输入存储加载文档。
"""

import logging
from pathlib import Path

import pandas as pd

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext
from graphrag_v2.pipeline.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """加载输入文档。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        工作流输出，包含加载的文档
    """
    logger.info("工作流开始: load_documents")
    
    # 从输入存储加载文档
    input_dir = Path(config.input.base_dir)
    documents = []
    
    # 支持的文件类型
    file_types = config.input.file_type
    if isinstance(file_types, str):
        file_types = [file_types]
    
    # 遍历输入目录
    for file_type in file_types:
        pattern = f"*.{file_type}"
        for file_path in input_dir.glob(pattern):
            try:
                # 读取文件内容
                with open(file_path, "r", encoding=config.input.encoding) as f:
                    text = f.read()
                
                # 创建文档记录
                documents.append({
                    "id": file_path.stem,  # 使用文件名（不含扩展名）作为 ID
                    "title": file_path.name,
                    "text": text,
                    "source": str(file_path),
                })
                
                logger.info(f"已加载文档: {file_path.name}")
            except Exception as e:
                logger.error(f"加载文档失败 {file_path}: {e}")
    
    # 转换为 DataFrame
    df = pd.DataFrame(documents)
    
    # 保存到输出存储
    await context.output_storage.set("documents", df)
    
    # 更新统计信息
    context.stats.num_documents = len(df)
    
    logger.info(f"工作流完成: load_documents (加载了 {len(df)} 个文档)")
    return WorkflowFunctionOutput(result=df)

