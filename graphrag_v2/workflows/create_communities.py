"""创建社区工作流。

使用社区检测算法从图谱中发现实体社区。
"""

import logging

import networkx as nx
import pandas as pd

from graphrag_v2.config.models.graph_rag_config import GraphRagConfig
from graphrag_v2.pipeline.context import PipelineRunContext
from graphrag_v2.pipeline.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


def create_graph_from_dataframes(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
) -> nx.Graph:
    """从实体和关系 DataFrame 创建 NetworkX 图。
    
    Args:
        entities_df: 实体 DataFrame
        relationships_df: 关系 DataFrame
        
    Returns:
        NetworkX 图
    """
    graph = nx.Graph()
    
    # 添加节点（实体）
    for _, entity in entities_df.iterrows():
        graph.add_node(
            entity['name'],
            id=entity['id'],
            type=entity.get('type', 'ENTITY'),
            description=entity.get('description', ''),
            rank=entity.get('rank', 0),
        )
    
    # 添加边（关系）
    for _, rel in relationships_df.iterrows():
        source = rel['source']
        target = rel['target']
        
        # 只添加两个节点都存在的边
        if source in graph.nodes and target in graph.nodes:
            graph.add_edge(
                source,
                target,
                id=rel['id'],
                weight=rel.get('weight', 1.0),
                description=rel.get('description', ''),
            )
    
    return graph


def detect_communities(
    graph: nx.Graph,
    max_level: int = 3,
) -> list[dict]:
    """使用 Louvain 算法检测社区。
    
    Args:
        graph: NetworkX 图
        max_level: 最大层级
        
    Returns:
        社区列表
    """
    import networkx.algorithms.community as nx_comm
    
    communities = []
    
    # 检查图是否为空
    if len(graph.nodes) == 0:
        logger.warning("图为空，无法检测社区")
        return communities
    
    # 检查图是否连通
    if not nx.is_connected(graph):
        logger.info("图不连通，将分别处理每个连通分量")
        # 获取所有连通分量
        connected_components = list(nx.connected_components(graph))
        logger.info(f"发现 {len(connected_components)} 个连通分量")
    else:
        connected_components = [set(graph.nodes)]
    
    community_id = 0
    
    # 对每个连通分量进行社区检测
    for component_idx, component in enumerate(connected_components):
        # 创建子图
        subgraph = graph.subgraph(component).copy()
        
        if len(subgraph.nodes) < 2:
            # 单节点社区
            node = list(subgraph.nodes)[0]
            communities.append({
                'id': f"community_{community_id}",
                'level': 0,
                'title': f"社区 {community_id}",
                'nodes': [node],
                'size': 1,
                'parent_community': None,
            })
            community_id += 1
            continue
        
        # 使用 Louvain 算法检测社区
        try:
            # Louvain 算法
            partition = nx_comm.louvain_communities(
                subgraph,
                weight='weight',
                seed=42,
            )
            
            logger.info(f"连通分量 {component_idx + 1}: 发现 {len(partition)} 个社区")
            
            # 创建社区记录
            for comm_nodes in partition:
                nodes_list = list(comm_nodes)
                communities.append({
                    'id': f"community_{community_id}",
                    'level': 0,
                    'title': f"社区 {community_id}",
                    'nodes': nodes_list,
                    'size': len(nodes_list),
                    'parent_community': None,
                })
                community_id += 1
                
        except Exception as e:
            logger.error(f"社区检测失败: {e}")
            # 将整个连通分量作为一个社区
            communities.append({
                'id': f"community_{community_id}",
                'level': 0,
                'title': f"社区 {community_id}",
                'nodes': list(component),
                'size': len(component),
                'parent_community': None,
            })
            community_id += 1
    
    return communities


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """创建社区工作流。
    
    从图谱中检测实体社区。
    
    Args:
        config: GraphRAG 配置
        context: Pipeline 运行上下文
        
    Returns:
        工作流输出，包含社区信息
    """
    logger.info("工作流开始: create_communities")
    
    # 从存储加载实体和关系
    entities_df = await context.output_storage.get("entities")
    relationships_df = await context.output_storage.get("relationships")
    
    if entities_df is None or relationships_df is None:
        logger.error("未找到实体或关系数据")
        return WorkflowFunctionOutput(result=None, stop=True)
    
    logger.info(f"加载了 {len(entities_df)} 个实体和 {len(relationships_df)} 个关系")
    
    # 创建图
    graph = create_graph_from_dataframes(entities_df, relationships_df)
    logger.info(f"创建图: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
    
    # 检测社区
    communities = detect_communities(graph)
    logger.info(f"检测到 {len(communities)} 个社区")
    
    # 转换为 DataFrame
    communities_df = pd.DataFrame(communities)
    
    # 计算社区统计信息
    if len(communities_df) > 0:
        # 添加社区的边数
        for idx, row in communities_df.iterrows():
            nodes = row['nodes']
            subgraph = graph.subgraph(nodes)
            communities_df.at[idx, 'num_edges'] = len(subgraph.edges)
            
            # 计算社区的平均度
            if len(nodes) > 0:
                degrees = [graph.degree(node) for node in nodes]
                communities_df.at[idx, 'avg_degree'] = sum(degrees) / len(degrees)
            else:
                communities_df.at[idx, 'avg_degree'] = 0.0
    
    # 保存到输出存储
    await context.output_storage.set("communities", communities_df)
    await context.output_storage.set("graph", graph)
    
    # 更新统计信息
    context.stats.num_communities = len(communities_df)
    
    logger.info(f"工作流完成: create_communities")
    logger.info(f"  - 检测到 {len(communities_df)} 个社区")
    
    # 打印社区统计
    if len(communities_df) > 0:
        logger.info("社区统计:")
        logger.info(f"  - 平均社区大小: {communities_df['size'].mean():.2f}")
        logger.info(f"  - 最大社区大小: {communities_df['size'].max()}")
        logger.info(f"  - 最小社区大小: {communities_df['size'].min()}")
    
    return WorkflowFunctionOutput(
        result={
            'communities': communities_df,
            'graph': graph,
        }
    )

