"""
接口定义文件 - 确保并行开发的关键
=====================================

⭐ 重要提示：
1. 这个文件定义了所有模块的接口
2. 双方必须严格遵守这些接口
3. 修改接口前必须提前沟通
4. 按照接口开发，可以互不依赖、并行工作

版本: v1.0
最后更新: 2025-10-23
"""

from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod


# ============================================================================
# 成员B负责实现：二部图构建器接口
# ============================================================================

class BipartiteGraphInterface(ABC):
    """
    二部图构建器接口 - 成员B负责实现
    
    功能：为非核心文本块构建轻量级的文本-关键词索引
    输入：文本块列表
    输出：可检索的二部图结构
    """
    
    @abstractmethod
    def build(self, text_chunks: List[str]) -> 'BipartiteGraphInterface':
        """
        从文本块构建二部图
        
        Args:
            text_chunks: 文本块列表
            例如: ["知识图谱是...", "图神经网络...", ...]
            
        Returns:
            self: 构建完成的二部图对象
            
        示例:
            >>> builder = BipartiteGraphBuilder(config)
            >>> builder.build(non_core_chunks)
            >>> print(f"构建完成，包含{len(builder.text_nodes)}个文本节点")
        """
        pass
    
    @abstractmethod
    def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
        """
        基于关键词检索相关文本块
        
        Args:
            query_keywords: 查询关键词列表
            例如: ["知识图谱", "实体", "关系"]
            top_k: 返回top-k个最相关的文本块
            
        Returns:
            检索结果列表，每个结果是一个字典：
            [
                {
                    'chunk_id': int,           # 文本块ID（对应原始列表的索引）
                    'text': str,               # 文本内容
                    'score': float,            # 匹配分数（0-1之间）
                    'matched_keywords': List[str]  # 匹配到的关键词
                },
                ...
            ]
            
        示例:
            >>> results = bipartite_graph.search(['知识图谱', '实体'], top_k=5)
            >>> for r in results:
            ...     print(f"Chunk {r['chunk_id']}: score={r['score']:.3f}")
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        保存二部图到文件
        
        Args:
            filepath: 保存路径，例如 'data/bipartite_graph.pkl'
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        从文件加载二部图
        
        Args:
            filepath: 文件路径
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取二部图的统计信息
        
        Returns:
            统计信息字典：
            {
                'num_text_nodes': int,      # 文本节点数量
                'num_keyword_nodes': int,   # 关键词节点数量
                'num_edges': int,           # 边数量
                'avg_keywords_per_text': float,  # 平均每个文本的关键词数
                'top_keywords': List[Tuple[str, int]]  # 高频关键词
            }
        """
        pass


# ============================================================================
# 成员A负责实现：核心块选择器接口
# ============================================================================

class CoreChunkSelectorInterface(ABC):
    """
    核心块选择器接口 - 成员A负责实现
    
    功能：从所有文本块中选择核心块（用于LLM提取）
    """
    
    @abstractmethod
    def select(self, chunks: List[str], ratio: float = 0.2) -> List[int]:
        """
        选择核心文本块
        
        Args:
            chunks: 所有文本块列表
            ratio: 核心块占比（0.1-0.5）
            
        Returns:
            核心块的索引列表
            例如: [0, 5, 12, 18, ...] 表示第0、5、12、18个chunk是核心块
            
        示例:
            >>> selector = CoreChunkSelector(method='pagerank')
            >>> core_indices = selector.select(all_chunks, ratio=0.2)
            >>> core_chunks = [all_chunks[i] for i in core_indices]
        """
        pass


# ============================================================================
# 成员A负责实现：骨架图构建器接口
# ============================================================================

class SkeletonGraphInterface(ABC):
    """
    骨架图构建器接口 - 成员A负责实现
    
    功能：为核心块构建高质量的实体-关系知识图谱
    """
    
    @abstractmethod
    def build(self, core_chunks: List[str]) -> Any:
        """
        从核心块构建骨架知识图谱
        
        Args:
            core_chunks: 核心文本块列表
            
        Returns:
            知识图谱对象（通常是 networkx.Graph）
        """
        pass
    
    @abstractmethod
    def local_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        在骨架图上进行局部检索
        
        Args:
            query: 查询文本
            k: 返回top-k结果
            
        Returns:
            检索结果列表
        """
        pass


# ============================================================================
# 成员A负责实现：混合检索器接口
# ============================================================================

class HybridRetrieverInterface(ABC):
    """
    混合检索器接口 - 成员A负责实现
    
    功能：同时从骨架图和二部图检索，合并结果
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            检索结果列表，格式与BipartiteGraphInterface.search()一致
        """
        pass


# ============================================================================
# 数据格式定义（双方共同遵守）
# ============================================================================

class DataFormats:
    """
    数据格式定义 - 确保数据交换的一致性
    """
    
    # 文档格式
    DOCUMENT_FORMAT = {
        'doc_id': str,          # 文档ID
        'title': str,           # 文档标题
        'content': str,         # 文档内容
        'metadata': dict        # 元数据（可选）
    }
    
    # 文本块格式
    CHUNK_FORMAT = {
        'chunk_id': int,        # 块ID
        'text': str,            # 文本内容
        'doc_id': str,          # 所属文档ID
        'position': int         # 在文档中的位置
    }
    
    # 查询格式
    QUERY_FORMAT = {
        'query_id': str,        # 查询ID
        'question': str,        # 问题文本
        'expected_answer': str  # 标准答案（用于评估）
    }
    
    # 检索结果格式（统一格式）
    RETRIEVAL_RESULT_FORMAT = {
        'chunk_id': int,                # 文本块ID
        'text': str,                    # 文本内容
        'score': float,                 # 相关性分数（0-1）
        'source': str,                  # 来源（'skeleton' 或 'bipartite'）
        'matched_keywords': List[str]   # 匹配的关键词（可选）
    }


# ============================================================================
# 配置格式定义
# ============================================================================

class ConfigFormats:
    """配置文件格式定义"""
    
    # 二部图配置
    BIPARTITE_CONFIG = {
        'top_k_keywords': 10,       # 每个文本提取的关键词数量
        'tfidf_weight': 0.4,        # TF-IDF方法的权重
        'textrank_weight': 0.4,     # TextRank方法的权重
        'ner_weight': 0.2,          # NER方法的权重
        'min_keyword_length': 2,    # 最小关键词长度
        'use_ner': True            # 是否使用NER
    }
    
    # 核心块选择配置
    CORE_SELECTOR_CONFIG = {
        'method': 'pagerank',       # 'random' 或 'pagerank'
        'ratio': 0.2,               # 核心块比例
        'similarity_threshold': 0.5  # PageRank相似度阈值
    }
    
    # 混合检索配置
    HYBRID_RETRIEVER_CONFIG = {
        'skeleton_weight': 0.6,     # 骨架图权重
        'bipartite_weight': 0.4,    # 二部图权重
        'top_k': 10                 # 返回结果数量
    }


# ============================================================================
# Mock实现（用于测试和并行开发）
# ============================================================================

class MockBipartiteGraph(BipartiteGraphInterface):
    """
    Mock二部图实现 - 成员A可以用这个进行测试
    
    成员A在等待成员B实现真正的二部图时，可以用这个Mock版本进行开发和测试
    """
    
    def __init__(self, config=None):
        self.text_nodes = []
        self.built = False
    
    def build(self, text_chunks: List[str]) -> 'MockBipartiteGraph':
        """Mock实现：直接存储文本块"""
        self.text_nodes = [
            {'chunk_id': i, 'text': chunk} 
            for i, chunk in enumerate(text_chunks)
        ]
        self.built = True
        print(f"[Mock] 二部图构建完成，包含{len(self.text_nodes)}个文本节点")
        return self
    
    def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
        """Mock实现：简单的关键词匹配"""
        if not self.built:
            raise RuntimeError("请先调用build()方法")
        
        results = []
        for node in self.text_nodes:
            # 简单的关键词匹配
            score = sum(1 for kw in query_keywords if kw in node['text'])
            if score > 0:
                results.append({
                    'chunk_id': node['chunk_id'],
                    'text': node['text'],
                    'score': score / len(query_keywords),
                    'matched_keywords': [kw for kw in query_keywords if kw in node['text']]
                })
        
        # 排序并返回top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def save(self, filepath: str) -> None:
        """Mock实现"""
        print(f"[Mock] 保存到 {filepath}")
    
    def load(self, filepath: str) -> None:
        """Mock实现"""
        print(f"[Mock] 从 {filepath} 加载")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Mock实现"""
        return {
            'num_text_nodes': len(self.text_nodes),
            'num_keyword_nodes': 0,
            'num_edges': 0,
            'avg_keywords_per_text': 0.0,
            'top_keywords': []
        }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    使用示例 - 展示如何使用接口进行并行开发
    """
    
    print("=" * 60)
    print("接口使用示例")
    print("=" * 60)
    
    # 示例数据
    all_chunks = [
        "知识图谱是一种结构化的语义知识库",
        "图神经网络可以处理非欧几里得数据",
        "检索增强生成结合了检索和生成两种方法",
        "大语言模型在自然语言处理任务中表现出色",
        "向量数据库用于存储和检索高维向量"
    ]
    
    print(f"\n总共 {len(all_chunks)} 个文本块")
    
    # 成员A可以用Mock版本测试自己的代码
    print("\n--- 成员A使用Mock二部图测试 ---")
    mock_bg = MockBipartiteGraph()
    mock_bg.build(all_chunks[:3])  # 模拟非核心块
    
    results = mock_bg.search(['知识图谱', '检索'], top_k=2)
    print(f"\n检索到 {len(results)} 个结果:")
    for r in results:
        print(f"  - Chunk {r['chunk_id']}: score={r['score']:.2f}")
        print(f"    匹配关键词: {r['matched_keywords']}")
    
    # 成员B实现真正的二部图后，只需替换即可
    print("\n--- 成员B实现后，无缝替换 ---")
    print("from src.bipartite_graph import BipartiteGraphBuilder")
    print("real_bg = BipartiteGraphBuilder(config)")
    print("real_bg.build(non_core_chunks)  # 接口完全相同！")
    
    print("\n✅ 接口设计完成，可以并行开发了！")
