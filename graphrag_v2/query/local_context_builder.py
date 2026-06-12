"""Local Search 上下文构建器实现。"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from graphrag_v2.query.context_builder import ContextBuilderResult, LocalContextBuilder

logger = logging.getLogger(__name__)


class EntityRelationshipContextBuilder(LocalContextBuilder):
    """基于实体和关系的上下文构建器。
    
    这个构建器：
    1. 使用查询嵌入找到最相关的实体
    2. 获取这些实体的关系
    3. 获取这些实体所属的社区
    4. 构建包含实体、关系、社区报告的上下文
    """
    
    def __init__(
        self,
        entities: pd.DataFrame | list,
        relationships: pd.DataFrame | list,
        community_reports: pd.DataFrame | list,
        entity_embeddings: pd.DataFrame | None = None,
        communities: pd.DataFrame | list | None = None,
        text_unit_embeddings: pd.DataFrame | None = None,
        top_k_entities: int = 20,
        top_k_relationships: int = 50,
        max_tokens: int = 4000,
    ):
        """初始化上下文构建器。
        
        Args:
            entities: 实体 DataFrame
            relationships: 关系 DataFrame
            community_reports: 社区报告 DataFrame
            entity_embeddings: 实体嵌入 DataFrame
            text_unit_embeddings: 文本单元嵌入 DataFrame（可选）
            top_k_entities: 检索的最相关实体数
            top_k_relationships: 检索的最相关关系数
            max_tokens: 最大 tokens 数
        """
        self.entities = self._to_dataframe(entities)
        self.relationships = self._to_dataframe(relationships)
        self.community_reports = self._to_dataframe(community_reports)
        self.communities = self._to_dataframe(communities or [])
        self.entity_embeddings = (
            entity_embeddings
            if entity_embeddings is not None
            else self._build_default_entity_embeddings(self.entities)
        )
        self.text_unit_embeddings = text_unit_embeddings
        self.top_k_entities = top_k_entities
        self.top_k_relationships = top_k_relationships
        self.max_tokens = max_tokens

    def _to_dataframe(self, value: pd.DataFrame | list) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            df = value.copy()
        else:
            df = pd.DataFrame([vars(item) for item in value])

        if "name" not in df.columns and "title" in df.columns:
            df["name"] = df["title"]
        if "title" not in df.columns and "name" in df.columns:
            df["title"] = df["name"]
        return df

    def _build_default_entity_embeddings(self, entities: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, entity in entities.iterrows():
            name = entity.get("name", entity.get("title", ""))
            embedding = entity.get("description_embedding")
            if embedding is None:
                embedding = self._generate_query_embedding(name).tolist()
            rows.append({"name": name, "embedding": embedding})
        return pd.DataFrame(rows)
    
    def build_context(
        self,
        query: str,
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """构建 Local Search 上下文。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            ContextBuilderResult: 上下文构建结果
        """
        logger.info(f"构建 Local Context，查询: {query}")
        
        # 步骤 1: 生成查询嵌入
        query_embedding = self._generate_query_embedding(query)
        
        # 步骤 2: 找到最相关的实体
        relevant_entities = self._find_relevant_entities(query_embedding)
        logger.info(f"找到 {len(relevant_entities)} 个相关实体")
        
        # 步骤 3: 找到相关的关系
        relevant_relationships = self._find_relevant_relationships(relevant_entities)
        logger.info(f"找到 {len(relevant_relationships)} 个相关关系")
        
        # 步骤 4: 找到相关的社区
        relevant_communities = self._find_relevant_communities(relevant_entities)
        logger.info(f"找到 {len(relevant_communities)} 个相关社区")
        
        # 步骤 5: 构建上下文文本
        context_text = self._build_context_text(
            relevant_entities,
            relevant_relationships,
            relevant_communities,
        )
        
        return ContextBuilderResult(
            context_chunks=context_text,
            context_records={
                "entities": relevant_entities,
                "relationships": relevant_relationships,
                "communities": relevant_communities,
            },
            llm_calls=0,
            prompt_tokens=0,
            output_tokens=0,
        )
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """生成查询嵌入。
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 查询嵌入向量
        """
        # 简化版本：使用确定性哈希生成伪随机向量
        import hashlib
        
        hash_bytes = hashlib.sha256(query.encode()).digest()
        dimension = 1536
        
        embedding = []
        for i in range(dimension):
            byte_idx = (i * 2) % len(hash_bytes)
            value = int.from_bytes(hash_bytes[byte_idx:byte_idx+2], 'big')
            normalized = (value / 65535.0) * 2 - 1
            embedding.append(normalized)
        
        # 归一化
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return np.array(embedding)
    
    def _find_relevant_entities(self, query_embedding: np.ndarray) -> pd.DataFrame:
        """找到最相关的实体。
        
        Args:
            query_embedding: 查询嵌入向量
            
        Returns:
            pd.DataFrame: 相关实体
        """
        # 计算余弦相似度
        similarities = []
        for _, row in self.entity_embeddings.iterrows():
            entity_embedding = np.array(row['embedding'])
            similarity = self._cosine_similarity(query_embedding, entity_embedding)
            similarities.append({
                'name': row['name'],
                'similarity': similarity,
            })
        
        # 排序并取 top-k
        similarities_df = pd.DataFrame(similarities)
        similarities_df = similarities_df.sort_values(by='similarity', ascending=False)
        top_entity_names = similarities_df.head(self.top_k_entities)['name'].tolist()
        
        # 获取完整的实体信息
        relevant_entities = self.entities[self.entities['name'].isin(top_entity_names)]
        return relevant_entities
    
    def _find_relevant_relationships(self, entities: pd.DataFrame) -> pd.DataFrame:
        """找到相关的关系。
        
        Args:
            entities: 实体 DataFrame
            
        Returns:
            pd.DataFrame: 相关关系
        """
        entity_names = entities['name'].tolist()
        
        # 找到涉及这些实体的关系
        relevant_relationships = self.relationships[
            (self.relationships['source'].isin(entity_names)) |
            (self.relationships['target'].isin(entity_names))
        ]
        
        # 按权重排序并限制数量
        relevant_relationships = relevant_relationships.sort_values(
            by='weight',
            ascending=False,
        ).head(self.top_k_relationships)
        
        return relevant_relationships
    
    def _find_relevant_communities(self, entities: pd.DataFrame) -> pd.DataFrame:
        """找到相关的社区。
        
        Args:
            entities: 实体 DataFrame
            
        Returns:
            pd.DataFrame: 相关社区
        """
        # 简化版本：假设实体有 community_id 字段
        # 在实际实现中，需要从社区成员关系中查找
        
        # 这里我们返回排名最高的几个社区
        relevant_communities = self.community_reports.sort_values(
            by='rank',
            ascending=False,
        ).head(5)
        
        return relevant_communities
    
    def _build_context_text(
        self,
        entities: pd.DataFrame,
        relationships: pd.DataFrame,
        communities: pd.DataFrame,
    ) -> str:
        """构建上下文文本。
        
        Args:
            entities: 实体 DataFrame
            relationships: 关系 DataFrame
            communities: 社区 DataFrame
            
        Returns:
            str: 上下文文本
        """
        context_parts = []
        
        # 添加实体信息
        context_parts.append("## 相关实体\n")
        for _, entity in entities.head(10).iterrows():
            context_parts.append(
                f"- **{entity['name']}** ({entity['type']}): {entity.get('description', 'N/A')}"
            )
        
        # 添加关系信息
        context_parts.append("\n## 相关关系\n")
        for _, rel in relationships.head(20).iterrows():
            context_parts.append(
                f"- {rel['source']} -> {rel['target']}: {rel.get('description', 'N/A')} (权重: {rel['weight']})"
            )
        
        # 添加社区信息
        context_parts.append("\n## 相关社区\n")
        for _, comm in communities.head(3).iterrows():
            context_parts.append(f"\n### {comm['title']}\n")
            context_parts.append(f"{comm['summary']}\n")
        
        return "\n".join(context_parts)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度。
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            float: 余弦相似度
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
