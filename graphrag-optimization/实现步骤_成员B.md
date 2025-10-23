# 实现步骤 - 成员B

> **这是一个建议的实现顺序，你可以按自己的节奏完成**

---

## 📋 总体流程

```
步骤1: 关键词提取器
   ↓
步骤2: 二部图构建器
   ↓
步骤3: 检索功能
   ↓
步骤4: 集成测试（与成员A）
   ↓
步骤5: 实验和可视化
```

---

## 步骤1: 实现关键词提取器

### 文件
`src/bipartite_graph/keyword_extractors.py`

### 需要实现的类

#### 1.1 TfidfKeywordExtractor
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

class TfidfKeywordExtractor:
    """TF-IDF关键词提取器"""
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
    
    def extract(self, texts: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        从文本列表中提取关键词
        
        Args:
            texts: 文本列表
            top_k: 每个文本提取的关键词数量
            
        Returns:
            每个文本的关键词列表 [[(keyword, score), ...], ...]
        """
        # TODO: 实现这个方法
        pass
```

**实现提示**:
1. 使用 `self.vectorizer.fit_transform(texts)` 训练模型
2. 获取特征名 `self.vectorizer.get_feature_names_out()`
3. 对每个文本，找到TF-IDF分数最高的top_k个词

#### 1.2 TextRankExtractor
```python
import jieba.analyse

class TextRankExtractor:
    """TextRank关键词提取器"""
    
    def extract(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用TextRank提取关键词
        
        Args:
            text: 单个文本
            top_k: 提取数量
            
        Returns:
            关键词及其分数列表 [(keyword, score), ...]
        """
        keywords = jieba.analyse.textrank(
            text,
            topK=top_k,
            withWeight=True
        )
        return keywords
```

### 测试
```python
# 在文件末尾添加测试代码
if __name__ == '__main__':
    test_texts = [
        "知识图谱是一种结构化的语义知识库",
        "图神经网络可以处理非欧几里得数据",
        "检索增强生成结合了检索和生成两种方法"
    ]
    
    # 测试TF-IDF
    tfidf = TfidfKeywordExtractor()
    results = tfidf.extract(test_texts, top_k=5)
    print("TF-IDF结果:")
    for i, keywords in enumerate(results):
        print(f"文本{i+1}: {keywords}")
    
    # 测试TextRank
    textrank = TextRankExtractor()
    for i, text in enumerate(test_texts):
        keywords = textrank.extract(text, top_k=5)
        print(f"文本{i+1}: {keywords}")
```

---

## 步骤2: 实现二部图构建器

### 文件
`src/bipartite_graph/builder.py`

### 核心类
```python
from typing import List, Dict, Tuple
from src.interfaces import BipartiteGraphInterface
from src.bipartite_graph.keyword_extractors import TfidfKeywordExtractor, TextRankExtractor

class BipartiteGraphBuilder(BipartiteGraphInterface):
    """二部图构建器"""
    
    def __init__(self, config: Dict):
        """
        初始化
        
        Args:
            config: 配置字典，例如:
            {
                'top_k_keywords': 10,
                'tfidf_weight': 0.5,
                'textrank_weight': 0.5
            }
        """
        self.config = config
        
        # 图结构
        self.text_nodes = []      # [{'chunk_id': 0, 'text': '...', 'keywords': [...]}, ...]
        self.keyword_nodes = {}   # {'知识图谱': {'count': 5, 'text_ids': [0,2,5]}, ...}
        self.edges = []           # [(chunk_id, keyword, weight), ...]
        self.inverted_index = {}  # {'知识图谱': [(0, 0.85), (2, 0.72)], ...}
        
        # 初始化提取器
        self.tfidf_extractor = TfidfKeywordExtractor()
        self.textrank_extractor = TextRankExtractor()
    
    def build(self, text_chunks: List[str]) -> 'BipartiteGraphBuilder':
        """构建二部图"""
        # TODO: 实现
        pass
    
    def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
        """检索"""
        # TODO: 实现
        pass
    
    def save(self, filepath: str) -> None:
        """保存"""
        # TODO: 实现
        pass
    
    def load(self, filepath: str) -> None:
        """加载"""
        # TODO: 实现
        pass
    
    def get_statistics(self) -> Dict:
        """统计信息"""
        # TODO: 实现
        pass
```

### 实现顺序

#### 2.1 实现 `build()` 方法
```python
def build(self, text_chunks: List[str]) -> 'BipartiteGraphBuilder':
    """构建二部图"""
    print(f"开始构建二部图，处理 {len(text_chunks)} 个文本块...")
    
    # 步骤1: 批量提取TF-IDF关键词
    tfidf_results = self.tfidf_extractor.extract(
        text_chunks, 
        top_k=self.config['top_k_keywords']
    )
    
    # 步骤2: 对每个文本处理
    for i, chunk_text in enumerate(text_chunks):
        # 获取TF-IDF关键词
        tfidf_kws = tfidf_results[i]
        
        # 获取TextRank关键词
        textrank_kws = self.textrank_extractor.extract(
            chunk_text,
            top_k=self.config['top_k_keywords']
        )
        
        # 合并关键词
        final_keywords = self._merge_keywords(tfidf_kws, textrank_kws)
        
        # 添加到图中
        self._add_to_graph(i, chunk_text, final_keywords)
    
    # 步骤3: 构建倒排索引
    self._build_inverted_index()
    
    print(f"✅ 二部图构建完成")
    return self
```

#### 2.2 实现 `_merge_keywords()` 方法
```python
def _merge_keywords(
    self, 
    tfidf_kws: List[Tuple[str, float]], 
    textrank_kws: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """合并多种方法的关键词"""
    scores = {}
    
    # TF-IDF的权重
    w_tfidf = self.config['tfidf_weight']
    for keyword, score in tfidf_kws:
        scores[keyword] = scores.get(keyword, 0) + w_tfidf * score
    
    # TextRank的权重
    w_textrank = self.config['textrank_weight']
    for keyword, score in textrank_kws:
        scores[keyword] = scores.get(keyword, 0) + w_textrank * score
    
    # 排序并返回top_k
    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:self.config['top_k_keywords']]
```

#### 2.3 实现 `_add_to_graph()` 方法
```python
def _add_to_graph(
    self, 
    chunk_id: int, 
    text: str, 
    keywords: List[Tuple[str, float]]
):
    """添加节点和边到图中"""
    # 添加文本节点
    self.text_nodes.append({
        'chunk_id': chunk_id,
        'text': text,
        'keywords': [kw for kw, score in keywords]
    })
    
    # 添加关键词节点和边
    for keyword, score in keywords:
        # 如果关键词节点不存在，创建它
        if keyword not in self.keyword_nodes:
            self.keyword_nodes[keyword] = {
                'count': 0,
                'text_ids': []
            }
        
        # 更新关键词节点
        self.keyword_nodes[keyword]['count'] += 1
        self.keyword_nodes[keyword]['text_ids'].append(chunk_id)
        
        # 添加边
        self.edges.append((chunk_id, keyword, score))
```

#### 2.4 实现 `_build_inverted_index()` 方法
```python
def _build_inverted_index(self):
    """构建倒排索引"""
    self.inverted_index = {}
    
    for chunk_id, keyword, weight in self.edges:
        if keyword not in self.inverted_index:
            self.inverted_index[keyword] = []
        self.inverted_index[keyword].append((chunk_id, weight))
```

---

## 步骤3: 实现检索功能

### 实现 `search()` 方法
```python
def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
    """基于关键词检索文本块"""
    if not self.text_nodes:
        raise RuntimeError("请先调用build()构建二部图")
    
    # 记录每个文本的得分
    text_scores = {}
    matched_keywords_per_text = {}
    
    # 对每个查询关键词
    for keyword in query_keywords:
        if keyword in self.inverted_index:
            # 从倒排索引中获取包含该关键词的文本
            for chunk_id, weight in self.inverted_index[keyword]:
                # 累加分数
                text_scores[chunk_id] = text_scores.get(chunk_id, 0) + weight
                
                # 记录匹配的关键词
                if chunk_id not in matched_keywords_per_text:
                    matched_keywords_per_text[chunk_id] = []
                matched_keywords_per_text[chunk_id].append(keyword)
    
    # 如果没有匹配，返回空
    if not text_scores:
        return []
    
    # 排序
    sorted_results = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 构建返回结果
    results = []
    for chunk_id, score in sorted_results[:top_k]:
        results.append({
            'chunk_id': chunk_id,
            'text': self.text_nodes[chunk_id]['text'],
            'score': score,
            'matched_keywords': matched_keywords_per_text[chunk_id]
        })
    
    return results
```

---

## 步骤4: 实现保存/加载和统计

### save() 和 load()
```python
def save(self, filepath: str):
    """保存到文件"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump({
            'text_nodes': self.text_nodes,
            'keyword_nodes': self.keyword_nodes,
            'edges': self.edges,
            'inverted_index': self.inverted_index
        }, f)
    print(f"✅ 已保存到: {filepath}")

def load(self, filepath: str):
    """从文件加载"""
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        self.text_nodes = data['text_nodes']
        self.keyword_nodes = data['keyword_nodes']
        self.edges = data['edges']
        self.inverted_index = data['inverted_index']
    print(f"✅ 已加载: {filepath}")
```

### get_statistics()
```python
def get_statistics(self) -> Dict:
    """获取统计信息"""
    top_keywords = sorted(
        self.keyword_nodes.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:10]
    
    return {
        'num_text_nodes': len(self.text_nodes),
        'num_keyword_nodes': len(self.keyword_nodes),
        'num_edges': len(self.edges),
        'avg_keywords_per_text': len(self.edges) / len(self.text_nodes) if self.text_nodes else 0,
        'top_keywords': [(kw, info['count']) for kw, info in top_keywords]
    }
```

---

## 步骤5: 测试

### 创建测试文件
`tests/test_bipartite.py`

```python
import pytest
from src.bipartite_graph.builder import BipartiteGraphBuilder

def test_build():
    """测试构建功能"""
    config = {
        'top_k_keywords': 5,
        'tfidf_weight': 0.5,
        'textrank_weight': 0.5
    }
    
    test_chunks = [
        "知识图谱是一种结构化的语义知识库",
        "图神经网络可以处理非欧几里得数据",
        "检索增强生成结合了检索和生成两种方法"
    ]
    
    builder = BipartiteGraphBuilder(config)
    builder.build(test_chunks)
    
    assert len(builder.text_nodes) == 3
    assert len(builder.keyword_nodes) > 0
    print("✅ 构建测试通过")

def test_search():
    """测试检索功能"""
    config = {'top_k_keywords': 5, 'tfidf_weight': 0.5, 'textrank_weight': 0.5}
    test_chunks = ["知识图谱...", "图神经网络...", "检索增强生成..."]
    
    builder = BipartiteGraphBuilder(config)
    builder.build(test_chunks)
    
    results = builder.search(['知识图谱'], top_k=2)
    
    assert len(results) > 0
    assert 'chunk_id' in results[0]
    assert 'score' in results[0]
    print("✅ 检索测试通过")
```

---

## 步骤6: 实验和可视化（后续完成）

这部分在二部图构建器完成后再做，包括：
- 4个专项实验
- 数据准备
- 结果可视化

详细内容参考 `docs/member_B_tasks.md`

---

## ✅ 完成标准

当你完成以下内容，就可以和我进行集成测试了：

- [ ] `keyword_extractors.py` 实现并测试通过
- [ ] `builder.py` 实现并测试通过
- [ ] `build()` 方法能正常构建二部图
- [ ] `search()` 方法能正常检索
- [ ] `save()` 和 `load()` 能正常工作
- [ ] 所有方法都有清晰的注释

---

**提示**: 按照这个顺序实现，每完成一步就测试一下，确保功能正确再继续下一步。

