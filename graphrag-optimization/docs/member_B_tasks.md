# 成员B任务清单 - 二部图构建器开发指南

> **写给同学B**: 这是你的完整任务说明，请仔细阅读。有任何问题随时找我（成员A）！

---

## 📋 任务概览

**你的角色**: 二部图模块负责人
**核心任务**: 实现KET-RAG的轻量级索引组件
**工作量**: 约5-6周，占项目30%
**技术难度**: ⭐⭐⭐（中等，主要是传统NLP方法）

### 你负责的模块

| 模块 | 时间 | 难度 | 说明 |
|-----|------|------|------|
| **二部图构建器** | 2-3周 | ⭐⭐⭐ | 核心技术模块 |
| **二部图专项实验** | 1-1.5周 | ⭐⭐ | 展示模块效果 |
| **数据准备** | 1周 | ⭐ | 测试数据集 |
| **结果可视化** | 1周 | ⭐⭐ | 实验结果展示 |

---

## 🎯 你的模块在整个系统中的位置

### 整体架构

```
原始GraphRAG问题: 所有文本都用LLM提取 → 成本太高💰💰💰

KET-RAG解决方案:
┌─────────────────────────────────────────┐
│  1000个文本块                            │
└─────────────────────────────────────────┘
         │
         ├─→ 核心块选择器 [成员A负责]
         │
    ┌────┴────┐
    │         │
核心块(200)  非核心块(800) ← 你负责这部分！
    │         │
    ↓         ↓
骨架图      二部图 ← 你的模块
(LLM提取)   (传统NLP)
💰💰💰      ✅省钱
    │         │
    └────┬────┘
         ↓
    混合检索器 [成员A负责]
         ↓
      最终答案
```

### 你的创新点

**用传统NLP方法替代昂贵的LLM调用！**
- TF-IDF: 统计方法提取关键词
- TextRank: 图算法提取关键词
- 二部图: 文本-关键词索引结构
- 倒排索引: 快速检索

**成本对比**:
- 骨架图（LLM）: 200块 × $0.01/块 = $2.00
- 二部图（你的）: 800块 × $0 = $0.00 ✅
- **总成本降低80%！**

---

## 📅 时间规划（6周）

### Week 1: 环境搭建 + 关键词提取 ⭐ 第一周最重要
**目标**: 完成TF-IDF和TextRank关键词提取

**Day 1-2: 环境搭建**
- [ ] 安装Python环境（Python 3.8+）
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 阅读文档:
  - [ ] `README.md` - 项目概览
  - [ ] `src/interfaces.py` - **必读！接口定义**
  - [ ] 本文档 - 你的任务清单
- [ ] 运行接口演示: `python src/interfaces.py`

**Day 3-4: 实现TF-IDF关键词提取**
- [ ] 创建文件: `src/bipartite_graph/keyword_extractors.py`
- [ ] 实现 `TfidfKeywordExtractor` 类
- [ ] 测试: `pytest tests/test_bipartite.py::test_tfidf_extraction -v`

**Day 5-6: 实现TextRank关键词提取**
- [ ] 在同一文件中实现 `TextRankExtractor` 类
- [ ] 测试: `pytest tests/test_bipartite.py::test_textrank_extraction -v`

**Day 7: 整合测试**
- [ ] 运行所有关键词提取测试
- [ ] 优化参数（top_k, ngram_range等）
- [ ] **周五下午3点**: 与成员A同步进度

**检查点**:
```bash
pytest tests/test_bipartite.py -k "keyword" -v
```

---

### Week 2: 二部图构建 ⭐ 核心功能
**目标**: 完成二部图的构建功能

**Day 1-2: 实现build()主流程**
- [ ] 创建文件: `src/bipartite_graph/builder.py`
- [ ] 实现 `BipartiteGraphBuilder` 类框架
- [ ] 实现 `build()` 方法的主流程

**Day 3-4: 实现关键词合并和图构建**
- [ ] 实现 `_merge_keywords()` - 合并多种方法的关键词
- [ ] 实现 `_add_to_graph()` - 添加节点和边

**Day 5: 实现倒排索引**
- [ ] 实现 `_build_inverted_index()` - 加速检索
- [ ] 实现 `get_statistics()` - 统计信息

**Day 6-7: 测试和调试**
- [ ] 完整测试: `pytest tests/test_bipartite.py::test_build -v`
- [ ] 修复bug，优化性能
- [ ] **周五下午3点**: 与成员A同步进度

**检查点**:
```bash
pytest tests/test_bipartite.py::test_build -v
```

---

### Week 3: 检索功能 + 第一次集成 🎯 重要里程碑
**目标**: 完成检索功能，与成员A进行第一次集成

**Day 1-3: 实现search()检索功能**
- [ ] 实现 `search()` 方法 - 基于关键词检索
- [ ] 使用倒排索引加速
- [ ] 测试: `pytest tests/test_bipartite.py::test_search -v`

**Day 4: 实现保存和加载**
- [ ] 实现 `save()` - 保存到文件
- [ ] 实现 `load()` - 从文件加载
- [ ] 测试保存/加载功能

**Day 5-6: 第一次集成测试** 🎯
- [ ] 与成员A一起运行集成测试
- [ ] 测试: `pytest tests/test_integration.py -v`
- [ ] 修复接口不兼容的问题

**Day 7: 优化和文档**
- [ ] 优化检索性能
- [ ] 添加代码注释
- [ ] **周五下午3点**: 集成测试总结会议

**检查点**:
```bash
pytest tests/test_bipartite.py -v  # 所有测试通过
pytest tests/test_integration.py -v  # 集成测试通过
```

---

### Week 4: 二部图专项实验 📊
**目标**: 完成4个实验，展示二部图的效果

**实验1 (Day 1-2): 关键词提取方法对比**
- [ ] 对比TF-IDF、TextRank、组合方法
- [ ] 生成对比表格和图表

**实验2 (Day 3-4): 二部图检索效果测试**
- [ ] 测试检索准确率
- [ ] 测试检索速度

**实验3 (Day 5): 成本对比分析**
- [ ] 计算LLM调用成本
- [ ] 对比骨架图vs二部图成本

**实验4 (Day 6-7): 参数敏感性分析**
- [ ] 测试不同top_k值的影响
- [ ] 测试不同权重的影响
- [ ] **周五下午3点**: 实验结果汇报

**交付物**: `results/bipartite/` 目录下的所有实验结果

---

### Week 5: 数据准备 + 可视化 📈
**目标**: 准备完整的测试数据集，生成所有可视化图表

**Day 1-2: 数据准备**
- [ ] 准备测试文档集（至少100个文档）
- [ ] 准备测试查询（至少50个查询）
- [ ] 准备标准答案（用于评估）

**Day 3-5: 可视化**
- [ ] 二部图结构可视化
- [ ] 实验结果可视化（柱状图、折线图）
- [ ] 成本对比可视化

**Day 6-7: 性能优化**
- [ ] 优化二部图构建速度
- [ ] 优化检索速度
- [ ] **周五下午3点**: 第二次集成测试

---

### Week 6: 报告撰写 + 最终集成 📝
**目标**: 撰写报告，完成最终系统集成

**Day 1-3: 撰写技术报告**
- [ ] 二部图设计部分
- [ ] 关键词提取算法说明
- [ ] 实现细节

**Day 4-5: 撰写实验报告**
- [ ] 实验设计
- [ ] 实验结果
- [ ] 结果分析

**Day 6-7: 最终集成**
- [ ] 与成员A进行最终集成测试
- [ ] 运行完整系统
- [ ] 准备答辩材料

---

## 🛠️ 技术实现指南

### 阶段1: 关键词提取（Week 1）

#### 1.1 TF-IDF实现

**文件位置**: `src/bipartite_graph/keyword_extractors.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfKeywordExtractor:
    """TF-IDF关键词提取器"""
    
    def __init__(self, max_features=1000):
        """
        Args:
            max_features: 最大特征数（词汇表大小）
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 1-2元组
            min_df=2,            # 至少在2个文档中出现
            max_df=0.8           # 最多在80%的文档中出现
        )
        self.feature_names = None
    
    def extract(self, texts: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        从文本列表中提取关键词
        
        Args:
            texts: 文本列表
            top_k: 每个文本提取的关键词数量
            
        Returns:
            每个文本的关键词列表
            例如: [[('知识图谱', 0.85), ('实体', 0.72)], ...]
        """
        # TODO: 你的实现
        # 提示:
        # 1. 使用 self.vectorizer.fit_transform(texts)
        # 2. 获取特征名 self.vectorizer.get_feature_names_out()
        # 3. 对每个文本，找到TF-IDF分数最高的top_k个词
        
        pass
```

**测试代码**: `tests/test_bipartite.py`

```python
def test_tfidf_extraction():
    """测试TF-IDF关键词提取"""
    extractor = TfidfKeywordExtractor()
    
    test_texts = [
        "知识图谱是一种结构化的语义知识库",
        "图神经网络可以处理非欧几里得数据"
    ]
    
    results = extractor.extract(test_texts, top_k=3)
    
    # 验证
    assert len(results) == 2
    assert len(results[0]) <= 3
    assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in results[0])
    
    print("✅ TF-IDF测试通过")
```

#### 1.2 TextRank实现

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
            关键词及其分数列表
        """
        # TODO: 你的实现
        # 提示: 使用 jieba.analyse.textrank()
        
        keywords = jieba.analyse.textrank(
            text,
            topK=top_k,
            withWeight=True
        )
        
        return keywords
```

---

### 阶段2: 二部图构建（Week 2）

#### 2.1 主构建函数

**文件位置**: `src/bipartite_graph/builder.py`

```python
from typing import List, Dict, Tuple
import sys
sys.path.append('/home/claude/graphrag-optimization')
from src.interfaces import BipartiteGraphInterface

class BipartiteGraphBuilder(BipartiteGraphInterface):
    """
    二部图构建器 - 你的核心实现
    """
    
    def __init__(self, config: Dict):
        """
        初始化
        
        Args:
            config: 配置字典，例如:
            {
                'top_k_keywords': 10,
                'tfidf_weight': 0.4,
                'textrank_weight': 0.4,
                'ner_weight': 0.2
            }
        """
        self.config = config
        
        # 图结构
        self.text_nodes = []      # [{'chunk_id': 0, 'text': '...', 'keywords': [...]}, ...]
        self.keyword_nodes = {}   # {'知识图谱': {'count': 5, 'text_ids': [0,2,5]}, ...}
        self.edges = []           # [(chunk_id, keyword, weight), ...]
        
        # 倒排索引：keyword -> [text_ids]
        self.inverted_index = {}
        
        # 初始化提取器
        self.tfidf_extractor = TfidfKeywordExtractor()
        self.textrank_extractor = TextRankExtractor()
    
    def build(self, text_chunks: List[str]) -> 'BipartiteGraphBuilder':
        """
        构建二部图
        
        这是你的核心函数！
        """
        print(f"\n[BipartiteGraph] 开始构建二部图...")
        print(f"  处理 {len(text_chunks)} 个文本块")
        
        # 步骤1: 批量提取TF-IDF关键词
        print("  [1/4] 批量提取TF-IDF关键词...")
        tfidf_results = self.tfidf_extractor.extract(
            text_chunks, 
            top_k=self.config['top_k_keywords']
        )
        
        # 步骤2: 对每个文本处理
        print("  [2/4] 处理每个文本块...")
        for i, chunk_text in enumerate(text_chunks):
            # 获取TF-IDF关键词
            tfidf_kws = tfidf_results[i]
            
            # 获取TextRank关键词
            textrank_kws = self.textrank_extractor.extract(
                chunk_text,
                top_k=self.config['top_k_keywords']
            )
            
            # TODO: 合并关键词
            final_keywords = self._merge_keywords(tfidf_kws, textrank_kws)
            
            # TODO: 添加到图中
            self._add_to_graph(i, chunk_text, final_keywords)
            
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{len(text_chunks)}")
        
        # 步骤3: 构建倒排索引
        print("  [3/4] 构建倒排索引...")
        self._build_inverted_index()
        
        # 步骤4: 统计
        print("  [4/4] 完成!")
        stats = self.get_statistics()
        print(f"✅ 二部图构建完成:")
        print(f"   - 文本节点: {stats['num_text_nodes']}")
        print(f"   - 关键词节点: {stats['num_keyword_nodes']}")
        print(f"   - 边数量: {stats['num_edges']}")
        
        return self
    
    def _merge_keywords(
        self, 
        tfidf_kws: List[Tuple[str, float]], 
        textrank_kws: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        合并多种方法的关键词
        
        TODO: 你的实现
        
        提示:
        1. 用字典累加分数: scores[keyword] += weight * score
        2. 按照config中的权重加权
        3. 排序并返回top_k
        """
        scores = {}
        
        # TF-IDF的权重
        w_tfidf = self.config['tfidf_weight']
        for keyword, score in tfidf_kws:
            scores[keyword] = scores.get(keyword, 0) + w_tfidf * score
        
        # TextRank的权重
        w_textrank = self.config['textrank_weight']
        for keyword, score in textrank_kws:
            scores[keyword] = scores.get(keyword, 0) + w_textrank * score
        
        # TODO: 如果使用NER，也加入进来
        
        # 排序
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top_k
        return sorted_keywords[:self.config['top_k_keywords']]
    
    def _add_to_graph(
        self, 
        chunk_id: int, 
        text: str, 
        keywords: List[Tuple[str, float]]
    ):
        """
        添加节点和边到图中
        
        TODO: 你的实现
        """
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
    
    def _build_inverted_index(self):
        """
        构建倒排索引: keyword -> [text_ids]
        
        加速检索用！
        """
        self.inverted_index = {}
        
        for chunk_id, keyword, weight in self.edges:
            if keyword not in self.inverted_index:
                self.inverted_index[keyword] = []
            self.inverted_index[keyword].append((chunk_id, weight))
    
    def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
        """
        检索功能 - Week 3实现
        """
        # TODO: Week 3实现
        pass
    
    def save(self, filepath: str):
        """保存"""
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
        """加载"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.text_nodes = data['text_nodes']
            self.keyword_nodes = data['keyword_nodes']
            self.edges = data['edges']
            self.inverted_index = data['inverted_index']
        print(f"✅ 已加载: {filepath}")
    
    def get_statistics(self) -> Dict:
        """统计信息"""
        # Top 10关键词
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

### 阶段3: 检索功能（Week 3）

```python
def search(self, query_keywords: List[str], top_k: int = 10) -> List[Dict]:
    """
    基于关键词检索文本块
    
    核心算法:
    1. 对每个查询关键词，找到包含它的文本（用倒排索引）
    2. 计算每个文本的匹配分数
    3. 排序并返回top_k
    """
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

## 🧪 测试指南

### 运行测试

```bash
# 测试单个函数
pytest tests/test_bipartite.py::test_tfidf_extraction -v

# 测试整个模块
pytest tests/test_bipartite.py -v

# 查看测试覆盖率
pytest tests/test_bipartite.py --cov=src/bipartite_graph
```

### 测试用例

**文件**: `tests/test_bipartite.py`

```python
import pytest
import sys
sys.path.append('/home/claude/graphrag-optimization')
from src.bipartite_graph.builder import BipartiteGraphBuilder

@pytest.fixture
def test_config():
    """测试配置"""
    return {
        'top_k_keywords': 5,
        'tfidf_weight': 0.5,
        'textrank_weight': 0.5,
        'ner_weight': 0.0
    }

@pytest.fixture
def test_chunks():
    """测试数据"""
    return [
        "知识图谱是一种结构化的语义知识库，用于存储实体和关系",
        "图神经网络可以处理非欧几里得数据，在社交网络分析中应用广泛",
        "检索增强生成结合了检索和生成两种方法，提升了大语言模型的性能"
    ]

def test_build(test_config, test_chunks):
    """测试构建功能"""
    builder = BipartiteGraphBuilder(test_config)
    builder.build(test_chunks)
    
    # 验证
    assert len(builder.text_nodes) == 3
    assert len(builder.keyword_nodes) > 0
    assert len(builder.edges) > 0
    
    print("✅ 构建测试通过")

def test_search(test_config, test_chunks):
    """测试检索功能"""
    builder = BipartiteGraphBuilder(test_config)
    builder.build(test_chunks)
    
    # 检索
    results = builder.search(['知识图谱', '实体'], top_k=2)
    
    # 验证
    assert len(results) > 0
    assert len(results) <= 2
    assert 'chunk_id' in results[0]
    assert 'score' in results[0]
    assert 'text' in results[0]
    
    print("✅ 检索测试通过")
```

---

## 📊 实验指南（Week 4）

### 实验1: 关键词提取方法对比

**目标**: 对比TF-IDF、TextRank、NER三种方法的效果

**文件**: `experiments/bipartite_experiments.py`

```python
import matplotlib.pyplot as plt

def exp1_keyword_extraction_comparison():
    """实验1: 关键词提取方法对比"""
    
    # 准备测试文本
    test_texts = [
        # 从数据集中选择10-20个代表性文本
    ]
    
    # 人工标注的标准答案（可选）
    ground_truth = [
        # 每个文本的标准关键词
    ]
    
    # 分别用三种方法提取
    tfidf_results = extract_with_tfidf(test_texts)
    textrank_results = extract_with_textrank(test_texts)
    combined_results = extract_with_combined(test_texts)
    
    # 计算准确率（如果有ground_truth）
    # ...
    
    # 生成对比表格
    # ...
    
    # 可视化
    plt.figure(figsize=(10, 6))
    # 绘制对比图
    plt.savefig('results/bipartite/exp1_keyword_comparison.png')
```

### 实验2-4: 类似实现

---

## 📈 可视化指南（Week 5）

### 1. 二部图结构可视化

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bipartite_graph(bipartite_graph, sample_size=30):
    """可视化二部图结构（抽样）"""
    
    # 创建networkx图
    G = nx.Graph()
    
    # 添加节点（抽样）
    sample_texts = bipartite_graph.text_nodes[:sample_size]
    # ...
    
    # 绘制
    plt.figure(figsize=(15, 10))
    # 使用bipartite_layout
    pos = nx.bipartite_layout(G, nodes=text_nodes)
    nx.draw(G, pos, ...)
    plt.savefig('results/bipartite/graph_structure.png')
```

---

## 🆘 常见问题

### Q1: TF-IDF提取的关键词质量不高怎么办？
**A**: 调整参数：
- 增加 `ngram_range=(1, 3)` 支持3元组
- 调整 `min_df` 和 `max_df`
- 过滤停用词

### Q2: 代码报错找不到模块？
**A**: 确保路径正确：
```python
import sys
sys.path.append('/home/claude/graphrag-optimization')
from src.bipartite_graph.builder import BipartiteGraphBuilder
```

### Q3: 如何与成员A集成？
**A**: 按照 `src/interfaces.py` 中的接口定义，确保：
- `build()` 返回 `self`
- `search()` 返回标准格式的列表

---

## ✅ 交付物清单

### Week 2末交付:
- [ ] `src/bipartite_graph/builder.py` (关键词提取 + 图构建)
- [ ] 通过 `tests/test_bipartite.py::test_build`

### Week 3末交付:
- [ ] `src/bipartite_graph/search.py` (检索功能)
- [ ] 通过 `tests/test_bipartite.py::test_search`
- [ ] 与成员A完成第一次集成

### Week 4末交付:
- [ ] `experiments/bipartite_experiments.py` (4个实验)
- [ ] `results/bipartite/` (所有实验结果和图表)

### Week 6末交付:
- [ ] 完整的二部图模块（所有功能）
- [ ] 技术文档和使用说明
- [ ] 实验报告（二部图部分）

---

## 📞 需要帮助？

- 每周五下午：与成员A同步进度
- 遇到技术问题：先查看文档，再问成员A
- 代码review：提前1天给成员A

**记住**: 你的模块是独立的，可以按自己的节奏开发！

---

**最后更新**: 2025-10-23  
**当前状态**: 🚀 准备开始  
**下一个检查点**: Week 1 - 关键词提取完成
