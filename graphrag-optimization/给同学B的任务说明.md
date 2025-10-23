# 给同学B的任务说明

> **Hi！** 这是我们GraphRAG优化项目的分工说明。你负责**二部图构建器**模块，这是项目的核心组件之一！

---

## 📋 快速了解

### 你的任务
- **模块名称**: 二部图构建器（Bipartite Graph Builder）
- **技术难度**: ⭐⭐⭐（中等，主要是传统NLP方法）
- **核心价值**: 用传统NLP方法替代昂贵的LLM调用，**降低80%成本**！

### 项目背景
我们在做一个基于KET-RAG论文的GraphRAG成本优化系统。核心思路是：

```
原始GraphRAG: 所有文本都用LLM提取 → 成本高💰💰💰

我们的优化:
├─ 核心文本(20%) → LLM提取 → 高质量知识图谱 [我负责]
└─ 非核心文本(80%) → 传统NLP → 轻量级索引 [你负责] ✅

最终: 成本降低80%，质量保持90%+
```

---

## 🎯 你需要完成的任务

### 任务1: 二部图构建器（核心模块）⭐
**文件位置**: `src/bipartite_graph/builder.py`

**功能**: 为非核心文本块构建轻量级的文本-关键词索引

**需要实现的方法**:
- `build(text_chunks)` - 从文本块构建二部图
- `search(query_keywords, top_k)` - 基于关键词检索文本
- `save(filepath)` / `load(filepath)` - 保存/加载
- `get_statistics()` - 获取统计信息

**技术栈**:
- TF-IDF: 统计方法提取关键词
- TextRank: 图算法提取关键词
- 二部图: 文本节点 ↔ 关键词节点
- 倒排索引: 快速检索

### 任务2: 关键词提取器
**文件位置**: `src/bipartite_graph/keyword_extractors.py`

**需要实现**:
- `TfidfKeywordExtractor` - TF-IDF关键词提取
- `TextRankExtractor` - TextRank关键词提取
- `merge_keywords()` - 合并多种方法的关键词

### 任务3: 二部图专项实验（展示效果）
**文件位置**: `experiments/bipartite_experiments.py`

**4个实验**:
1. 关键词提取方法对比（TF-IDF vs TextRank vs 组合）
2. 二部图检索效果测试
3. 成本对比分析（LLM vs 传统NLP）
4. 参数敏感性分析

### 任务4: 数据准备
**文件位置**: `data/processed/`

**需要准备**:
- 测试文档集（建议100个文档）
- 测试查询（建议50个查询）
- 标准答案（用于评估）

### 任务5: 结果可视化
**文件位置**: `visualization/plot_results.py`

**需要生成**:
- 二部图结构可视化
- 实验结果图表（柱状图、折线图）
- 成本对比图

---

## 📚 必读文档（按顺序阅读）

### 第一步：快速了解
- [ ] **本文档** - 任务概览
- [ ] `README.md` - 项目整体介绍
- [ ] `src/interfaces.py` - **⭐ 最重要！接口定义**

### 第二步：详细了解
- [ ] `docs/member_B_tasks.md` - 详细任务清单（包含完整代码示例）
- [ ] `COLLABORATION.md` - 协作规范

### 第三步：开发参考
- [ ] `config/ketrag_config.yaml` - 配置文件
- [ ] `tests/test_bipartite.py` - 测试用例

---

## 🚀 快速开始

### 步骤1: 搭建环境
```bash
# 1. 进入项目目录
cd graphrag-optimization

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python src/interfaces.py  # 应该看到接口演示
```

### 步骤2: 阅读接口定义
```bash
# 查看接口定义（最重要！）
cat src/interfaces.py

# 重点看这个类：
# class BipartiteGraphInterface(ABC):
#     def build(text_chunks: List[str]) -> 'BipartiteGraphInterface'
#     def search(query_keywords: List[str], top_k: int = 10) -> List[Dict]
#     def save(filepath: str) -> None
#     def load(filepath: str) -> None
#     def get_statistics() -> Dict[str, Any]
```

### 步骤3: 开始编码
```bash
# 1. 创建你的工作目录
mkdir -p src/bipartite_graph

# 2. 创建文件
touch src/bipartite_graph/__init__.py
touch src/bipartite_graph/keyword_extractors.py
touch src/bipartite_graph/builder.py

# 3. 开始实现
# 参考 docs/member_B_tasks.md 中的代码示例
```

---

## 🤝 协作方式

### 沟通渠道
- **微信/QQ**: 日常沟通
- **定期同步**: 我们会定期讨论进度
- **有问题随时找我**: 不要憋着

### 代码共享和分支管理

⚠️ **重要：请在自己的分支上开发，不要直接在main分支上修改！**

```bash
# 1. 克隆项目
git clone https://github.com/codingfeng-fufu/KETGraphRAG.git
cd KETGraphRAG

# 2. 创建你自己的开发分支
git checkout -b feature-bipartite

# 3. 在你的分支上开发和提交
git add .
git commit -m "[feat] 实现TF-IDF关键词提取"
git push origin feature-bipartite

# 4. 开发完成后，创建Pull Request
# 访问 GitHub 网页，点击 "New Pull Request"
# 选择 feature-bipartite -> main
# 我会review后合并到main分支
```

**分支说明**：
- `main` 分支：稳定版本，只有经过测试的代码才能合并
- `feature-bipartite` 分支：你的开发分支，可以自由提交
- 其他分支：我会在 `feature-core` 等分支上开发

### 工作原则
1. ✅ **严格遵守接口定义** - 不要擅自修改 `interfaces.py`
2. ✅ **独立开发** - 你的模块不依赖我的，可以并行工作
3. ✅ **及时沟通** - 遇到问题随时问我

---

## 💡 技术提示

### 关键词提取示例
```python
# TF-IDF示例
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)
# 提取每个文本的top-k关键词
```

### 二部图数据结构
```python
# 数据结构示例
self.text_nodes = [
    {'chunk_id': 0, 'text': '...', 'keywords': ['知识图谱', '实体']},
    {'chunk_id': 1, 'text': '...', 'keywords': ['图神经网络', '社交网络']},
]

self.keyword_nodes = {
    '知识图谱': {'count': 5, 'text_ids': [0, 2, 5]},
    '实体': {'count': 3, 'text_ids': [0, 3, 7]},
}

# 倒排索引（加速检索）
self.inverted_index = {
    '知识图谱': [(0, 0.85), (2, 0.72), (5, 0.68)],  # (chunk_id, weight)
    '实体': [(0, 0.75), (3, 0.80), (7, 0.65)],
}
```

---

## 📊 最终交付物

### 代码模块
- ✅ `src/bipartite_graph/keyword_extractors.py` - 关键词提取器
- ✅ `src/bipartite_graph/builder.py` - 二部图构建器
- ✅ `src/bipartite_graph/search.py` - 检索功能（可选，也可以放在builder.py中）
- ✅ `tests/test_bipartite.py` - 单元测试

### 实验结果
- ✅ `experiments/bipartite_experiments.py` - 4个实验脚本
- ✅ `results/bipartite/` - 所有实验结果和数据

### 数据和可视化
- ✅ `data/processed/` - 测试数据集
- ✅ `visualization/plot_results.py` - 可视化脚本
- ✅ 各种图表（PNG/PDF格式）

### 文档
- ✅ 代码注释（清晰的函数和类说明）
- ✅ 技术报告（二部图部分）

---

## 🆘 常见问题

**Q: 我不太懂TF-IDF怎么办？**
A: 没关系！`docs/member_B_tasks.md` 里有详细的代码示例，直接参考就行。

**Q: 接口定义看不懂？**
A: 运行 `python src/interfaces.py` 看演示，或者直接问我。

**Q: 我的进度慢了怎么办？**
A: 及时告诉我，我们可以调整任务或者我帮你。

**Q: 需要什么数据集？**
A: 开始可以用小数据测试（3-5个文档就够），后面再准备完整数据集。

**Q: 测试怎么写？**
A: 参考 `tests/test_bipartite.py`，我已经写好了测试框架。

**Q: 什么时候需要和你集成？**
A: 当你的二部图构建器和检索功能都实现好了，我们就可以集成测试了。

---

## ✅ 开始前的准备

建议你先完成这些：

- [ ] 阅读本文档
- [ ] 阅读 `README.md`
- [ ] 阅读 `src/interfaces.py` ⭐ 最重要
- [ ] 搭建开发环境（运行 `python src/interfaces.py` 成功）
- [ ] 创建工作目录 `src/bipartite_graph/`
- [ ] 回复我：确认收到任务，环境搭建完成

---

## 📞 联系我

- **微信/QQ**: [你的联系方式]
- **邮箱**: [你的邮箱]

有任何问题随时找我！我们一起把这个项目做好！💪

---

**最后更新**: 2025-10-23
**项目状态**: 🚀 启动中

