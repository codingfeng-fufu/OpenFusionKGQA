# GraphRAG成本优化系统 - KET-RAG实现

> **基于KET-RAG论文的GraphRAG成本优化系统，通过多粒度索引降低80%+的LLM调用成本**

---

## 🌳 分支管理

⚠️ **重要：开发时请使用独立分支，不要直接在main分支上修改！**

**分支说明**：
- **`main`** - 稳定版本，只有经过测试的代码才能合并
- **`feature-bipartite`** - 成员B的开发分支（二部图模块）
- **`feature-core`** - 成员A的开发分支（核心块选择器、骨架图、混合检索器）

**开发流程**：
```bash
# 1. 克隆项目
git clone https://github.com/codingfeng-fufu/KETGraphRAG.git
cd KETGraphRAG

# 2. 创建并切换到你的开发分支
git checkout -b feature-bipartite  # 成员B
# 或
git checkout -b feature-core       # 成员A

# 3. 在你的分支上开发和提交
git add .
git commit -m "[feat] 实现某功能"
git push origin feature-bipartite

# 4. 完成后创建Pull Request，review后合并到main
```


## 🎯 项目目标

### 核心问题
原始GraphRAG系统对所有文本都使用LLM提取实体和关系，成本太高💰💰💰

### 我们的解决方案
```
KET-RAG优化策略:
┌─────────────────────────────────────┐
│  1000个文本块                        │
└─────────────────────────────────────┘
         │
         ├─→ 核心块选择器
         │
    ┌────┴────┐
    │         │
核心块(20%)  非核心块(80%)
    │         │
    ↓         ↓
骨架图      二部图
(LLM提取)   (传统NLP)
💰💰💰      ✅省钱
    │         │
    └────┬────┘
         ↓
    混合检索器
         ↓
      最终答案

成本降低: 80%+
质量保持: 90%+
```

---

## 👥 团队分工

### 👨‍💻 成员A - 核心架构负责人
**负责模块**:
- ✅ 核心块选择器 (Core Chunk Selector)
- ✅ 骨架图构建器 (Skeleton Graph Builder)
- ✅ 混合检索器 (Hybrid Retriever)
- ✅ 系统集成与主实验流程

**工作量**: 约70%

### 👩‍💻 成员B - 二部图与实验负责人
**负责模块**:
- ⭐ **二部图构建器** (Bipartite Graph Builder) - 核心技术模块
- ⭐ 二部图专项实验 (4个实验)
- ⭐ 数据准备
- ⭐ 结果可视化

**工作量**: 约30%

**给成员B**: 请先阅读 `给同学B的任务说明.md` 📄

---

## 📁 项目结构

```
graphrag-optimization/
├── 📄 README.md                       # 项目说明（本文件）
├── 📄 给同学B的任务说明.md             # ⭐ 成员B必读
├── 📄 COLLABORATION.md                # 协作指南
├── 📄 QUICK_START.md                  # 快速开始
├── 📄 requirements.txt                # 依赖包
│
├── 📁 config/                         # 配置文件
│   ├── baseline_config.yaml          # Baseline GraphRAG配置
│   └── ketrag_config.yaml            # KET-RAG优化配置
│
├── 📁 src/                            # 源代码
│   ├── 📄 interfaces.py               # ⭐⭐⭐ 接口定义（必读！）
│   │
│   ├── 📁 core_selector/              # 核心块选择器 [成员A]
│   │   ├── __init__.py
│   │   ├── selector.py
│   │   └── algorithms.py
│   │
│   ├── 📁 skeleton_graph/             # 骨架图构建器 [成员A]
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── community.py
│   │
│   ├── 📁 bipartite_graph/             # ⭐ 二部图构建器 [成员B负责]
│   │   ├── __init__.py
│   │   ├── builder.py               # 主实现文件
│   │   ├── keyword_extractors.py    # 关键词提取（TF-IDF + TextRank）
│   │   └── search.py                # 检索功能
│   │
│   ├── 📁 retrieval/                  # 混合检索器 [成员A]
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py
│   │
│   ├── 📁 evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── cost_tracker.py          # 成本统计 [成员A]
│   │   └── quality_metrics.py       # 质量评估 [共同]
│   │
│   └── 📁 utils/                      # 工具函数
│       ├── __init__.py
│       ├── llm_wrapper.py           # LLM API封装
│       └── data_utils.py            # 数据处理工具
│
├── 📁 data/                           # 数据目录 [成员B准备]
│   ├── raw/                         # 原始数据
│   ├── processed/                   # 处理后的数据
│   │   ├── documents.json
│   │   ├── test_queries.json
│   │   └── ground_truth.json
│   └── prepared_by_memberB/         # 成员B准备的数据
│
├── 📁 experiments/                    # 实验脚本
│   ├── main_experiment.py           # 主实验流程 [成员A]
│   ├── bipartite_experiments.py     # ⭐ 二部图实验 [成员B]
│   └── run_all.py                   # 运行所有实验
│
├── 📁 tests/                          # 单元测试
│   ├── test_core_selector.py       # [成员A]
│   ├── test_skeleton.py             # [成员A]
│   ├── test_bipartite.py            # ⭐ [成员B]
│   └── test_integration.py          # 集成测试 [共同]
│
├── 📁 notebooks/                      # Jupyter notebooks
│   ├── demo.ipynb                   # 系统演示
│   └── bipartite_analysis.ipynb     # 二部图分析 [成员B]
│
├── 📁 results/                        # 实验结果
│   ├── baseline/                    # Baseline结果
│   ├── optimized/                   # KET-RAG结果
│   └── bipartite/                   # 二部图专项实验 [成员B]
│
├── 📁 docs/                           # 文档
│   ├── member_B_tasks.md           # ⭐ 成员B详细任务清单（700行）
│   ├── weekly_progress.md          # 周进度记录
│   └── interface_spec.md           # 接口规范
│
└── 📁 visualization/                  # 可视化
    ├── plot_results.py             # 结果可视化 [成员B]
    └── templates/                  # 图表模板
```

---

## 🚀 快速开始

### 成员A（你）- 开始开发
```bash
# 1. 环境搭建
cd graphrag-optimization
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 验证安装
python src/interfaces.py  # 应该看到接口演示

# 3. 开始实现核心块选择器
cd src/core_selector
# 创建 selector.py 并实现 CoreChunkSelectorInterface

# 4. 运行测试
pytest tests/test_core_selector.py -v
```

### 成员B（队友）- 开始开发
```bash
# 1. ⭐ 先读这3个文档（按顺序）
cat 给同学B的任务说明.md              # 快速了解（10分钟）
cat src/interfaces.py                   # 接口定义（20分钟）
cat docs/member_B_tasks.md             # 详细任务清单（1小时）

# 2. 环境搭建
cd graphrag-optimization
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 验证安装
python src/interfaces.py  # 应该看到接口演示

# 4. 开始实现二部图
mkdir -p src/bipartite_graph
cd src/bipartite_graph
# 创建 keyword_extractors.py 并实现TF-IDF提取

# 5. 运行测试
pytest tests/test_bipartite.py -v
```

---

## 🤝 协作流程

### 每周同步（每周五下午3点）
1. ✅ 检查本周完成情况
2. ✅ 更新 `docs/weekly_progress.md`
3. ✅ 讨论下周计划
4. ✅ 解决集成问题

### 代码共享（推荐Git）
```bash
# 成员A工作在 main 分支
git checkout main
git add .
git commit -m "[feat] 实现核心块选择器"
git push

# 成员B工作在 feature-bipartite 分支
git checkout -b feature-bipartite
git add .
git commit -m "[feat] 实现TF-IDF关键词提取"
git push origin feature-bipartite
```

### 集成测试时间点
- **Week 3 周五**: 第一次集成（核心块选择器 + 二部图）🎯
- **Week 5 周五**: 第二次集成（完整系统）🎯

---

## ⚠️ 重要说明

### ⭐ 并行开发的关键
1. **严格遵守接口定义** (`src/interfaces.py`)
2. **不修改对方的代码**（除非提前沟通）
3. **及时更新进度** (`docs/weekly_progress.md`)

### ⭐ 成员B注意事项
- 你的模块是**独立的**，不依赖成员A的实现
- 按照 `docs/member_B_tasks.md` 的步骤，一步步实现
- 遇到问题先看文档，再问成员A
- 每周五前完成本周任务并更新进度

### ⭐ 检查点
- **Week 2末**: 二部图关键词提取完成
- **Week 3末**: 二部图完整实现 + 第一次集成
- **Week 4末**: 二部图实验完成
- **Week 5末**: 完整系统集成

## 常见问题

**Q: 我的模块依赖对方的模块怎么办？**
A: 按照 `interfaces.py` 中的接口定义，先用mock数据测试，等对方完成后再集成。

**Q: 接口需要修改怎么办？**
A: 提前沟通，双方同意后才能修改 `interfaces.py`。

**Q: 测试数据在哪里？**
A: 成员B负责准备，放在 `data/processed/` 目录。

**Q: 如何运行完整系统？**
A: 等双方模块都完成后，运行 `experiments/run_all.py`。

## 联系方式
- 成员A: [你的联系方式]
- 成员B: [队友的联系方式]
- 项目讨论: [微信群/QQ群/其他]

## 时间规划
详见 `docs/weekly_progress.md`

---
**最后更新**: 2025-10-23
**项目状态**: 🚀 启动中
