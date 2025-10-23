# 协作指南 - GraphRAG优化项目

## 🤝 协作原则

### 核心原则
1. **接口至上**: 严格遵守 `src/interfaces.py` 中的接口定义
2. **互不阻塞**: 两人可以完全并行开发
3. **定期同步**: 每周五下午固定同步时间
4. **及时沟通**: 遇到问题及时交流

---

## 📁 代码管理

### 方案A: Git版本控制（强烈推荐）

#### 初始设置

```bash
# 成员A初始化仓库
cd graphrag-optimization
git init
git add .
git commit -m "Initial project structure"

# 创建远程仓库（GitHub/Gitee）
# 然后推送
git remote add origin <your-repo-url>
git push -u origin main

# 成员B克隆仓库
git clone <your-repo-url>
cd graphrag-optimization
```

#### 分支策略

```bash
# 成员A工作在 feature-core 分支
git checkout -b feature-core
# 开发...
git add .
git commit -m "实现核心块选择器"
git push origin feature-core

# 成员B工作在 feature-bipartite 分支
git checkout -b feature-bipartite
# 开发...
git add .
git commit -m "实现TF-IDF关键词提取"
git push origin feature-bipartite
```

#### 合并流程

```bash
# Week 3末，第一次集成
# 成员A先合并自己的代码到main
git checkout main
git merge feature-core

# 成员B合并时
git checkout main
git pull origin main  # 获取最新代码
git checkout feature-bipartite
git merge main  # 合并main到自己的分支
# 解决冲突（如果有）
git push origin feature-bipartite
# 然后提交PR/MR，由成员A review后合并
```

---

### 方案B: 共享文件夹

如果不用Git，可以用云盘：

```
OneDrive/Dropbox/百度网盘/
└── graphrag-optimization/
    ├── member_A/  # 成员A的代码
    │   ├── core_selector/
    │   ├── skeleton_graph/
    │   └── retrieval/
    │
    ├── member_B/  # 成员B的代码
    │   ├── bipartite_graph/
    │   └── experiments/
    │
    └── shared/    # 共享文件
        ├── interfaces.py  # 接口定义（只读！）
        ├── data/          # 数据集
        └── results/       # 实验结果
```

**规则**:
- 每人只修改自己的目录
- 修改前先同步最新版本
- 每天下班前上传代码

---

## 📅 集成时间点

### 第一次集成 (Week 3末)

**目标**: 验证接口兼容性

**成员A准备**:
- [ ] 核心块选择器完成
- [ ] 提供测试数据 `core_chunks` 和 `non_core_chunks`

**成员B准备**:
- [ ] 二部图构建和检索功能完成
- [ ] 通过所有单元测试

**集成测试**:
```python
# tests/test_integration.py

def test_first_integration():
    """第一次集成测试"""
    
    # 1. 成员A: 选择核心块
    selector = CoreChunkSelector()
    core_indices = selector.select(all_chunks, ratio=0.2)
    
    core_chunks = [all_chunks[i] for i in core_indices]
    non_core_chunks = [all_chunks[i] for i in range(len(all_chunks)) 
                      if i not in core_indices]
    
    # 2. 成员B: 构建二部图
    bipartite_builder = BipartiteGraphBuilder(config)
    bipartite_graph = bipartite_builder.build(non_core_chunks)
    
    # 3. 测试检索
    results = bipartite_graph.search(['知识图谱', '实体'], top_k=5)
    
    # 验证
    assert len(results) > 0
    assert 'chunk_id' in results[0]
    
    print("✅ 第一次集成成功！")
```

---

### 第二次集成 (Week 5末)

**目标**: 完整系统联调

**准备**:
- [ ] 成员A: 所有模块完成（核心选择器 + 骨架图 + 混合检索器）
- [ ] 成员B: 所有模块完成（二部图 + 实验 + 数据）

**集成测试**: 运行完整的实验流程

---

## 📞 沟通机制

### 每周同步会议（周五下午）

**时间**: 每周五 15:00-16:00

**议程**:
1. 回顾本周进度（各5分钟）
2. 展示本周成果（各10分钟）
3. 讨论遇到的问题（10分钟）
4. 计划下周任务（10分钟）
5. 其他事项（5分钟）

**成员B需要准备**:
- 本周完成的代码
- 遇到的问题和解决方案
- 下周计划

---

### 日常沟通

**工作日**:
- 早上10:00前: 简短同步今日计划（微信/QQ）
- 下午18:00: 更新今日进度

**响应时间**:
- 紧急问题: 2小时内响应
- 一般问题: 当天响应
- 代码review: 24小时内

---

## 🔧 技术决策流程

### 修改接口

**流程**:
1. 提出需求（说明为什么要改）
2. 讨论方案（两人都同意）
3. 更新 `interfaces.py`
4. 双方更新各自代码

**示例**:
```
成员B: "我想在search()返回结果中增加一个'keyword_weights'字段，
       用于展示每个关键词的权重，这样实验分析会更清楚。"
       
成员A: "好的，这不影响我的代码。我会更新混合检索器来处理这个
       新字段。我们今晚都更新一下接口定义。"
```

---

### 技术选型

**原则**: 谁负责谁决定

- 成员A负责的模块: 成员A决定技术方案
- 成员B负责的模块: 成员B决定技术方案
- 共同模块: 讨论后决定

**示例**:
```
成员B: "关键词提取我想用jieba + TextRank，可以吗？"
成员A: "可以，只要接口符合定义就行。"
```

---

## 🐛 问题解决

### 冲突解决

#### 代码冲突
```bash
# 如果Git merge时出现冲突
git status  # 查看冲突文件

# 手动编辑冲突文件
# 保留正确的代码，删除冲突标记
<<<<<<< HEAD
你的代码
=======
对方的代码
>>>>>>> branch-name

# 解决后
git add <冲突文件>
git commit -m "解决冲突"
```

#### 接口冲突
如果发现接口不匹配:
1. 立即通知对方
2. 开会讨论原因
3. 决定谁改动（通常是发现问题的人）
4. 更新文档

---

### Bug追踪

**使用Issues**（如果用GitHub）:
```
标题: [Bug] 二部图检索返回结果格式不正确
标签: bug, bipartite-graph
指派: 成员B

描述:
search()返回的结果缺少'matched_keywords'字段

复现步骤:
1. 构建二部图
2. 调用search(['知识图谱'], top_k=5)
3. 打印results[0]

期望: {'chunk_id': 0, 'text': '...', 'score': 0.8, 'matched_keywords': ['知识图谱']}
实际: {'chunk_id': 0, 'text': '...', 'score': 0.8}
```

---

## 📊 进度跟踪

### 使用进度文档

**文件**: `docs/weekly_progress.md`

```markdown
# 项目进度记录

## Week 1 (2025-10-23 ~ 2025-10-29)

### 成员A
- [x] 环境搭建
- [x] 接口定义
- [x] 核心块选择器 - PageRank算法实现
- [ ] 核心块选择器 - 重要性算法实现

进度: 80%
问题: 无

### 成员B
- [x] 环境搭建
- [x] 阅读文档
- [x] TF-IDF关键词提取实现
- [ ] TextRank关键词提取实现

进度: 70%
问题: jieba安装遇到问题，已解决

### 下周计划
- 成员A: 完成核心块选择器，开始骨架图构建
- 成员B: 完成关键词提取，开始二部图构建
```

---

## 📝 文档规范

### 代码注释

```python
class BipartiteGraphBuilder:
    """
    二部图构建器
    
    功能: 为非核心文本块构建轻量级索引
    作者: 成员B
    日期: 2025-10-23
    
    使用示例:
        >>> builder = BipartiteGraphBuilder(config)
        >>> builder.build(text_chunks)
        >>> results = builder.search(['关键词'], top_k=10)
    """
    
    def build(self, text_chunks: List[str]) -> 'BipartiteGraphBuilder':
        """
        构建二部图
        
        Args:
            text_chunks: 文本块列表
            
        Returns:
            self: 构建完成的对象
            
        Raises:
            ValueError: 如果text_chunks为空
        """
        pass
```

### Commit规范

```bash
# 格式: [类型] 简短描述

# 类型:
- feat: 新功能
- fix: Bug修复
- docs: 文档更新
- test: 测试相关
- refactor: 代码重构

# 示例:
git commit -m "[feat] 实现TF-IDF关键词提取"
git commit -m "[fix] 修复search()返回格式错误"
git commit -m "[docs] 更新接口文档"
```

---

## ✅ 检查清单

### 提交代码前
- [ ] 代码符合接口定义
- [ ] 通过所有单元测试
- [ ] 添加了必要的注释
- [ ] 更新了文档（如果需要）

### 集成测试前
- [ ] 本地所有测试通过
- [ ] 代码已推送到远程
- [ ] 通知了对方准备集成

### 每周五会议前
- [ ] 更新进度文档
- [ ] 准备演示材料
- [ ] 列出需要讨论的问题

---

## 🆘 紧急联系

**紧急情况**（需要立即响应）:
- 系统崩溃无法恢复
- 接口严重冲突
- 数据丢失

**联系方式**:
- 成员A: [手机号/微信]
- 成员B: [手机号/微信]

**备份方案**:
- 代码: 每天备份到云盘
- 数据: 多处备份
- 文档: Git + 云盘双保险

---

## 📚 参考资料

### 必读文档
1. `src/interfaces.py` - 接口定义
2. `docs/member_B_tasks.md` - 成员B任务清单
3. `README.md` - 项目概览

### 技术文档
- KET-RAG论文: [链接]
- TF-IDF教程: [链接]
- jieba文档: [链接]

---

**最后更新**: 2025-10-23  
**维护者**: 成员A
