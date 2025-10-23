# GraphRAG v2

基于微软 GraphRAG 源码学习和重构的知识图谱增强检索系统。

## 项目目标

参考微软开源的 GraphRAG 实现，从零开始构建一个生产级别的 GraphRAG 系统，学习并应用最佳实践。

## 当前进度

- [x] 项目初始化
- [x] **阶段1：配置系统重构** ✅
  - [x] 1.1 学习微软的配置模型设计
  - [x] 1.2 创建基础配置类
  - [x] 1.3 实现配置加载机制
  - [x] 1.4 添加配置验证
  - [x] 1.5 创建默认配置模板
- [x] **阶段2：数据模型标准化** ✅
  - [x] 2.1 学习微软的数据模型
  - [x] 2.2 定义核心数据类
  - [x] 2.3 定义Schema常量
  - [x] 2.4 实现数据转换工具
  - [x] 2.5 添加数据验证
- [x] **阶段3：索引Pipeline重构（基础版本）** ✅
  - [x] 3.1 学习微软的Pipeline架构
  - [x] 3.2 实现Pipeline核心模块
  - [x] 3.3 实现文档加载工作流
  - [x] 3.4 实现文本分块工作流
  - [x] 3.5 实现Pipeline工厂和运行器
- [x] **阶段3扩展1：实体提取和社区检测** ✅
  - [x] 3.6 实现实体提取工作流（规则版本）
  - [x] 3.7 实现社区检测工作流（Louvain算法）
  - [x] 3.8 创建端到端测试
- [x] **阶段3扩展2：社区报告和嵌入生成** ✅
  - [x] 3.9 实现社区报告生成工作流
  - [x] 3.10 实现文本嵌入生成工作流
  - [x] 3.11 实现嵌入相似度搜索
  - [x] 3.12 创建完整Pipeline测试
- [x] **阶段4：查询引擎优化** ✅
  - [x] 4.1 学习微软的查询架构
  - [x] 4.2 实现查询基础模块
  - [x] 4.3 实现 Global Search
  - [x] 4.4 实现 Local Search
  - [x] 4.5 创建查询测试
- [x] **阶段5：Prompt工程优化** ✅
  - [x] 5.1 创建 Prompt 模板系统
  - [x] 5.2 实现实体提取 Prompt
  - [x] 5.3 实现社区报告 Prompt
  - [x] 5.4 实现查询 Prompt（Map/Reduce/Local）
  - [x] 5.5 集成 GLM API
  - [x] 5.6 创建端到端测试
- [/] **阶段6：测试与文档** ⏳ **接近完成**
  - [x] 6.1 创建测试基础设施（Pytest + Fixtures）
  - [x] 6.2 创建单元测试套件（91个测试）
  - [x] 6.3 创建集成测试
  - [x] 6.4 修复核心模块测试（**80%通过率** 🎉）
    - [x] Prompt 模块 100% 通过
    - [x] LLM 模块 100% 通过
    - [x] Data Model 模块 100% 通过
    - [/] Config 模块 29% 通过（待修复）
  - [ ] 6.5 编写 API 文档
  - [ ] 6.6 创建部署指南
  - [ ] 6.7 性能优化和基准测试

## 项目结构

```
graphrag_v2/
├── config/                    # 配置模块
│   ├── models/               # 配置数据模型
│   ├── defaults.py           # 默认配置
│   └── loader.py             # 配置加载器
├── data_model/               # 数据模型
├── pipeline/                 # Pipeline 模块
│   ├── context.py            # Pipeline 上下文
│   ├── runner.py             # Pipeline 运行器
│   └── factory.py            # Pipeline 工厂
├── workflows/                # 工作流模块
│   ├── load_documents.py    # 文档加载
│   ├── create_base_text_units.py  # 文本分块
│   ├── extract_graph.py      # 实体提取
│   ├── create_communities.py # 社区检测
│   ├── create_community_reports.py  # 社区报告
│   └── generate_embeddings.py  # 嵌入生成
├── query/                    # 查询模块
│   ├── base.py               # 查询基类
│   ├── context_builder.py    # 上下文构建器
│   ├── global_search.py      # Global Search
│   └── local_search.py       # Local Search
├── prompts/                  # Prompt 模板
│   ├── base.py               # 模板基类
│   ├── entity_extraction.py  # 实体提取 Prompt
│   ├── community_report.py   # 社区报告 Prompt
│   └── query_prompts.py      # 查询 Prompt
├── llm/                      # LLM 模块
│   └── glm_client.py         # GLM 客户端
├── storage/                  # 存储模块
├── utils/                    # 工具函数
└── tests/                    # 测试
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
python test_config.py
```

### 使用示例

```python
from graphrag_v2 import GraphRagConfig, create_default_config, load_config

# 创建默认配置
config = create_default_config()

# 从文件加载配置
config = load_config("settings.yaml")

# 获取模型配置
chat_model = config.get_language_model_config("default_chat_model")
print(f"使用模型: {chat_model.model}")
```

## 学习笔记

### 阶段1：配置系统重构 ✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)

#### 关键学习点

1. **使用 Pydantic BaseModel**
   - 类型安全的配置管理
   - 自动验证和序列化
   - 使用 `@model_validator` 和 `@field_validator`

2. **分层配置结构**
   - 主配置类 `GraphRagConfig`
   - 子配置类（LLM、Storage、Chunking 等）
   - 使用 `Field` 定义字段和默认值

3. **验证机制**
   - 字段级验证：`@field_validator`
   - 模型级验证：`@model_validator(mode="after")`
   - 自定义验证方法

4. **默认值管理**
   - 集中在 `defaults.py` 中
   - 使用 dataclass 定义
   - 通过 `Field(default=...)` 引用

5. **环境变量支持**
   - 使用 python-dotenv 加载 .env 文件
   - 支持环境变量覆盖配置
   - 灵活的配置管理

#### 实现的功能

- ✅ 完整的配置模型类（6个）
- ✅ 配置加载器（支持 YAML/JSON）
- ✅ 环境变量覆盖
- ✅ 配置验证机制
- ✅ 配置模板和示例
- ✅ 完整的测试套件

---

### 阶段2：数据模型标准化 ✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md)

#### 关键学习点

1. **使用 dataclass**
   - 简洁的数据类定义
   - 自动生成 `__init__`、`__repr__` 等方法
   - 完整的类型注解

2. **继承层次结构**
   - `Identified` → `Named` → 具体类
   - 注意 dataclass 继承时的默认值问题
   - 使用 `from_dict` 类方法创建对象

3. **Schema 常量**
   - 集中定义所有列名
   - 定义最终输出列的顺序
   - 确保数据处理的一致性

4. **数据转换**
   - dataclass ↔ DataFrame 双向转换
   - 处理列表和字典类型（JSON 序列化）
   - 处理 NaN 值和字段过滤

5. **数据验证**
   - 单个对象验证
   - 批量验证
   - 返回详细的错误消息

#### 实现的功能

- ✅ 核心数据类（9个）
- ✅ Schema 常量定义
- ✅ 数据转换工具（14个函数）
- ✅ 数据验证工具（14个函数）
- ✅ 完整的测试套件

---

### 阶段3：索引 Pipeline 重构（基础版本）✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)

#### 关键学习点

1. **Pipeline 架构**
   - Pipeline 是工作流的有序集合
   - Workflow 是 (名称, 函数) 的元组
   - 统一的工作流函数签名

2. **运行上下文**
   - 所有工作流共享同一个上下文
   - 通过存储抽象实现数据传递
   - 支持统计信息、缓存、回调

3. **异步编程**
   - 使用 async/await 模式
   - 异步生成器返回结果
   - 支持并发执行

4. **工厂模式**
   - 集中管理所有工作流
   - 支持自定义工作流注册
   - 灵活的 Pipeline 组合

5. **文本分块**
   - 使用 tiktoken 进行基于 token 的分块
   - 支持可配置的块大小和重叠
   - 生成唯一的文本单元 ID

#### 实现的功能

- ✅ Pipeline 核心模块（5个文件）
- ✅ 工作流实现（2个工作流）
- ✅ Pipeline 工厂和运行器
- ✅ 存储和缓存抽象
- ✅ 回调机制
- ✅ 完整的测试套件

---

### 阶段3扩展1：实体提取和社区检测 ✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE3_EXTENDED_SUMMARY.md](PHASE3_EXTENDED_SUMMARY.md)

#### 新增工作流

1. **实体提取工作流** (`extract_graph.py`)
   - 使用规则提取实体和关系
   - 支持中英文实体识别
   - 自动合并重复实体
   - 计算实体和关系排名

2. **社区检测工作流** (`create_communities.py`)
   - 使用 Louvain 算法检测社区
   - 处理不连通图
   - 计算社区统计信息
   - 生成 NetworkX 图对象

#### 测试结果

使用 15 行测试文本，成功提取：
- ✅ 60 个唯一实体
- ✅ 174 个唯一关系
- ✅ 5 个社区
- ✅ 平均社区大小: 12.00
- ✅ 运行时间: 0.29秒

---

### 阶段3扩展2：社区报告和嵌入生成 ✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE3_FINAL_SUMMARY.md](PHASE3_FINAL_SUMMARY.md)

#### 新增工作流

3. **社区报告生成工作流** (`create_community_reports.py`)
   - 为每个社区生成标题、摘要和完整内容
   - 计算社区排名（基于大小、密度、平均度）
   - 生成关键发现（最重要实体、最强关系、主导类型）
   - 支持 Markdown 格式输出

4. **文本嵌入生成工作流** (`generate_embeddings.py`)
   - 为文本单元和实体生成向量嵌入
   - 使用 mock 嵌入（确定性哈希，1536维）
   - 支持相似度搜索（余弦相似度）
   - 支持 top-k 检索

#### 完整 Pipeline 测试结果

使用 30 行测试文本，成功完成：
- ✅ 109 个唯一实体
- ✅ 257 个唯一关系
- ✅ 21 个社区
- ✅ 21 个社区报告
- ✅ 110 个嵌入向量（1 文本单元 + 109 实体）
- ✅ 嵌入相似度搜索测试通过
- ✅ 总运行时间: 0.43秒

#### 实现的功能

- ✅ 完整的 6 个工作流 Pipeline
- ✅ 社区报告生成（规则版本）
- ✅ 文本嵌入生成（mock 版本）
- ✅ 嵌入相似度搜索
- ✅ 完整的端到端测试（`test_full_pipeline.py`）
- ✅ 详细的日志记录和统计信息

---

### 阶段4：查询引擎优化 ✅

**完成时间**: 2025-10-16
**详细总结**: 见 [PHASE4_SUMMARY.md](PHASE4_SUMMARY.md)

#### 实现的功能

1. **查询基础模块** (`query/base.py`, `query/context_builder.py`)
   - BaseSearch: 搜索引擎基类
   - SearchResult: 搜索结果数据类
   - ContextBuilder: 上下文构建器基类
   - GlobalContextBuilder 和 LocalContextBuilder

2. **Global Search** (`query/global_search.py`, `query/global_context_builder.py`)
   - 使用 Map-Reduce 模式
   - Map 阶段：对每批社区报告并行生成中间答案
   - Reduce 阶段：合并中间答案生成最终答案
   - CommunityContextBuilder: 基于社区报告的上下文构建器

3. **Local Search** (`query/local_search.py`, `query/local_context_builder.py`)
   - 基于向量相似度检索
   - 使用查询嵌入找到最相关的实体
   - 获取相关的关系和社区
   - EntityRelationshipContextBuilder: 基于实体和关系的上下文构建器

#### 测试结果

**Global Search**:
- ✅ 查询: "总结整个数据集的主要主题"
- ✅ Map 响应数: 2
- ✅ LLM 调用次数: 3
- ✅ 完成时间: 0.00 秒

**Local Search**:
- ✅ 查询: "GraphRAG 是什么？"
- ✅ 相关实体数: 3
- ✅ 相关关系数: 4
- ✅ 完成时间: 0.00 秒

#### 关键学习点

1. **Global Search vs Local Search**
   - Global: 全局视角，Map-Reduce 模式，适合总结性问题
   - Local: 局部细节，向量检索，适合具体问题

2. **Map-Reduce 模式**
   - Map: 并行处理数据批次
   - Reduce: 合并中间结果
   - 使用 asyncio.gather 实现并发

3. **向量相似度检索**
   - 生成查询嵌入
   - 计算余弦相似度
   - Top-K 检索最相关实体

4. **上下文构建策略**
   - Global: 分批社区报告
   - Local: 实体 + 关系 + 社区的多层次上下文

### 阶段5：Prompt 工程优化（2025-10-16）

#### 实现内容

**1. Prompt 模板系统** (`prompts/base.py`)
- ✅ PromptTemplate 类：支持变量替换、默认值、条件渲染
- ✅ PromptLibrary 类：集中管理多个 Prompt 模板

**2. 实体提取 Prompt** (`prompts/entity_extraction.py`)
- ✅ 完全中文化的 Prompt 模板
- ✅ 3 个高质量 Few-shot 示例（科技公司、商业新闻、国际新闻）
- ✅ 结构化输出格式（使用分隔符）
- ✅ 支持自定义实体类型

**3. 社区报告 Prompt** (`prompts/community_report.py`)
- ✅ 角色扮演式 Prompt
- ✅ JSON 格式输出（标题、摘要、评分、发现）
- ✅ 数据引用要求
- ✅ 长度控制

**4. 查询 Prompt** (`prompts/query_prompts.py`)
- ✅ Global Search Map Prompt（提取关键点）
- ✅ Global Search Reduce Prompt（综合答案）
- ✅ Local Search Prompt（详细回答）

**5. GLM 客户端** (`llm/glm_client.py`)
- ✅ 智谱 AI GLM-4 集成
- ✅ 自动降级到 mock 模式（无 API key 时）
- ✅ 重试机制（默认 3 次）
- ✅ 统计跟踪（调用次数、Token 数、错误次数）
- ✅ 智能 mock 响应（为每种 Prompt 类型提供合适的响应）

#### 测试结果

**测试 1: 实体提取**
- ✅ Prompt 长度: 1027 字符
- ✅ 提取到 2 个实体
- ✅ 提取到 1 个关系

**测试 2: 社区报告**
- ✅ Prompt 长度: 3804 字符
- ✅ JSON 格式正确
- ✅ 包含标题、评分、发现

**测试 3: Global Search**
- ✅ Map 和 Reduce 阶段都正常
- ✅ 响应格式正确

**测试 4: Local Search**
- ✅ Prompt 长度: 1315 字符
- ✅ 响应格式正确

#### 关键学习点

1. **Prompt 工程最佳实践**
   - 清晰的角色定义："你是一个..."
   - 明确的目标："生成一个..."
   - 分步骤指导："1. 识别... 2. 提取..."
   - Few-shot 示例：提供 2-3 个高质量示例
   - 结构化输出：使用分隔符或 JSON 格式
   - 数据引用：要求引用数据源
   - 长度控制：限制响应长度

2. **GLM-4 vs OpenAI**
   - GLM-4 对中文的理解和生成能力更强
   - API 接口与 OpenAI 类似，易于迁移
   - 成本更低，服务更稳定（国内）

3. **Mock 模式的价值**
   - 无需 API key 即可开发和测试
   - 避免频繁调用 API 产生费用
   - 加速开发和测试流程
   - 支持离线开发

4. **中文优化策略**
   - 完全中文化的 Prompt 和示例
   - 使用中文实体类型（"组织"、"人物"）
   - 要求以中文返回结果
   - 中文场景的 Few-shot 示例

## 参考资料

- 微软 GraphRAG: `../graphrag-main/`
- 重构计划: `../graphrag_fixed/REFACTOR_PLAN.md`
- 智谱 AI 文档: https://open.bigmodel.cn/

