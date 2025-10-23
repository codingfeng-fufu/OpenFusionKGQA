# 阶段2总结：数据模型标准化

## ✅ 完成情况

**状态**: 已完成 ✓  
**完成时间**: 2025-10-16  
**耗时**: 约 1 小时

## 📚 学习成果

### 1. 微软 GraphRAG 数据模型的核心设计

通过深入研究微软 GraphRAG 的数据模型，我们学到了以下关键设计模式：

#### 1.1 继承层次结构

```
Identified (基础类)
├── Named (带名称的类)
│   ├── Entity (实体)
│   ├── Community (社区)
│   ├── CommunityReport (社区报告)
│   └── Document (文档)
├── Relationship (关系)
├── TextUnit (文本单元)
└── Covariate (协变量)
```

#### 1.2 使用 dataclass

- **简洁性**: 使用 `@dataclass` 装饰器自动生成 `__init__`、`__repr__` 等方法
- **类型安全**: 所有字段都有明确的类型注解
- **可选字段**: 使用 `| None` 表示可选字段
- **默认值**: 为可选字段提供合理的默认值

#### 1.3 from_dict 模式

- 每个数据类都提供 `from_dict` 类方法
- 支持自定义键名映射
- 便于从 DataFrame 或 JSON 创建对象

#### 1.4 Schema 常量

- 集中定义所有列名常量
- 定义最终输出列的顺序
- 确保数据处理的一致性

## 🎯 实现的功能

### 1. 核心数据类（9个）

#### 1.1 基础类

**Identified** (`identified.py`)
- `id`: 唯一标识符
- `short_id`: 人类可读的短 ID

**Named** (`named.py`)
- 继承自 `Identified`
- `title`: 名称/标题

#### 1.2 实体和关系

**Entity** (`entity.py`)
- 继承自 `Named`
- 字段：type, description, embeddings, community_ids, text_unit_ids, rank, attributes
- 表示知识图谱中的实体（人物、组织、地点等）

**Relationship** (`relationship.py`)
- 继承自 `Identified`
- 字段：source, target, weight, description, embedding, text_unit_ids, rank, attributes
- 表示实体之间的关系

#### 1.3 社区相关

**Community** (`community.py`)
- 继承自 `Named`
- 字段：level, parent, children, entity_ids, relationship_ids, text_unit_ids, covariate_ids, size, period
- 表示通过社区检测算法发现的实体社区
- 支持层次化结构

**CommunityReport** (`community_report.py`)
- 继承自 `Named`
- 字段：community_id, summary, full_content, rank, full_content_embedding, size, period
- 表示 LLM 生成的社区摘要报告

#### 1.4 文本处理

**TextUnit** (`text_unit.py`)
- 继承自 `Identified`
- 字段：text, entity_ids, relationship_ids, covariate_ids, n_tokens, document_ids, attributes
- 表示文档分块后的文本单元

**Document** (`document.py`)
- 继承自 `Named`
- 字段：type, text_unit_ids, text, attributes
- 表示原始文档

#### 1.5 协变量

**Covariate** (`covariate.py`)
- 继承自 `Identified`
- 字段：subject_id, subject_type, covariate_type, text_unit_ids, attributes
- 表示与主体相关的元数据（如实体声明）

### 2. Schema 常量定义

**schemas.py** - 定义了所有 Parquet 文件的列名和输出列顺序：

- ✅ 通用字段常量（ID, TITLE, DESCRIPTION 等）
- ✅ 节点表 Schema（NODE_DEGREE, NODE_FREQUENCY 等）
- ✅ 边表 Schema（EDGE_SOURCE, EDGE_TARGET, EDGE_WEIGHT 等）
- ✅ 社区表 Schema（COMMUNITY_ID, COMMUNITY_LEVEL 等）
- ✅ 最终输出列定义（7个表的列顺序）

### 3. 数据转换工具

**converters.py** - 提供 dataclass 与 DataFrame 之间的转换：

#### 3.1 通用转换函数
- `dataclass_to_dict()`: dataclass → 字典
- `dataclass_list_to_dataframe()`: dataclass 列表 → DataFrame
- `dataframe_to_dataclass_list()`: DataFrame → dataclass 列表

#### 3.2 专用转换函数（7对）
- `entities_to_dataframe()` / `dataframe_to_entities()`
- `relationships_to_dataframe()` / `dataframe_to_relationships()`
- `communities_to_dataframe()` / `dataframe_to_communities()`
- `community_reports_to_dataframe()` / `dataframe_to_community_reports()`
- `text_units_to_dataframe()` / `dataframe_to_text_units()`
- `documents_to_dataframe()` / `dataframe_to_documents()`
- `covariates_to_dataframe()` / `dataframe_to_covariates()`

#### 3.3 特殊处理
- 自动处理列表和字典类型（JSON 序列化）
- 处理 NaN 值（转换为 None）
- 字段过滤（只保留 dataclass 中定义的字段）

### 4. 数据验证工具

**validators.py** - 提供数据完整性验证：

#### 4.1 单个对象验证（7个函数）
- `validate_entity()`: 验证实体（ID、标题、排名）
- `validate_relationship()`: 验证关系（ID、源、目标、权重、排名）
- `validate_community()`: 验证社区（ID、标题、层级、大小）
- `validate_community_report()`: 验证报告（ID、标题、社区ID、排名、大小）
- `validate_text_unit()`: 验证文本单元（ID、文本、token数）
- `validate_document()`: 验证文档（ID、标题）
- `validate_covariate()`: 验证协变量（ID、主体ID）

#### 4.2 批量验证（7个函数）
- `validate_entities()`: 批量验证实体列表
- `validate_relationships()`: 批量验证关系列表
- `validate_communities()`: 批量验证社区列表
- `validate_community_reports()`: 批量验证报告列表
- `validate_text_units()`: 批量验证文本单元列表
- `validate_documents()`: 批量验证文档列表
- `validate_covariates()`: 批量验证协变量列表

返回格式：`dict[str, list[str]]` - ID 到错误消息列表的映射

## 🧪 测试验证

创建了完整的测试套件 `test_data_model.py`，包含：

1. ✅ **测试 1**: 实体创建和验证
2. ✅ **测试 2**: 关系创建和验证
3. ✅ **测试 3**: 社区创建和验证
4. ✅ **测试 4**: DataFrame 转换（双向转换和一致性验证）
5. ✅ **测试 5**: 文本单元和文档创建
6. ✅ **测试 6**: 社区报告创建

**测试结果**: 所有测试通过 ✓

## 📁 项目结构

```
graphrag_v2/
├── data_model/                      # 数据模型模块
│   ├── __init__.py                 # 模块导出
│   ├── identified.py               # 基础类：Identified
│   ├── named.py                    # 基础类：Named
│   ├── entity.py                   # 实体数据模型
│   ├── relationship.py             # 关系数据模型
│   ├── community.py                # 社区数据模型
│   ├── community_report.py         # 社区报告数据模型
│   ├── text_unit.py                # 文本单元数据模型
│   ├── document.py                 # 文档数据模型
│   ├── covariate.py                # 协变量数据模型
│   ├── schemas.py                  # Schema 常量定义
│   ├── converters.py               # 数据转换工具
│   └── validators.py               # 数据验证工具
├── test_data_model.py              # 数据模型测试
└── requirements.txt                # 更新的依赖包
```

## 💡 关键学习点

### 1. dataclass 继承的注意事项

```python
# ❌ 错误：父类有默认值，子类不能有非默认值字段
@dataclass
class Parent:
    id: str
    short_id: str | None = None  # 有默认值

@dataclass
class Child(Parent):
    title: str  # ❌ 错误！非默认值字段在默认值字段之后

# ✅ 正确：移除父类的默认值，或者子类字段也有默认值
@dataclass
class Parent:
    id: str
    short_id: str | None  # 无默认值

@dataclass
class Child(Parent):
    title: str  # ✓ 正确
```

### 2. DataFrame 与 dataclass 的转换

- 使用 `asdict()` 将 dataclass 转为字典
- 使用 `fields()` 获取 dataclass 的字段信息
- 列表和字典需要 JSON 序列化才能存储在 DataFrame 中
- 从 DataFrame 恢复时需要 JSON 反序列化

### 3. 数据验证的重要性

- 在数据进入系统时进行验证
- 提供清晰的错误消息
- 支持批量验证以提高效率

### 4. Schema 常量的作用

- 避免硬编码字符串
- 确保列名的一致性
- 便于重构和维护

## 🎓 与微软 GraphRAG 的对比

| 特性 | 微软 GraphRAG | 我们的实现 | 说明 |
|------|--------------|-----------|------|
| 数据类框架 | dataclass | dataclass | ✓ 完全一致 |
| 继承结构 | Identified → Named | Identified → Named | ✓ 完全一致 |
| from_dict 方法 | 支持 | 支持 | ✓ 完全一致 |
| Schema 常量 | 完善 | 完善 | ✓ 完全一致 |
| 数据转换 | 有 | 有 | ✓ 实现了核心功能 |
| 数据验证 | 基本 | 完善 | ✓ 增强了验证逻辑 |
| 文档注释 | 英文 | 中文 | 更易理解 |

## 🚀 下一步计划

阶段2已完成，接下来进入**阶段3：索引 Pipeline 重构**

主要任务：
1. 学习微软的 Pipeline 架构
2. 设计 Workflow 抽象
3. 实现文本分块工作流
4. 实现实体提取工作流
5. 实现社区检测工作流
6. 实现社区报告工作流
7. 实现嵌入工作流
8. 实现 Pipeline 编排器

## 📝 总结

阶段2成功完成了数据模型的标准化，我们：

1. ✅ 深入学习了微软 GraphRAG 的数据模型设计
2. ✅ 使用 dataclass 实现了 9 个核心数据类
3. ✅ 定义了完整的 Schema 常量
4. ✅ 实现了 dataclass 与 DataFrame 的双向转换
5. ✅ 创建了完善的数据验证工具
6. ✅ 编写了完整的测试套件

**关键成就**:
- 代码结构清晰，遵循最佳实践
- 完整的类型注解和中文文档
- 所有测试通过
- 为后续的索引和查询阶段提供了坚实的数据基础

这为阶段3的 Pipeline 开发做好了充分准备！🎉

