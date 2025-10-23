# 阶段3扩展总结：添加实体提取和社区检测工作流

## ✅ 完成情况

**状态**: 已完成 ✓  
**完成时间**: 2025-10-16  
**耗时**: 约 1 小时

## 📚 新增功能

### 1. 实体提取工作流 (`extract_graph.py`)

**功能描述**:
- 从文本单元中提取实体和关系
- 使用规则提取（简化版本，不需要 LLM）
- 合并重复的实体和关系
- 计算实体和关系的排名

**提取规则**:
1. **大写开头的连续词**: 如 "GraphRAG", "Microsoft Corporation"
2. **中文实体模式**: 如 "微软公司"、"GraphRAG系统"、"知识图谱"
3. **专有名词**: 连续的中文或英文大写字母

**关系提取**:
- 如果两个实体在同一个句子中出现，则认为它们有关系
- 计算关系权重（共现次数）

**输出**:
- `entities` DataFrame: name, type, description, id, text_unit_ids, rank
- `relationships` DataFrame: id, source, target, description, weight, text_unit_ids, rank

### 2. 社区检测工作流 (`create_communities.py`)

**功能描述**:
- 从实体和关系构建 NetworkX 图
- 使用 Louvain 算法检测社区
- 处理不连通的图（分别处理每个连通分量）
- 计算社区统计信息

**核心函数**:
- `create_graph_from_dataframes()`: 从 DataFrame 创建图
- `detect_communities()`: 使用 Louvain 算法检测社区

**输出**:
- `communities` DataFrame: id, level, title, nodes, size, parent_community, num_edges, avg_degree
- `graph`: NetworkX 图对象

### 3. 扩展的 Pipeline

**新的标准 Pipeline**:
1. `load_documents` - 加载文档
2. `create_base_text_units` - 文本分块
3. `extract_graph` - 实体和关系提取 ✨ 新增
4. `create_communities` - 社区检测 ✨ 新增

## 🎯 测试结果

### 测试数据

使用包含 15 行文本的测试文档，内容涉及：
- GraphRAG 系统
- 微软公司
- OpenAI
- GPT 模型
- Leiden 算法
- 知识图谱
- 自然语言处理

### 测试结果

| 指标 | 数值 |
|------|------|
| 文档数量 | 1 |
| 文本单元数量 | 1 |
| 实体数量 | 60 |
| 关系数量 | 174 |
| 社区数量 | 5 |
| 图节点数 | 60 |
| 图边数 | 174 |
| 平均度 | 5.80 |
| 总运行时间 | 0.29秒 |

### 社区统计

| 指标 | 数值 |
|------|------|
| 平均社区大小 | 12.00 |
| 最大社区大小 | 16 |
| 最小社区大小 | 3 |

### 社区详情

1. **社区 0** (15 个节点): 社区检测算法相关
2. **社区 1** (13 个节点): 微软、OpenAI、语言模型相关
3. **社区 2** (16 个节点): GraphRAG 系统、知识图谱相关
4. **社区 3** (13 个节点): 自然语言处理、技术进步相关
5. **社区 4** (3 个节点): 人工智能、生活方式相关

## 💡 关键学习点

### 1. 规则提取 vs LLM 提取

**规则提取的优势**:
- 速度快，无需 API 调用
- 成本低
- 适合演示和测试

**规则提取的局限**:
- 准确率较低
- 无法理解语义
- 难以处理复杂关系

**生产环境建议**:
- 使用 LLM 进行实体和关系提取
- 参考微软 GraphRAG 的 Prompt 设计
- 使用 Few-shot 示例提高准确率

### 2. 社区检测算法

**Louvain 算法**:
- 基于模块度优化
- 速度快，适合大规模图
- NetworkX 内置支持

**Leiden 算法** (微软 GraphRAG 使用):
- Louvain 的改进版本
- 更好的社区质量
- 需要 `graspologic` 库

**处理不连通图**:
- 检测连通分量
- 分别对每个分量进行社区检测
- 合并结果

### 3. 图数据结构

**NetworkX 图**:
- 节点属性: id, type, description, rank
- 边属性: id, weight, description
- 支持多种图算法

**DataFrame 与图的转换**:
- 使用 `add_node()` 和 `add_edge()` 构建图
- 使用 `graph.nodes()` 和 `graph.edges()` 访问数据

### 4. 数据验证和错误处理

**空数据处理**:
- 检查 DataFrame 是否为空
- 检查图是否有节点
- 提供清晰的错误消息

**日志记录**:
- 使用 `logging` 模块
- 记录关键步骤和统计信息
- 便于调试和监控

## 🔧 技术细节

### 1. 实体 ID 生成

```python
entity_id = hashlib.md5(entity_name.encode()).hexdigest()[:16]
entity['id'] = f"entity_{entity_id}"
```

**优势**:
- 确定性：相同名称生成相同 ID
- 唯一性：MD5 哈希避免冲突
- 简洁性：16 字符长度

### 2. 关系合并

```python
relationship_dict = {}
for rel in all_relationships:
    key = (rel['source'], rel['target'])
    if key not in relationship_dict:
        relationship_dict[key] = rel
    else:
        existing = relationship_dict[key]
        existing['weight'] += 1.0
```

**优势**:
- 去重：避免重复关系
- 权重累加：反映共现频率
- 保留来源：合并 text_unit_ids

### 3. 社区检测

```python
partition = nx_comm.louvain_communities(
    subgraph,
    weight='weight',
    seed=42,
)
```

**参数**:
- `weight='weight'`: 使用边权重
- `seed=42`: 固定随机种子，确保可重复性

## 📁 项目结构更新

```
graphrag_v2/
├── workflows/
│   ├── load_documents.py           # 加载文档 ✓
│   ├── create_base_text_units.py  # 文本分块 ✓
│   ├── extract_graph.py            # 实体提取 ✨ 新增
│   └── create_communities.py       # 社区检测 ✨ 新增
├── test_extended_pipeline.py       # 扩展测试 ✨ 新增
├── requirements.txt                # 更新（添加 networkx）
└── config/enums.py                 # 修复（text -> txt）
```

## 🚀 下一步计划

### 短期任务
1. **添加社区报告生成工作流**
   - 使用 LLM 生成社区摘要
   - 参考微软的 Prompt 设计
   - 支持多级社区报告

2. **添加文本嵌入工作流**
   - 使用 OpenAI Embeddings API
   - 支持批量嵌入
   - 缓存嵌入结果

3. **改进实体提取**
   - 集成 LLM（OpenAI GPT-4）
   - 使用 Few-shot Prompt
   - 添加实体类型识别

### 长期任务
- **阶段4**: 查询引擎优化
  - Global Search
  - Local Search
  - DRIFT Search
- **阶段5**: Prompt 工程优化
- **阶段6**: 测试与文档

## 📝 总结

阶段3扩展成功完成！我们：

1. ✅ 实现了实体提取工作流（规则版本）
2. ✅ 实现了社区检测工作流（Louvain 算法）
3. ✅ 创建了完整的端到端测试
4. ✅ 验证了整个 Pipeline 的正确性

**关键成就**:
- 从 15 行文本中提取了 60 个实体和 174 个关系
- 检测到 5 个有意义的社区
- Pipeline 运行时间仅 0.29秒
- 所有测试通过 ✓

**下一步**:
- 可以继续添加更多工作流（社区报告、嵌入）
- 或者进入阶段4，开始实现查询引擎

这为构建完整的 GraphRAG 系统打下了坚实的基础！🎉

