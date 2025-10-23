# GraphRAG v2 使用示例

本文档提供 GraphRAG v2 的使用示例。

## 目录

1. [Prompt 模板使用](#prompt-模板使用)
2. [GLM 客户端使用](#glm-客户端使用)
3. [实体提取示例](#实体提取示例)
4. [社区报告生成示例](#社区报告生成示例)
5. [查询示例](#查询示例)

---

## Prompt 模板使用

### 基础模板

```python
from graphrag_v2.prompts.base import PromptTemplate

# 创建模板
template = PromptTemplate("你好，{name:世界}！今天天气{weather}。")

# 格式化
result = template.format(name="Alice", weather="晴朗")
print(result)  # "你好，Alice！今天天气晴朗。"

# 使用默认值
result = template.format(weather="多云")
print(result)  # "你好，世界！今天天气多云。"
```

### 条件渲染

```python
template = PromptTemplate("""
你好！
{?premium}
感谢您成为高级会员！
{/premium}
""")

# 高级会员
result = template.format(premium=True)
# 输出包含感谢信息

# 普通用户
result = template.format(premium=False)
# 输出不包含感谢信息
```

### Prompt 库

```python
from graphrag_v2.prompts.base import PromptLibrary

# 创建库
library = PromptLibrary()

# 注册模板
library.register("greeting", "你好，{name}！")
library.register("farewell", "再见，{name}！")

# 使用模板
greeting = library.format("greeting", name="Alice")
farewell = library.format("farewell", name="Alice")
```

---

## GLM 客户端使用

### 基础使用

```python
from graphrag_v2.llm import GLMClient

# 初始化客户端（自动从环境变量读取 ZHIPUAI_API_KEY）
client = GLMClient(model="glm-4")

# 调用 API
messages = [
    {"role": "system", "content": "你是一个专业的助手。"},
    {"role": "user", "content": "什么是 GraphRAG？"}
]

response = client.chat_completion(
    messages=messages,
    temperature=0.7,
    max_tokens=1000,
)

print(response)
```

### 使用 API Key

```python
# 方式1：通过参数传递
client = GLMClient(api_key="your-api-key-here")

# 方式2：通过环境变量
import os
os.environ["ZHIPUAI_API_KEY"] = "your-api-key-here"
client = GLMClient()
```

### Mock 模式

```python
# 不提供 API key，自动使用 mock 模式
client = GLMClient()

# 调用会返回 mock 响应
response = client.chat_completion(messages)
```

### 统计信息

```python
client = GLMClient()

# 进行多次调用
for i in range(10):
    client.chat_completion(messages)

# 获取统计
stats = client.get_stats()
print(f"总调用次数: {stats['total_calls']}")
print(f"总 Token 数: {stats['total_tokens']}")
print(f"总错误次数: {stats['total_errors']}")

# 重置统计
client.reset_stats()
```

---

## 实体提取示例

### 基础使用

```python
from graphrag_v2.prompts import get_entity_extraction_prompt
from graphrag_v2.llm import GLMClient

# 准备文本
text = """
GraphRAG 是微软研究院开发的一种创新技术，它将知识图谱与检索增强生成（RAG）相结合。
该技术由微软的研究团队在2024年发布，旨在提升大语言模型在复杂问答任务中的表现。
"""

# 生成 Prompt
prompt = get_entity_extraction_prompt(
    entity_types=["组织", "技术", "人物"],
    input_text=text,
    include_examples=True,  # 包含 Few-shot 示例
)

# 调用 GLM
client = GLMClient()
messages = [
    {"role": "system", "content": "你是一个专业的信息提取助手。"},
    {"role": "user", "content": prompt}
]

response = client.chat_completion(messages, temperature=0.3)
print(response)
```

### 解析响应

```python
# 解析实体和关系
lines = response.split("<|>")

entities = []
relationships = []

for line in lines:
    line = line.strip()
    if line.startswith('("entity"'):
        # 解析实体
        # 格式: ("entity"<|>名称<|>类型<|>描述)
        entities.append(line)
    elif line.startswith('("relationship"'):
        # 解析关系
        # 格式: ("relationship"<|>源<|>目标<|>描述<|>强度)
        relationships.append(line)

print(f"提取到 {len(entities)} 个实体")
print(f"提取到 {len(relationships)} 个关系")
```

### 自定义分隔符

```python
prompt = get_entity_extraction_prompt(
    entity_types=["组织", "技术"],
    input_text=text,
    tuple_delimiter="||",
    record_delimiter="##",
    completion_delimiter="##DONE##",
)
```

---

## 社区报告生成示例

### 基础使用

```python
from graphrag_v2.prompts import get_community_report_prompt
from graphrag_v2.llm import GLMClient
import json

# 准备社区数据
community_data = """
实体

id,entity,description
1,GraphRAG,一种结合知识图谱和检索增强生成的技术
2,微软,GraphRAG 的开发者
3,Leiden算法,用于社区检测的算法

关系

id,source,target,description
1,微软,GraphRAG,微软开发了 GraphRAG 技术
2,GraphRAG,Leiden算法,GraphRAG 使用 Leiden 算法进行社区检测
"""

# 生成 Prompt
prompt = get_community_report_prompt(
    input_text=community_data,
    role="技术分析师",
    report_length="500-1000字",
)

# 调用 GLM
client = GLMClient()
messages = [
    {"role": "system", "content": "你是一个专业的数据分析师。"},
    {"role": "user", "content": prompt}
]

response = client.chat_completion(messages, temperature=0.5)

# 解析 JSON 响应
report = json.loads(response)
print(f"标题: {report['title']}")
print(f"摘要: {report['summary']}")
print(f"评分: {report['rating']}")
print(f"发现数量: {len(report['findings'])}")
```

---

## 查询示例

### Global Search

```python
from graphrag_v2.prompts import (
    get_global_search_map_prompt,
    get_global_search_reduce_prompt,
)
from graphrag_v2.llm import GLMClient
import json

client = GLMClient()

# 准备社区报告数据
reports = """
报告 ID: 1
标题: GraphRAG 技术社区
摘要: GraphRAG 是一种结合知识图谱和 RAG 的技术，由微软开发。

报告 ID: 2
标题: 社区检测算法
摘要: Leiden 算法用于检测社区结构，在 GraphRAG 中发挥重要作用。
"""

# Map 阶段
map_prompt = get_global_search_map_prompt(
    context_data=reports,
    max_length=200,
)

messages = [
    {"role": "system", "content": "你是一个专业的数据分析师。"},
    {"role": "user", "content": map_prompt + "\n\n问题: GraphRAG 的主要特点是什么？"}
]

map_response = client.chat_completion(messages, temperature=0.3)
map_result = json.loads(map_response)

print("Map 阶段关键点:")
for point in map_result["points"]:
    print(f"- {point['description']} (分数: {point['score']})")

# Reduce 阶段
reduce_prompt = get_global_search_reduce_prompt(
    report_data=f"分析师报告:\n{map_response}",
    response_type="简短段落",
    max_length=300,
)

messages = [
    {"role": "system", "content": "你是一个专业的数据分析师。"},
    {"role": "user", "content": reduce_prompt + "\n\n问题: GraphRAG 的主要特点是什么？"}
]

reduce_response = client.chat_completion(messages, temperature=0.5)
print("\nReduce 阶段最终答案:")
print(reduce_response)
```

### Local Search

```python
from graphrag_v2.prompts import get_local_search_prompt
from graphrag_v2.llm import GLMClient

# 准备数据表
context_data = """
实体表:
id,entity,type,description
1,GraphRAG,技术,结合知识图谱和 RAG 的技术
2,微软,组织,GraphRAG 的开发者

关系表:
id,source,target,description
1,微软,GraphRAG,微软开发了 GraphRAG

社区表:
id,title,summary
1,GraphRAG 技术生态,包含 GraphRAG 及其相关技术和组织
"""

# 生成 Prompt
prompt = get_local_search_prompt(
    context_data=context_data,
    response_type="简短段落",
)

# 调用 GLM
client = GLMClient()
messages = [
    {"role": "system", "content": "你是一个专业的问答助手。"},
    {"role": "user", "content": prompt + "\n\n问题: GraphRAG 是什么？"}
]

response = client.chat_completion(messages, temperature=0.5)
print(response)
```

---

## 完整示例：端到端流程

```python
from graphrag_v2.prompts import (
    get_entity_extraction_prompt,
    get_community_report_prompt,
    get_local_search_prompt,
)
from graphrag_v2.llm import GLMClient
import json

# 初始化客户端
client = GLMClient()

# 步骤1：提取实体
text = "GraphRAG 是微软开发的技术..."
entity_prompt = get_entity_extraction_prompt(
    entity_types=["组织", "技术"],
    input_text=text,
)

entities_response = client.chat_completion([
    {"role": "system", "content": "你是一个专业的信息提取助手。"},
    {"role": "user", "content": entity_prompt}
])

print("步骤1：实体提取完成")

# 步骤2：生成社区报告
community_data = f"实体和关系数据:\n{entities_response}"
report_prompt = get_community_report_prompt(
    input_text=community_data,
)

report_response = client.chat_completion([
    {"role": "system", "content": "你是一个专业的数据分析师。"},
    {"role": "user", "content": report_prompt}
])

report = json.loads(report_response)
print(f"步骤2：社区报告生成完成 - {report['title']}")

# 步骤3：回答问题
search_prompt = get_local_search_prompt(
    context_data=community_data,
)

answer = client.chat_completion([
    {"role": "system", "content": "你是一个专业的问答助手。"},
    {"role": "user", "content": search_prompt + "\n\n问题: GraphRAG 是什么？"}
])

print(f"步骤3：问题回答完成")
print(f"\n答案:\n{answer}")

# 查看统计
stats = client.get_stats()
print(f"\n总调用次数: {stats['total_calls']}")
print(f"总 Token 数: {stats['total_tokens']}")
```

---

## 环境配置

### 设置 API Key

```bash
# Linux/Mac
export ZHIPUAI_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:ZHIPUAI_API_KEY="your-api-key-here"

# Windows (CMD)
set ZHIPUAI_API_KEY=your-api-key-here
```

### 安装依赖

```bash
pip install zhipuai
```

---

## 最佳实践

1. **使用 Mock 模式进行开发**
   - 在开发和测试阶段不提供 API key
   - 避免不必要的 API 调用费用

2. **合理设置温度参数**
   - 实体提取：0.1-0.3（需要精确）
   - 社区报告：0.5-0.7（需要创造性）
   - 问答：0.3-0.5（平衡精确和流畅）

3. **控制 Token 使用**
   - 设置合理的 max_tokens
   - 使用统计功能跟踪使用量

4. **错误处理**
   - 使用 try-except 捕获异常
   - 检查响应格式是否正确
   - 实现重试机制

5. **Prompt 优化**
   - 根据实际效果调整 Prompt
   - 添加更多 Few-shot 示例
   - 明确输出格式要求

---

## 故障排除

### 问题1：API Key 错误

```python
# 检查环境变量
import os
print(os.getenv("ZHIPUAI_API_KEY"))

# 或直接传递
client = GLMClient(api_key="your-key")
```

### 问题2：JSON 解析失败

```python
import json

try:
    report = json.loads(response)
except json.JSONDecodeError as e:
    print(f"JSON 解析失败: {e}")
    print(f"原始响应: {response}")
```

### 问题3：响应格式不正确

```python
# 检查响应内容
print(f"响应长度: {len(response)}")
print(f"前100字符: {response[:100]}")

# 调整 Prompt 或温度参数
```

---

## 更多资源

- [PHASE5_SUMMARY.md](PHASE5_SUMMARY.md) - 阶段5详细总结
- [README.md](README.md) - 项目概述
- [智谱 AI 文档](https://open.bigmodel.cn/) - GLM API 文档

