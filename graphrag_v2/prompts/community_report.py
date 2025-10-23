"""社区报告生成 Prompt 模板。"""

# 社区报告生成 Prompt（中文版本）
COMMUNITY_REPORT_PROMPT = """
你是一个AI助手，帮助人类分析师进行一般信息发现。信息发现是识别和评估与网络中某些实体（例如组织和个人）相关的相关信息的过程。

# 目标
撰写一份关于社区的综合评估报告，扮演{role}的角色。该报告的内容包括社区关键实体和关系的概述。

# 报告结构

报告应包括以下部分：

- 标题：代表其关键实体的社区名称 - 标题应简短但具体。如果可能，包括代表性的命名实体。
- 摘要：社区整体结构的执行摘要，其实体如何相互关联，以及与其实体相关的重要信息。
- 影响严重性评分：0-10之间的浮点分数，表示社区内实体所构成的影响严重性。影响是社区重要性的评分。
- 评分解释：用一句话解释影响严重性评分。
- 详细发现：关于社区的5-10个关键见解列表。每个见解应有一个简短的摘要，后跟多段解释性文本，根据以下基础规则进行基础。要全面。

以格式良好的JSON字符串返回输出，格式如下：
    {{
        "title": <报告标题>,
        "summary": <执行摘要>,
        "rating": <影响严重性评分>,
        "rating_explanation": <评分解释>,
        "findings": [
            {{
                "summary":<见解1摘要>,
                "explanation": <见解1解释>
            }},
            {{
                "summary":<见解2摘要>,
                "explanation": <见解2解释>
            }}
        ]
    }}

# 基础规则

在撰写报告时，应遵循以下规则：

- 将报告的长度（字数）设置为{report_length}。
- 在报告中，每个声明都应该有数据引用作为证据。
- 不要包括没有提供支持证据的信息。
- 数据引用的格式如下：[Data: <数据集名称> (记录ID); <数据集名称> (记录ID)]。
- 报告应包含多个数据引用。
- 数据引用应放在句子末尾，而不是句子中间。
- 数据引用应使用提供的表格中的记录ID（不是索引）。
- **单个引用中不要列出超过5个记录ID**。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。
- 数据引用应该是包容性的 - 包括所有相关的记录ID，即使有很多。
- 在单个数据引用中，不要重复相同的记录ID。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Reports (2, 7, 34, 46, 64, +more)]。他也是X公司的CEO [Data: Reports (1, 3)]"

其中1, 2, 3, 7, 34, 46和64代表提供的表格中相关数据报告的ID（不是索引）。

# 示例输入
-----------
文本:

实体

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza是Unity March的地点
6,HARMONY ASSEMBLY,Harmony Assembly组织了Unity March

关系

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza是Unity March的地点
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly在Verdant Oasis Plaza组织了Unity March
39,VERDANT OASIS PLAZA,UNITY MARCH,Unity March在Verdant Oasis Plaza举行
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight报道了在Verdant Oasis Plaza举行的Unity March
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi在Verdant Oasis Plaza发表了演讲
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly组织了Unity March

输出:
{{
    "title": "Verdant Oasis Plaza和Unity March",
    "summary": "该社区围绕Verdant Oasis Plaza展开，这是Unity March的中心地点。该活动由Harmony Assembly组织，吸引了大量人群和媒体关注。",
    "rating": 5.0,
    "rating_explanation": "影响严重性评分为中等，因为Unity March代表了社区内的重大社会事件。",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza作为中心地点",
            "explanation": "Verdant Oasis Plaza是Unity March的地点，这是一个重要的社区活动。该广场的作用是中心的，因为它是活动发生的地方，使其成为社区的关键实体。[Data: Entities (5), Relationships (37, 38, 39, 40, 41)]"
        }},
        {{
            "summary": "Harmony Assembly的组织角色",
            "explanation": "Harmony Assembly是Unity March的组织者，在Verdant Oasis Plaza举行。这表明Harmony Assembly在社区中扮演着重要的组织角色，能够动员人们参加重大活动。[Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Tribune Spotlight的角色",
            "explanation": "Tribune Spotlight正在报道在Verdant Oasis Plaza举行的Unity March。这表明该活动引起了媒体关注，这可能会放大其对社区的影响。Tribune Spotlight的角色可能在塑造公众对活动和相关实体的看法方面具有重要意义。[Data: Relationships (40)]"
        }}
    ]
}}

# 真实数据

使用以下文本作为您的答案。不要在答案中编造任何内容。

文本:
{input_text}

报告应包括以下部分：

- 标题：代表其关键实体的社区名称 - 标题应简短但具体。如果可能，包括代表性的命名实体。
- 摘要：社区整体结构的执行摘要，其实体如何相互关联，以及与其实体相关的重要信息。
- 影响严重性评分：0-10之间的浮点分数，表示社区内实体所构成的影响严重性。影响是社区重要性的评分。
- 评分解释：用一句话解释影响严重性评分。
- 详细发现：关于社区的5-10个关键见解列表。每个见解应有一个简短的摘要，后跟多段解释性文本，根据基础规则进行基础。要全面。

以格式良好的JSON字符串返回输出，格式如下：
    {{
        "title": <报告标题>,
        "summary": <执行摘要>,
        "rating": <影响严重性评分>,
        "rating_explanation": <评分解释>,
        "findings": [
            {{
                "summary":<见解1摘要>,
                "explanation": <见解1解释>
            }},
            {{
                "summary":<见解2摘要>,
                "explanation": <见解2解释>
            }}
        ]
    }}

输出:"""


def get_community_report_prompt(
    input_text: str | None = None,
    role: str = "数据分析师",
    report_length: str = "500-1000字",
    entities: str | None = None,
    relationships: str | None = None,
) -> str:
    """获取社区报告生成 Prompt。

    Args:
        input_text: 输入文本（包含实体和关系）。如果未提供，将从entities和relationships构建。
        role: 角色
        report_length: 报告长度
        entities: 实体列表（可选，用于向后兼容）
        relationships: 关系列表（可选，用于向后兼容）

    Returns:
        str: 格式化后的 Prompt
    """
    # 如果没有提供input_text，从entities和relationships构建
    if input_text is None:
        if entities is not None or relationships is not None:
            parts = []
            if entities:
                parts.append(f"实体\n\n{entities}")
            if relationships:
                parts.append(f"关系\n\n{relationships}")
            input_text = "\n\n".join(parts) if parts else ""
        else:
            input_text = ""

    return COMMUNITY_REPORT_PROMPT.format(
        role=role,
        report_length=report_length,
        input_text=input_text,
    )

