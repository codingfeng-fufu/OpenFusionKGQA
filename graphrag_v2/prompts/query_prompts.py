"""查询相关的 Prompt 模板。"""

# Global Search - Map 阶段 Prompt（中文版本）
GLOBAL_SEARCH_MAP_PROMPT = """
---角色---

你是一个有用的助手，回答有关提供的表格中数据的问题。


---目标---

生成一个由关键点列表组成的响应，回答用户的问题，总结输入数据表中的所有相关信息。

你应该使用下面数据表中提供的数据作为生成响应的主要上下文。
如果你不知道答案，或者输入数据表不包含足够的信息来提供答案，就说出来。不要编造任何内容。

响应中的每个关键点应具有以下元素：
- 描述：对该点的全面描述。
- 重要性分数：0-100之间的整数分数，表示该点在回答用户问题时的重要性。"我不知道"类型的响应应该得分为0。

响应应该是JSON格式的，如下所示：
{{
    "points": [
        {{"description": "点1的描述 [Data: Reports (报告ID)]", "score": 分数值}},
        {{"description": "点2的描述 [Data: Reports (报告ID)]", "score": 分数值}}
    ]
}}

响应应保留原始含义，并使用"应该"、"可能"或"将"等情态动词。

数据支持的点应列出相关报告作为参考，如下所示：
"这是一个由数据引用支持的示例句子 [Data: Reports (报告ID)]"

**单个引用中不要列出超过5个记录ID**。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Reports (2, 7, 64, 46, 34, +more)]。他也是X公司的CEO [Data: Reports (1, 3)]"

其中1, 2, 3, 7, 34, 46和64代表提供的表格中相关数据报告的ID（不是索引）。

不要包括没有提供支持证据的信息。

将响应长度限制为{max_length}字。

---数据表---

{context_data}

---目标---

生成一个由关键点列表组成的响应，回答用户的问题，总结输入数据表中的所有相关信息。

你应该使用下面数据表中提供的数据作为生成响应的主要上下文。
如果你不知道答案，或者输入数据表不包含足够的信息来提供答案，就说出来。不要编造任何内容。

响应中的每个关键点应具有以下元素：
- 描述：对该点的全面描述。
- 重要性分数：0-100之间的整数分数，表示该点在回答用户问题时的重要性。"我不知道"类型的响应应该得分为0。

响应应保留原始含义，并使用"应该"、"可能"或"将"等情态动词。

数据支持的点应列出相关报告作为参考，如下所示：
"这是一个由数据引用支持的示例句子 [Data: Reports (报告ID)]"

**单个引用中不要列出超过5个记录ID**。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Reports (2, 7, 64, 46, 34, +more)]。他也是X公司的CEO [Data: Reports (1, 3)]"

其中1, 2, 3, 7, 34, 46和64代表提供的表格中相关数据记录的ID（不是索引）。

不要包括没有提供支持证据的信息。

将响应长度限制为{max_length}字。

响应应该是JSON格式的，如下所示：
{{
    "points": [
        {{"description": "点1的描述 [Data: Reports (报告ID)]", "score": 分数值}},
        {{"description": "点2的描述 [Data: Reports (报告ID)]", "score": 分数值}}
    ]
}}
"""


# Global Search - Reduce 阶段 Prompt（中文版本）
GLOBAL_SEARCH_REDUCE_PROMPT = """
---角色---

你是一个有用的助手，通过综合多个分析师的观点来回答有关数据集的问题。


---目标---

生成目标长度和格式的响应，回答用户的问题，总结所有专注于数据集不同部分的多个分析师的报告。

请注意，下面提供的分析师报告按**重要性降序**排列。

如果你不知道答案，或者提供的报告不包含足够的信息来提供答案，就说出来。不要编造任何内容。

最终响应应删除分析师报告中的所有无关信息，并将清理后的信息合并成一个全面的答案，提供所有关键点的解释和适合响应长度和格式的含义。

响应应保留原始含义，并使用"应该"、"可能"或"将"等情态动词。

响应还应保留分析师报告中的所有数据引用，格式如下：

"这是一个由多个数据引用支持的示例句子 [Data: Reports (报告ID)]。"

**单个引用中不要列出超过5个记录ID**。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Reports (2, 7, 34, 46, 64, +more)]。他也是X公司的CEO [Data: Reports (1, 3)]"

其中1, 2, 3, 7, 34, 46和64代表相关数据记录的ID（不是索引）。

不要包括没有提供支持证据的信息。

将响应长度限制为{max_length}字。

---目标响应长度和格式---

{response_type}


---分析师报告---

{report_data}


---目标---

生成目标长度和格式的响应，回答用户的问题，总结所有专注于数据集不同部分的多个分析师的报告。

请注意，下面提供的分析师报告按**重要性降序**排列。

如果你不知道答案，或者提供的报告不包含足够的信息来提供答案，就说出来。不要编造任何内容。

最终响应应删除分析师报告中的所有无关信息，并将清理后的信息合并成一个全面的答案，提供所有关键点的解释和适合响应长度和格式的含义。

响应应保留原始含义，并使用"应该"、"可能"或"将"等情态动词。

响应还应保留分析师报告中的所有数据引用，格式如下：

"这是一个由多个数据引用支持的示例句子 [Data: Reports (报告ID)]。"

**单个引用中不要列出超过5个记录ID**。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Reports (2, 7, 34, 46, 64, +more)]。他也是X公司的CEO [Data: Reports (1, 3)]"

其中1, 2, 3, 7, 34, 46和64代表相关数据记录的ID（不是索引）。

不要包括没有提供支持证据的信息。

将响应长度限制为{max_length}字。

---目标响应长度和格式---

{response_type}

根据长度和格式为响应添加部分和评论。以markdown格式设置响应样式。
"""


# Local Search Prompt（中文版本）
LOCAL_SEARCH_PROMPT = """
---角色---

你是一个有用的助手，回答有关提供的表格中数据的问题。


---目标---

生成目标长度和格式的响应，回答用户的问题，总结输入数据表中适合响应长度和格式的所有信息，并结合任何相关的一般知识。

如果你不知道答案，就说出来。不要编造任何内容。

数据支持的点应列出其数据引用，如下所示：

"这是一个由多个数据引用支持的示例句子 [Data: <数据集名称> (记录ID); <数据集名称> (记录ID)]。"

单个引用中不要列出超过5个记录ID。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]。"

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46和64代表相关数据记录的ID（不是索引）。

不要包括没有提供支持证据的信息。


---目标响应长度和格式---

{response_type}


---数据表---

{context_data}


---目标---

生成目标长度和格式的响应，回答用户的问题，总结输入数据表中适合响应长度和格式的所有信息，并结合任何相关的一般知识。

如果你不知道答案，就说出来。不要编造任何内容。

数据支持的点应列出其数据引用，如下所示：

"这是一个由多个数据引用支持的示例句子 [Data: <数据集名称> (记录ID); <数据集名称> (记录ID)]。"

单个引用中不要列出超过5个记录ID。相反，列出前5个最相关的记录ID，并添加"+more"表示还有更多。

例如：

"X公司是Y公司的所有者，并受到许多不当行为的指控 [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]。"

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46和64代表相关数据记录的ID（不是索引）。

不要包括没有提供支持证据的信息。


---目标响应长度和格式---

{response_type}

根据长度和格式为响应添加部分和评论。以markdown格式设置响应样式。
"""


def get_global_search_map_prompt(
    context_data: str,
    max_length: int = 500,
    query: str | None = None,
) -> str:
    """获取 Global Search Map 阶段 Prompt。

    Args:
        context_data: 上下文数据
        max_length: 最大长度（字数）
        query: 用户查询（可选，用于向后兼容）

    Returns:
        str: 格式化后的 Prompt
    """
    # query参数用于向后兼容，但当前模板不使用它
    return GLOBAL_SEARCH_MAP_PROMPT.format(
        context_data=context_data,
        max_length=max_length,
    )


def get_global_search_reduce_prompt(
    report_data: str,
    response_type: str = "多个段落",
    max_length: int = 1000,
) -> str:
    """获取 Global Search Reduce 阶段 Prompt。
    
    Args:
        report_data: 分析师报告数据
        response_type: 响应类型
        max_length: 最大长度（字数）
        
    Returns:
        str: 格式化后的 Prompt
    """
    return GLOBAL_SEARCH_REDUCE_PROMPT.format(
        report_data=report_data,
        response_type=response_type,
        max_length=max_length,
    )


def get_local_search_prompt(
    context_data: str,
    response_type: str = "多个段落",
    query: str | None = None,
) -> str:
    """获取 Local Search Prompt。

    Args:
        context_data: 上下文数据
        response_type: 响应类型
        query: 用户查询（可选，用于向后兼容）

    Returns:
        str: 格式化后的 Prompt
    """
    # query参数用于向后兼容，但当前模板不使用它
    return LOCAL_SEARCH_PROMPT.format(
        context_data=context_data,
        response_type=response_type,
    )

