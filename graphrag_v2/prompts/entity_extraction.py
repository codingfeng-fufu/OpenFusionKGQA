"""实体提取 Prompt 模板。"""

# 实体提取 Prompt（中文版本）
ENTITY_EXTRACTION_PROMPT = """
-目标-
给定一个可能与此活动相关的文本文档和实体类型列表，从文本中识别这些类型的所有实体，以及识别出的实体之间的所有关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，首字母大写
- entity_type: 以下类型之一：[{entity_types}]
- entity_description: 实体属性和活动的全面描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1中识别的实体中，识别所有*明确相关*的（source_entity, target_entity）对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体的名称，如步骤1中识别的
- target_entity: 目标实体的名称，如步骤1中识别的
- relationship_description: 解释为什么你认为源实体和目标实体彼此相关
- relationship_strength: 表示源实体和目标实体之间关系强度的数字分数（1-10）
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 以中文返回输出，作为步骤1和步骤2中识别的所有实体和关系的单个列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

######################
-示例-
######################
{examples}

######################
-真实数据-
######################
实体类型: {entity_types}
文本: {input_text}
######################
输出:"""


# Few-shot 示例模板
EXAMPLE_TEMPLATE = """
示例 {n}:

实体类型: [{entity_types}]
文本:
{input_text}
------------------------
输出:
{output}
#############################

"""


# 中文示例1：科技公司
EXAMPLE_1 = {
    "n": 1,
    "entity_types": "组织,人物",
    "input_text": """微软公司计划在周一和周四召开董事会会议，公司计划在周四下午1:30发布最新的战略决策，随后将举行新闻发布会，微软CEO萨提亚·纳德拉将回答问题。投资者预计战略委员会将维持其基准利率在3.5%-3.75%的范围内。""",
    "output": """("entity"<|>微软<|>组织<|>微软是一家科技公司，将在周一和周四召开董事会会议)
<|>
("entity"<|>萨提亚·纳德拉<|>人物<|>萨提亚·纳德拉是微软的CEO)
<|>
("entity"<|>战略委员会<|>组织<|>战略委员会是微软的决策机构，负责制定公司战略)
<|>
("relationship"<|>萨提亚·纳德拉<|>微软<|>萨提亚·纳德拉是微软的CEO，将在新闻发布会上回答问题<|>9)
<|COMPLETE|>"""
}


# 中文示例2：商业新闻
EXAMPLE_2 = {
    "n": 2,
    "entity_types": "组织",
    "input_text": """字节跳动（ByteDance）的股票在周四全球交易所的首日交易中飙升。但IPO专家警告说，这家互联网公司在公开市场上的首次亮相并不代表其他新上市公司的表现。

字节跳动曾是一家上市公司，2014年被腾讯控股私有化。这家知名的互联网公司表示，它为85%的高端智能手机提供技术支持。""",
    "output": """("entity"<|>字节跳动<|>组织<|>字节跳动是一家现在在全球交易所上市的股票公司，为85%的高端智能手机提供技术支持)
<|>
("entity"<|>腾讯控股<|>组织<|>腾讯控股是一家曾经拥有字节跳动的公司)
<|>
("relationship"<|>字节跳动<|>腾讯控股<|>腾讯控股在2014年至今拥有字节跳动<|>5)
<|COMPLETE|>"""
}


# 中文示例3：国际新闻
EXAMPLE_3 = {
    "n": 3,
    "entity_types": "组织,地点,人物",
    "input_text": """五名在德黑兰被监禁8年的美国人被广泛认为是人质，他们正在返回美国的途中。

这次由卡塔尔协调的交换在80亿美元的伊朗资金转移到多哈（卡塔尔首都）的金融机构后完成。

交换在德黑兰首都开始，导致四名男子和一名女子（他们也是伊朗国民）登上了飞往多哈的包机。

他们受到美国高级官员的欢迎，现在正在前往美国首都华盛顿的途中。

这些美国人包括39岁的商人约翰·史密斯，他被关押在德黑兰的埃文监狱，以及59岁的记者莎拉·约翰逊和53岁的环保主义者迈克尔·布朗，后者还持有加拿大国籍。""",
    "output": """("entity"<|>德黑兰<|>地点<|>德黑兰关押美国人质)
<|>
("entity"<|>美国<|>地点<|>寻求释放人质的国家)
<|>
("entity"<|>卡塔尔<|>地点<|>协商用资金交换人质的国家)
<|>
("entity"<|>多哈<|>地点<|>卡塔尔的首都城市)
<|>
("entity"<|>华盛顿<|>地点<|>美国的首都城市)
<|>
("entity"<|>约翰·史密斯<|>人物<|>在德黑兰埃文监狱度过时间的美国人)
<|>
("entity"<|>埃文监狱<|>地点<|>德黑兰的监狱)
<|>
("entity"<|>莎拉·约翰逊<|>人物<|>被关押的美国记者)
<|>
("entity"<|>迈克尔·布朗<|>人物<|>加拿大国民和被关押的环保主义者)
<|>
("relationship"<|>德黑兰<|>美国<|>德黑兰与美国协商人质交换<|>2)
<|>
("relationship"<|>卡塔尔<|>美国<|>卡塔尔促成了德黑兰和美国之间的人质交换<|>2)
<|>
("relationship"<|>卡塔尔<|>德黑兰<|>卡塔尔促成了德黑兰和美国之间的人质交换<|>2)
<|>
("relationship"<|>约翰·史密斯<|>埃文监狱<|>约翰·史密斯是埃文监狱的囚犯<|>8)
<|>
("relationship"<|>约翰·史密斯<|>迈克尔·布朗<|>约翰·史密斯和迈克尔·布朗在同一次人质释放中被交换<|>2)
<|>
("relationship"<|>约翰·史密斯<|>莎拉·约翰逊<|>约翰·史密斯和莎拉·约翰逊在同一次人质释放中被交换<|>2)
<|>
("relationship"<|>迈克尔·布朗<|>莎拉·约翰逊<|>迈克尔·布朗和莎拉·约翰逊在同一次人质释放中被交换<|>2)
<|>
("relationship"<|>约翰·史密斯<|>德黑兰<|>约翰·史密斯是德黑兰的人质<|>2)
<|>
("relationship"<|>迈克尔·布朗<|>德黑兰<|>迈克尔·布朗是德黑兰的人质<|>2)
<|>
("relationship"<|>莎拉·约翰逊<|>德黑兰<|>莎拉·约翰逊是德黑兰的人质<|>2)
<|COMPLETE|>"""
}


# 默认分隔符
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "<|>"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


def format_examples(
    examples: list[dict],
    tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
    record_delimiter: str = DEFAULT_RECORD_DELIMITER,
) -> str:
    """格式化示例。
    
    Args:
        examples: 示例列表
        tuple_delimiter: 元组分隔符
        record_delimiter: 记录分隔符
        
    Returns:
        str: 格式化后的示例
    """
    formatted_examples = []
    for example in examples:
        formatted_example = EXAMPLE_TEMPLATE.format(
            n=example["n"],
            entity_types=example["entity_types"],
            input_text=example["input_text"],
            output=example["output"],
        )
        formatted_examples.append(formatted_example)
    
    return "\n".join(formatted_examples)


def get_entity_extraction_prompt(
    entity_types: list[str],
    input_text: str,
    tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
    record_delimiter: str = DEFAULT_RECORD_DELIMITER,
    completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
    include_examples: bool = True,
) -> str:
    """获取实体提取 Prompt。
    
    Args:
        entity_types: 实体类型列表
        input_text: 输入文本
        tuple_delimiter: 元组分隔符
        record_delimiter: 记录分隔符
        completion_delimiter: 完成分隔符
        include_examples: 是否包含示例
        
    Returns:
        str: 格式化后的 Prompt
    """
    # 格式化示例
    if include_examples:
        examples = format_examples(
            [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
        )
    else:
        examples = ""
    
    # 格式化 Prompt
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(entity_types),
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        examples=examples,
        input_text=input_text,
    )
    
    return prompt

