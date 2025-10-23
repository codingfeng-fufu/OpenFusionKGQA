"""数据模型的 Schema 常量定义。

定义了所有 Parquet 文件的列名和最终输出列的顺序。
这些常量用于确保数据处理的一致性。
"""

# ============================================================
# 通用字段
# ============================================================

ID = "id"
"""唯一标识符。"""

SHORT_ID = "human_readable_id"
"""人类可读的短 ID。"""

TITLE = "title"
"""标题/名称。"""

DESCRIPTION = "description"
"""描述。"""

TYPE = "type"
"""类型。"""

# ============================================================
# 节点（实体）表 Schema
# ============================================================

NODE_DEGREE = "degree"
"""节点度数（连接的边数量）。"""

NODE_FREQUENCY = "frequency"
"""节点频率（出现次数）。"""

NODE_DETAILS = "node_details"
"""节点详细信息。"""

NODE_X = "x"
"""节点 X 坐标（用于可视化）。"""

NODE_Y = "y"
"""节点 Y 坐标（用于可视化）。"""

# ============================================================
# 边（关系）表 Schema
# ============================================================

EDGE_SOURCE = "source"
"""边的源节点。"""

EDGE_TARGET = "target"
"""边的目标节点。"""

EDGE_DEGREE = "combined_degree"
"""边的组合度数。"""

EDGE_DETAILS = "edge_details"
"""边的详细信息。"""

EDGE_WEIGHT = "weight"
"""边的权重。"""

# ============================================================
# 声明（Claim）表 Schema
# ============================================================

CLAIM_SUBJECT = "subject_id"
"""声明的主体 ID。"""

CLAIM_STATUS = "status"
"""声明的状态。"""

CLAIM_DETAILS = "claim_details"
"""声明的详细信息。"""

# ============================================================
# 社区层次表 Schema
# ============================================================

SUB_COMMUNITY = "sub_community"
"""子社区。"""

# ============================================================
# 社区上下文表 Schema
# ============================================================

ALL_CONTEXT = "all_context"
"""所有上下文。"""

CONTEXT_STRING = "context_string"
"""上下文字符串。"""

CONTEXT_SIZE = "context_size"
"""上下文大小。"""

CONTEXT_EXCEED_FLAG = "context_exceed_limit"
"""上下文是否超出限制的标志。"""

# ============================================================
# 社区报告表 Schema
# ============================================================

COMMUNITY_ID = "community"
"""社区 ID。"""

COMMUNITY_LEVEL = "level"
"""社区层级。"""

COMMUNITY_PARENT = "parent"
"""父社区 ID。"""

COMMUNITY_CHILDREN = "children"
"""子社区 ID 列表。"""

SUMMARY = "summary"
"""摘要。"""

FINDINGS = "findings"
"""发现/要点。"""

RATING = "rank"
"""评分/排名。"""

EXPLANATION = "rating_explanation"
"""评分说明。"""

FULL_CONTENT = "full_content"
"""完整内容。"""

FULL_CONTENT_JSON = "full_content_json"
"""完整内容（JSON 格式）。"""

# ============================================================
# 关联 ID 字段
# ============================================================

ENTITY_IDS = "entity_ids"
"""实体 ID 列表。"""

RELATIONSHIP_IDS = "relationship_ids"
"""关系 ID 列表。"""

TEXT_UNIT_IDS = "text_unit_ids"
"""文本单元 ID 列表。"""

COVARIATE_IDS = "covariate_ids"
"""协变量 ID 字典。"""

DOCUMENT_IDS = "document_ids"
"""文档 ID 列表。"""

# ============================================================
# 时间和大小字段
# ============================================================

PERIOD = "period"
"""时间段。"""

SIZE = "size"
"""大小（通常是文本单元数量）。"""

# ============================================================
# 文本单元字段
# ============================================================

ENTITY_DEGREE = "entity_degree"
"""实体度数。"""

ALL_DETAILS = "all_details"
"""所有详细信息。"""

TEXT = "text"
"""文本内容。"""

N_TOKENS = "n_tokens"
"""Token 数量。"""

# ============================================================
# 元数据字段
# ============================================================

CREATION_DATE = "creation_date"
"""创建日期。"""

METADATA = "metadata"
"""元数据。"""

# ============================================================
# 最终输出列定义
# ============================================================

ENTITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TYPE,
    DESCRIPTION,
    TEXT_UNIT_IDS,
    NODE_FREQUENCY,
    NODE_DEGREE,
    NODE_X,
    NODE_Y,
]
"""实体表的最终输出列。"""

RELATIONSHIPS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    EDGE_SOURCE,
    EDGE_TARGET,
    DESCRIPTION,
    EDGE_WEIGHT,
    EDGE_DEGREE,
    TEXT_UNIT_IDS,
]
"""关系表的最终输出列。"""

COMMUNITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    TEXT_UNIT_IDS,
    PERIOD,
    SIZE,
]
"""社区表的最终输出列。"""

COMMUNITY_REPORTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    SUMMARY,
    FULL_CONTENT,
    RATING,
    EXPLANATION,
    FINDINGS,
    FULL_CONTENT_JSON,
    PERIOD,
    SIZE,
]
"""社区报告表的最终输出列。"""

COVARIATES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    "covariate_type",
    TYPE,
    DESCRIPTION,
    "subject_id",
    "object_id",
    "status",
    "start_date",
    "end_date",
    "source_text",
    "text_unit_id",
]
"""协变量表的最终输出列。"""

TEXT_UNITS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TEXT,
    N_TOKENS,
    DOCUMENT_IDS,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    COVARIATE_IDS,
]
"""文本单元表的最终输出列。"""

DOCUMENTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TEXT,
    TEXT_UNIT_IDS,
    CREATION_DATE,
    METADATA,
]
"""文档表的最终输出列。"""

