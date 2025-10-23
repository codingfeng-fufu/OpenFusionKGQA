"""社区报告数据模型。"""

from dataclasses import dataclass
from typing import Any

from graphrag_v2.data_model.named import Named


@dataclass
class CommunityReport(Named):
    """社区报告数据模型。
    
    表示由 LLM 生成的社区摘要报告。
    每个社区都有一个对应的报告，描述该社区的主要内容和发现。
    
    Attributes:
        id: 唯一标识符
        short_id: 人类可读的短 ID（可选）
        title: 报告标题
        community_id: 关联的社区 ID
        summary: 报告摘要（默认为空字符串）
        full_content: 报告完整内容（默认为空字符串）
        rank: 报告的重要性排名，数值越大越重要（可选，默认为 1.0）
        full_content_embedding: 完整内容的向量嵌入（可选）
        attributes: 额外的属性字典（可选）
        size: 报告大小（文本单元数量，可选）
        period: 报告时间段（可选）
    """

    community_id: str = ""
    """关联的社区 ID（默认为空字符串）。"""

    summary: str = ""
    """报告摘要（默认为空字符串）。"""

    full_content: str = ""
    """报告完整内容（默认为空字符串）。"""

    rank: float | None = 1.0
    """报告的重要性排名，数值越大越重要（可选，默认为 1.0）。"""

    full_content_embedding: list[float] | None = None
    """完整内容的向量嵌入（可选）。"""

    attributes: dict[str, Any] | None = None
    """额外的属性字典（可选）。"""

    size: int | None = None
    """报告大小（文本单元数量，可选）。"""

    period: str | None = None
    """报告时间段（可选）。"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        community_id_key: str = "community",
        short_id_key: str = "human_readable_id",
        summary_key: str = "summary",
        full_content_key: str = "full_content",
        rank_key: str = "rank",
        attributes_key: str = "attributes",
        size_key: str = "size",
        period_key: str = "period",
    ) -> "CommunityReport":
        """从字典创建社区报告对象。
        
        Args:
            d: 包含社区报告数据的字典
            id_key: ID 字段的键名
            title_key: 标题字段的键名
            community_id_key: 社区 ID 字段的键名
            short_id_key: 短 ID 字段的键名
            summary_key: 摘要字段的键名
            full_content_key: 完整内容字段的键名
            rank_key: 排名字段的键名
            attributes_key: 属性字段的键名
            size_key: 大小字段的键名
            period_key: 时间段字段的键名
            
        Returns:
            CommunityReport: 社区报告对象
        """
        return CommunityReport(
            id=d[id_key],
            title=d[title_key],
            community_id=d[community_id_key],
            short_id=d.get(short_id_key),
            summary=d[summary_key],
            full_content=d[full_content_key],
            rank=d[rank_key],
            attributes=d.get(attributes_key),
            size=d.get(size_key),
            period=d.get(period_key),
        )

