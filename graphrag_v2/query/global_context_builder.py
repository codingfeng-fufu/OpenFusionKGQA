"""Global Search 上下文构建器实现。"""

import logging
from typing import Any

import pandas as pd

from graphrag_v2.query.context_builder import ContextBuilderResult, GlobalContextBuilder

logger = logging.getLogger(__name__)


class CommunityContextBuilder(GlobalContextBuilder):
    """基于社区报告的上下文构建器。
    
    这个构建器：
    1. 加载所有社区报告
    2. 按排名排序
    3. 将报告分批（每批不超过 max_tokens）
    4. 返回批次列表供 Map-Reduce 使用
    """
    
    def __init__(
        self,
        community_reports: pd.DataFrame,
        max_tokens: int = 8000,
        batch_size: int = 5,
    ):
        """初始化上下文构建器。
        
        Args:
            community_reports: 社区报告 DataFrame
            max_tokens: 每批最大 tokens 数
            batch_size: 每批最大报告数
        """
        self.community_reports = community_reports
        self.max_tokens = max_tokens
        self.batch_size = batch_size
    
    async def build_context(
        self,
        query: str,
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """构建 Global Search 上下文。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            ContextBuilderResult: 上下文构建结果
        """
        logger.info(f"构建 Global Context，社区报告数: {len(self.community_reports)}")
        
        # 按排名排序（排名越高越重要）
        sorted_reports = self.community_reports.sort_values(
            by="rank",
            ascending=False,
        )
        
        # 分批
        batches = []
        current_batch = []
        current_tokens = 0
        
        for _, report in sorted_reports.iterrows():
            # 估计 tokens 数（简化版本：字符数 / 4）
            report_text = self._format_report(report)
            report_tokens = len(report_text) // 4
            
            # 检查是否需要开始新批次
            if (
                len(current_batch) >= self.batch_size
                or current_tokens + report_tokens > self.max_tokens
            ):
                if current_batch:
                    batches.append("\n\n".join(current_batch))
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(report_text)
            current_tokens += report_tokens
        
        # 添加最后一批
        if current_batch:
            batches.append("\n\n".join(current_batch))
        
        logger.info(f"生成了 {len(batches)} 个批次")
        
        return ContextBuilderResult(
            context_chunks=batches,
            context_records={"community_reports": self.community_reports},
            llm_calls=0,
            prompt_tokens=0,
            output_tokens=0,
        )
    
    def _format_report(self, report: pd.Series) -> str:
        """格式化社区报告。
        
        Args:
            report: 社区报告（Series）
            
        Returns:
            str: 格式化的报告文本
        """
        return f"""# 社区 {report['community_id']}

**标题**: {report['title']}

**摘要**: {report['summary']}

**排名**: {report['rank']:.3f}

**详细内容**:
{report['full_content']}

**关键发现**:
{report.get('findings', 'N/A')}
"""

