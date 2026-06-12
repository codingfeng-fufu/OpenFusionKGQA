"""Question router for local vs global QA."""

from __future__ import annotations

from graphrag_v2.qa.models import LinkedEntity, RoutingDecision

GLOBAL_KEYWORDS = (
    "总结",
    "概括",
    "主题",
    "整体",
    "比较",
    "区别",
    "有哪些",
    "全局",
    "社区",
    "报告",
    "主要内容",
)

LOCAL_KEYWORDS = (
    "是什么",
    "是谁",
    "关系",
    "如何",
    "怎么",
    "为什么",
    "做了什么",
    "开发了",
    "使用了",
)


class QueryRouter:
    """Route questions to local or global retrieval."""

    def route(
        self,
        question: str,
        linked_entities: list[LinkedEntity] | None = None,
        community_report_count: int = 0,
    ) -> RoutingDecision:
        normalized = question.strip()
        if not normalized:
            return RoutingDecision("local", "Empty questions default to local routing.")

        for keyword in GLOBAL_KEYWORDS:
            if keyword in normalized:
                return RoutingDecision("global", f"Detected global cue: {keyword}")

        if linked_entities:
            names = ", ".join(entity.name for entity in linked_entities[:3])
            return RoutingDecision("local", f"Linked entities found: {names}")

        for keyword in LOCAL_KEYWORDS:
            if keyword in normalized:
                return RoutingDecision("local", f"Detected local cue: {keyword}")

        if community_report_count > 0:
            return RoutingDecision(
                "global",
                "No entity match found and community reports are available.",
            )

        return RoutingDecision("local", "Defaulted to local retrieval.")
