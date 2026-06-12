"""Graph-grounded QA orchestration."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from graphrag_v2.config.models.graph_store_config import GraphStoreConfig
from graphrag_v2.qa.answerer import LLMAnswerer, MockAnswerer
from graphrag_v2.qa.community_retriever import CommunityRetriever
from graphrag_v2.qa.evidence_retriever import EvidenceRetriever
from graphrag_v2.qa.entity_linker import EntityLinker
from graphrag_v2.qa.graph_retriever import GraphRetriever
from graphrag_v2.qa.models import QAResult, RouteType
from graphrag_v2.qa.prompts import QA_ANSWER_PROMPT_VERSION
from graphrag_v2.qa.query_router import QueryRouter
from graphrag_v2.qa.sources import QADataSource, load_qa_data_source


class GraphGroundedQA:
    """Single-entry QA pipeline over graph artifacts."""

    def __init__(
        self,
        data_source: QADataSource,
        router: QueryRouter | None = None,
        entity_linker: EntityLinker | None = None,
        graph_retriever: GraphRetriever | None = None,
        community_retriever: CommunityRetriever | None = None,
        evidence_retriever: EvidenceRetriever | None = None,
        answerer: MockAnswerer | LLMAnswerer | None = None,
    ):
        self.data_source = data_source
        self.router = router or QueryRouter()
        self.entity_linker = entity_linker or EntityLinker()
        self.graph_retriever = graph_retriever or GraphRetriever()
        self.community_retriever = community_retriever or CommunityRetriever()
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self.answerer = answerer or MockAnswerer()

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        prefer_neo4j: bool = True,
        allow_neo4j_fallback: bool = True,
        graph_store_config: GraphStoreConfig | None = None,
        answerer: MockAnswerer | LLMAnswerer | None = None,
    ) -> "GraphGroundedQA":
        data_source = load_qa_data_source(
            index_path,
            config=graph_store_config,
            prefer_neo4j=prefer_neo4j,
            allow_neo4j_fallback=allow_neo4j_fallback,
        )
        return cls(data_source=data_source, answerer=answerer)

    def ask(self, question: str) -> QAResult:
        entities = self.data_source.entities()
        relationships = self.data_source.relationships()
        communities = self.data_source.communities()
        reports = self.data_source.community_reports()
        text_units = self.data_source.text_units()

        linked_entities = self.entity_linker.link(question, entities)
        decision = self.router.route(
            question=question,
            linked_entities=linked_entities,
            community_report_count=len(reports),
        )

        graph_evidence = self.graph_retriever.retrieve(
            linked_entities,
            relationships,
            question=question,
        )
        community_evidence = []
        if decision.route == "global":
            community_evidence = self.community_retriever.retrieve(
                question=question,
                communities=communities,
                reports=reports,
            )
        if decision.route == "global" and not community_evidence and not graph_evidence.text_chunk_ids:
            fallback_chunk_ids = [
                str(item.get("chunk_id", ""))
                for item in text_units[: min(len(text_units), 5)]
                if item.get("chunk_id")
            ]
        else:
            fallback_chunk_ids = []

        chunk_ids = _unique(
            [
                *graph_evidence.text_chunk_ids,
                *[chunk_id for entity in linked_entities for chunk_id in entity.evidence_chunk_ids],
                *fallback_chunk_ids,
                *[
                    chunk_id
                    for community in community_evidence
                    for chunk_id in community.evidence_chunk_ids
                ],
            ]
        )
        text_evidence = self.evidence_retriever.retrieve(chunk_ids, text_units)

        refusal_reason = None
        if not text_evidence:
            answer = "证据不足，无法回答该问题。"
            confidence = 0.0
            refusal_reason = "no_source_evidence"
        else:
            answer = self.answerer.answer(
                question=question,
                route=decision.route,
                graph_evidence=graph_evidence,
                community_evidence=community_evidence,
                text_evidence=text_evidence,
            )
            confidence = _confidence(
                route=decision.route,
                graph_evidence=graph_evidence,
                community_evidence=community_evidence,
                text_evidence=text_evidence,
            )

        citations = _unique([item.chunk_id for item in text_evidence])
        used_entities = _unique([entity.id for entity in linked_entities])
        used_relationships = _unique([relationship.id for relationship in graph_evidence.relationships])
        used_communities = _unique([community.community_id for community in community_evidence])
        used_reports = _unique([community.report_id for community in community_evidence])
        result_metadata = {
            "routing_reason": decision.reason,
            "source_provider": self.data_source.provider,
            "answer_prompt_version": QA_ANSWER_PROMPT_VERSION,
            "query_trace": _query_trace(
                route=decision.route,
                routing_reason=decision.reason,
                linked_entities=linked_entities,
                graph_evidence=graph_evidence,
                community_evidence=community_evidence,
                text_evidence=text_evidence,
            ),
        }
        source_metadata = self.data_source.metadata()
        for key in ("qa_fallback_from_provider", "qa_fallback_reason"):
            if key in source_metadata:
                result_metadata[key] = source_metadata[key]

        return QAResult(
            question=question,
            route=decision.route,
            answer=answer,
            citations=citations,
            refusal_reason=refusal_reason,
            used_entities=used_entities,
            used_relationships=used_relationships,
            used_communities=used_communities,
            used_community_reports=used_reports,
            confidence=confidence,
            graph_evidence=graph_evidence,
            community_evidence=community_evidence,
            text_evidence=text_evidence,
            source_provider=self.data_source.provider,
            metadata=result_metadata,
        )


def format_qa_result(result: QAResult) -> str:
    lines = [
        f"Question: {result.question}",
        f"Route: {result.route}",
        "",
        "Answer:",
        result.answer,
        "",
        "Graph Evidence:",
    ]

    if result.graph_evidence.linked_entities:
        for entity in result.graph_evidence.linked_entities:
            lines.append(
                f"- {entity.name} ({entity.type or 'unknown'}) [score={entity.score:.2f}]"
            )
    else:
        lines.append("- none")

    if result.graph_evidence.relationships:
        for relationship in result.graph_evidence.relationships:
            lines.append(
                f"  * {relationship.source_name} {relationship.relation} {relationship.target_name}"
                f" [score={relationship.score:.2f}]"
            )
    else:
        lines.append("  * none")

    lines.extend(["", "Community Evidence:"])
    if result.community_evidence:
        for community in result.community_evidence:
            lines.append(
                f"- {community.title} [score={community.score:.2f}, report={community.report_id}]"
            )
            if community.summary:
                lines.append(f"  * {community.summary}")
    else:
        lines.append("- none")

    lines.extend(["", "Citations:"])
    if result.citations:
        for citation in result.citations:
            lines.append(f"- {citation}")
    else:
        lines.append("- none")

    lines.extend(["", f"Confidence: {result.confidence:.2f}"])
    if result.refusal_reason:
        lines.extend(["", f"Refusal Reason: {result.refusal_reason}"])
    return "\n".join(lines)


def format_qa_result_json(result: QAResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def _unique(values: list[str]) -> list[str]:
    ordered = OrderedDict.fromkeys(value for value in values if value)
    return list(ordered.keys())


def _confidence(
    route: RouteType,
    graph_evidence,
    community_evidence,
    text_evidence,
) -> float:
    score = 0.1
    score += min(len(graph_evidence.linked_entities), 3) * 0.12
    score += min(len(graph_evidence.relationships), 4) * 0.10
    score += min(len(community_evidence), 3) * 0.12
    score += min(len(text_evidence), 5) * 0.06
    if route == "global" and community_evidence:
        score += 0.08
    if route == "local" and graph_evidence.relationships:
        score += 0.08
    return round(min(score, 1.0), 2)


def _query_trace(
    *,
    route: RouteType,
    routing_reason: str,
    linked_entities,
    graph_evidence,
    community_evidence,
    text_evidence,
) -> dict:
    trace = {
        "route": route,
        "routing_reason": routing_reason,
        "linked_entities": [
            {
                "id": entity.id,
                "name": entity.name,
                "score": entity.score,
            }
            for entity in linked_entities
        ],
        "retrieved_relationships": [
            {
                "id": relationship.id,
                "source": relationship.source_name,
                "relation": relationship.relation,
                "target": relationship.target_name,
                "score": relationship.score,
                "hop": relationship.hop,
            }
            for relationship in graph_evidence.relationships
        ],
        "retrieved_communities": [
            {
                "community_id": community.community_id,
                "report_id": community.report_id,
                "title": community.title,
                "score": community.score,
            }
            for community in community_evidence
        ],
        "retrieved_text_chunks": [
            {
                "chunk_id": text.chunk_id,
                "source_path": text.source_path,
                "chunk_index": text.chunk_index,
            }
            for text in text_evidence
        ],
    }
    trace.update(graph_evidence.retrieval_metadata)
    return trace
