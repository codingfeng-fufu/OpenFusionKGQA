"""Community report generation."""

from __future__ import annotations

from dataclasses import replace

from graphrag_v2.community.models import Community, CommunityReport, GraphProjection


class MockCommunityReporter:
    """Rule-based community reporter for stable demos and tests."""

    def generate(
        self,
        communities: list[Community],
        projection: GraphProjection,
    ) -> tuple[list[Community], list[CommunityReport]]:
        entities_by_id = {entity["id"]: entity for entity in projection.entities}
        relationships_by_id = {
            relationship["id"]: relationship for relationship in projection.relationships
        }
        updated_communities: list[Community] = []
        reports: list[CommunityReport] = []
        for community in communities:
            community_entities = [
                entities_by_id[entity_id]
                for entity_id in community.entity_ids
                if entity_id in entities_by_id
            ]
            community_relationships = [
                relationships_by_id[relationship_id]
                for relationship_id in community.relationship_ids
                if relationship_id in relationships_by_id
            ]
            key_entities = [entity.get("name", entity["id"]) for entity in community_entities]
            key_relationships = [relationship["id"] for relationship in community_relationships]
            evidence_chunk_ids = sorted(
                set(community.text_unit_ids)
                | {
                    chunk_id
                    for relationship in community_relationships
                    for chunk_id in relationship.get("evidence_chunk_ids", [])
                }
            )
            summary = (
                f"This community contains {len(community_entities)} entities and "
                f"{len(community_relationships)} relationships."
            )
            if key_entities:
                summary += f" Key entities: {', '.join(key_entities[:3])}."
            findings = _findings(key_entities, community_relationships)
            full_content = _full_content(
                community=community,
                entities=community_entities,
                relationships=community_relationships,
                findings=findings,
            )
            updated_communities.append(replace(community, summary=summary))
            reports.append(
                CommunityReport(
                    id=f"report_{community.id}",
                    community_id=community.id,
                    title=community.title,
                    summary=summary,
                    full_content=full_content,
                    findings=findings,
                    key_entities=key_entities,
                    key_relationships=key_relationships,
                    evidence_chunk_ids=evidence_chunk_ids,
                    rank=community.rank,
                    metadata={"reporter": "mock"},
                )
            )
        return updated_communities, reports


def _findings(
    key_entities: list[str],
    relationships: list[dict],
) -> list[str]:
    findings: list[str] = []
    if key_entities:
        findings.append(f"Primary entities include {', '.join(key_entities[:3])}.")
    if relationships:
        top_relationship = sorted(
            relationships,
            key=lambda relationship: float(relationship.get("confidence") or 0.0),
            reverse=True,
        )[0]
        findings.append(
            "Strongest relationship: "
            f"{top_relationship.get('source_name')} "
            f"{top_relationship.get('relation')} "
            f"{top_relationship.get('target_name')}."
        )
    if not findings:
        findings.append("No relationship evidence was available for this community.")
    return findings


def _full_content(
    community: Community,
    entities: list[dict],
    relationships: list[dict],
    findings: list[str],
) -> str:
    lines = [
        f"# {community.title}",
        "",
        f"Size: {community.size}",
        f"Rank: {community.rank:.4f}",
        "",
        "## Entities",
    ]
    for entity in entities:
        lines.append(
            f"- {entity.get('name', entity['id'])} "
            f"({entity.get('type', 'Entity')}): {entity.get('description', '')}"
        )
    lines.extend(["", "## Relationships"])
    for relationship in relationships:
        lines.append(
            f"- {relationship.get('source_name')} -> "
            f"{relationship.get('target_name')}: "
            f"{relationship.get('description', relationship.get('relation', ''))}"
        )
    lines.extend(["", "## Findings"])
    for finding in findings:
        lines.append(f"- {finding}")
    return "\n".join(lines)
