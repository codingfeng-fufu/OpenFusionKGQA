"""Validation helpers for extraction results."""

from __future__ import annotations

from graphrag_v2.extraction.models import ExtractionResult


def validate_extraction_result(result: ExtractionResult) -> list[str]:
    """Return validation errors for an extraction result."""
    errors: list[str] = []
    entity_names = {entity.name for entity in result.entities if entity.name.strip()}

    for entity in result.entities:
        if not entity.name.strip():
            errors.append(f"Entity {entity.id} has empty name")
        if not _valid_confidence(entity.confidence):
            errors.append(f"Entity {entity.id} has invalid confidence")
        if not entity.evidence_chunk_ids:
            errors.append(f"Entity {entity.id} has no evidence chunks")

    for relationship in result.relationships:
        if not relationship.source.strip() or not relationship.target.strip():
            errors.append(f"Relationship {relationship.id} has empty endpoint")
        if relationship.source not in entity_names:
            errors.append(f"Relationship {relationship.id} source not found")
        if relationship.target not in entity_names:
            errors.append(f"Relationship {relationship.id} target not found")
        if not _valid_confidence(relationship.confidence):
            errors.append(f"Relationship {relationship.id} has invalid confidence")
        if not relationship.evidence_chunk_ids:
            errors.append(f"Relationship {relationship.id} has no evidence chunks")

    for triple in result.triples:
        if triple.source_name not in entity_names:
            errors.append(f"Triple {triple.id} source not found")
        if triple.target_name not in entity_names:
            errors.append(f"Triple {triple.id} target not found")
        if not _valid_confidence(triple.extraction_confidence):
            errors.append(f"Triple {triple.id} has invalid extraction confidence")
        if not triple.evidence_chunk_ids:
            errors.append(f"Triple {triple.id} has no evidence chunks")

    return errors


def _valid_confidence(value: float) -> bool:
    return 0.0 <= value <= 1.0
