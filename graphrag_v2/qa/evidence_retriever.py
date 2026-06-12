"""Text evidence retrieval for QA."""

from __future__ import annotations

from typing import Any

from graphrag_v2.qa.models import TextEvidence


class EvidenceRetriever:
    """Retrieve source text units by evidence chunk id."""

    def retrieve(
        self,
        chunk_ids: list[str],
        text_units: list[dict[str, Any]],
    ) -> list[TextEvidence]:
        if not chunk_ids:
            return []

        by_chunk_id = {
            str(text_unit.get("chunk_id", "")): text_unit for text_unit in text_units
        }
        seen: set[str] = set()
        evidence: list[TextEvidence] = []
        for chunk_id in chunk_ids:
            if chunk_id in seen:
                continue
            text_unit = by_chunk_id.get(chunk_id)
            if not text_unit:
                continue
            evidence.append(
                TextEvidence(
                    chunk_id=chunk_id,
                    doc_id=str(text_unit.get("doc_id", "")),
                    source_path=str(text_unit.get("source_path", "")),
                    chunk_index=int(text_unit.get("chunk_index") or 0),
                    text=str(text_unit.get("text", "")),
                    score=1.0,
                )
            )
            seen.add(chunk_id)
        evidence.sort(key=lambda item: (item.chunk_index, item.chunk_id))
        return evidence
