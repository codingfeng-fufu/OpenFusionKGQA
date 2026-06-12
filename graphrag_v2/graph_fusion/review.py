"""Human review export for graph fusion artifacts."""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import pandas as pd


def export_review_queue(
    index_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Export accepted/rejected fusion decisions to JSONL without Neo4j."""
    index_dir = Path(index_path)
    output = Path(output_path)
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Review queue output_path must end with .jsonl")
    if not index_dir.exists():
        raise FileNotFoundError(f"Index path does not exist: {index_dir}")

    records: list[dict[str, Any]] = []
    records.extend(_accepted_relationship_records(index_dir))
    records.extend(_rejected_triple_records(index_dir))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    accepted_count = sum(1 for record in records if record["status"] == "accepted")
    rejected_count = sum(1 for record in records if record["status"] == "rejected")
    return {
        "output_path": str(output.resolve()),
        "num_records": len(records),
        "num_accepted": accepted_count,
        "num_rejected": rejected_count,
    }


def _accepted_relationship_records(index_dir: Path) -> list[dict[str, Any]]:
    path = index_dir / "relationships.parquet"
    if not path.exists():
        return []
    rows = pd.read_parquet(path).to_dict(orient="records")
    records: list[dict[str, Any]] = []
    for row in rows:
        metadata = _dict_value(row.get("metadata"))
        records.append(
            {
                "type": "relationship",
                "status": "accepted",
                "id": row.get("id"),
                "source_name": row.get("source_name"),
                "target_name": row.get("target_name"),
                "relation": row.get("relation"),
                "description": row.get("description"),
                "confidence": row.get("confidence"),
                "evidence_chunk_ids": _list_value(row.get("evidence_chunk_ids")),
                "source_triple_ids": _list_value(metadata.get("source_triple_ids")),
                "source_relationship_ids": _list_value(
                    metadata.get("source_relationship_ids")
                ),
                "source_entity_ids": _list_value(metadata.get("source_entity_ids")),
                "rejection_reasons": [],
                "metadata": metadata,
            }
        )
    return records


def _rejected_triple_records(index_dir: Path) -> list[dict[str, Any]]:
    path = index_dir / "rejected_triples.parquet"
    if not path.exists():
        return []
    rows = pd.read_parquet(path).to_dict(orient="records")
    records: list[dict[str, Any]] = []
    for row in rows:
        metadata = _dict_value(row.get("metadata"))
        records.append(
            {
                "type": "triple",
                "status": "rejected",
                "id": row.get("id"),
                "source_name": row.get("source_name"),
                "target_name": row.get("target_name"),
                "relation": row.get("canonical_relation")
                or row.get("relation_mention"),
                "description": row.get("description"),
                "confidence": row.get("triple_score"),
                "evidence_chunk_ids": _list_value(row.get("evidence_chunk_ids")),
                "source_triple_ids": [row.get("id")] if row.get("id") else [],
                "source_relationship_ids": [
                    metadata["relationship_id"]
                ]
                if metadata.get("relationship_id")
                else [],
                "source_entity_ids": [],
                "rejection_reasons": _list_value(
                    metadata.get("rejection_reasons")
                ),
                "metadata": metadata,
            }
        )
    return records


def _dict_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: _normalize_metadata_value(item) for key, item in value.items()}


def _normalize_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _normalize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _normalize_metadata_value(to_list())
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        return _normalize_metadata_value(parsed)
    return value


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        parsed = to_list()
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        if isinstance(parsed, list):
            return parsed
    return [value]
