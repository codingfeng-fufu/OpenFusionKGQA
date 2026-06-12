#!/usr/bin/env python
"""Evaluate extracted graph artifacts against a small expected graph."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted entities and relationships for a KGQA index.",
    )
    parser.add_argument("--index", required=True, help="Index/artifact directory.")
    parser.add_argument("--expected", required=True, help="Expected graph JSON path.")
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    expected_path = Path(args.expected)
    result = evaluate(index_path=index_path, expected_path=expected_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def evaluate(index_path: Path, expected_path: Path) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    entities = _read_records(index_path / "entities.parquet")
    relationships = _read_records(index_path / "relationships.parquet")

    actual_entities = {_name_key(str(entity.get("name", ""))) for entity in entities}
    actual_relationships = [
        (
            _name_key(str(row.get("source_name") or row.get("source") or "")),
            _relation_key(str(row.get("relation", ""))),
            _name_key(str(row.get("target_name") or row.get("target") or "")),
        )
        for row in relationships
    ]

    missing_entities = [
        _display_name(spec)
        for spec in expected.get("required_entities", [])
        if not _matches_entity(spec, actual_entities)
    ]
    missing_relationships = [
        _display_relationship(spec)
        for spec in expected.get("required_relationships", [])
        if not _matches_relationship(spec, actual_relationships)
    ]
    required_entities = len(expected.get("required_entities", []) or [])
    required_relationships = len(expected.get("required_relationships", []) or [])
    matched_entities = required_entities - len(missing_entities)
    matched_relationships = required_relationships - len(missing_relationships)

    return {
        "passed": not missing_entities and not missing_relationships,
        "num_entities": len(entities),
        "num_relationships": len(relationships),
        "required_entities": required_entities,
        "matched_entities": matched_entities,
        "entity_recall": _ratio(matched_entities, required_entities),
        "required_relationships": required_relationships,
        "matched_relationships": matched_relationships,
        "relationship_recall": _ratio(
            matched_relationships,
            required_relationships,
        ),
        "missing_entities": missing_entities,
        "missing_relationships": missing_relationships,
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict(orient="records")


def _matches_entity(spec: Any, actual_entities: set[str]) -> bool:
    return bool(_alias_keys(spec, _name_key) & actual_entities)


def _matches_relationship(
    spec: dict[str, Any],
    actual_relationships: list[tuple[str, str, str]],
) -> bool:
    sources = _alias_keys(spec.get("source"), _name_key)
    relations = _alias_keys(spec.get("relation"), _relation_key)
    targets = _alias_keys(spec.get("target"), _name_key)
    return any(
        source in sources and relation in relations and target in targets
        for source, relation, target in actual_relationships
    )


def _alias_keys(spec: Any, normalizer) -> set[str]:
    aliases = _aliases(spec)
    return {normalizer(alias) for alias in aliases if normalizer(alias)}


def _aliases(spec: Any) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        aliases: list[str] = []
        for item in spec:
            aliases.extend(_aliases(item))
        return aliases
    if isinstance(spec, dict):
        aliases = []
        for key in ("name", "value"):
            if spec.get(key):
                aliases.append(str(spec[key]))
        for key in ("aliases", "any_of"):
            aliases.extend(_aliases(spec.get(key)))
        return aliases
    return [str(spec)]


def _display_name(spec: Any) -> str:
    aliases = _aliases(spec)
    return aliases[0] if aliases else str(spec)


def _display_relationship(spec: dict[str, Any]) -> dict[str, str]:
    return {
        "source": _display_name(spec.get("source")),
        "relation": _display_name(spec.get("relation")),
        "target": _display_name(spec.get("target")),
    }


def _name_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _relation_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    sys.exit(main())
