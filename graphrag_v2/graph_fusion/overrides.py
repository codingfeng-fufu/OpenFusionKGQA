"""Manual graph-fusion overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionOverrides:
    """Optional manual aliases imported before graph fusion."""

    entity_aliases: dict[str, str] = field(default_factory=dict)
    relation_aliases: dict[str, str] = field(default_factory=dict)


def load_fusion_overrides(path: str | Path) -> FusionOverrides:
    """Load manual entity/relation overrides from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fusion overrides must be a JSON object.")
    return FusionOverrides(
        entity_aliases=_string_mapping(payload.get("entity_aliases", {})),
        relation_aliases=_string_mapping(payload.get("relation_aliases", {})),
    )


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Fusion override sections must be objects.")
    result: dict[str, str] = {}
    for key, mapped in value.items():
        if not str(key).strip() or not str(mapped).strip():
            raise ValueError("Fusion override keys and values cannot be empty.")
        result[str(key)] = str(mapped)
    return result
