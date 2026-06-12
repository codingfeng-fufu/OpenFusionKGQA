"""Index identity helpers for local artifacts and graph stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_index_id(output_path: str | Path) -> str:
    """Return a stable index id for an artifact output path."""
    resolved = str(Path(output_path).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"kgqa_{digest}"


def read_index_metadata(index_path: str | Path) -> dict[str, Any]:
    """Read index metadata if present."""
    metadata_path = Path(index_path) / "index_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_index_id(index_path: str | Path) -> str:
    """Resolve the persisted index id, falling back to the deterministic default."""
    metadata = read_index_metadata(index_path)
    index_id = metadata.get("index_id")
    if index_id:
        return str(index_id)
    return compute_index_id(index_path)
