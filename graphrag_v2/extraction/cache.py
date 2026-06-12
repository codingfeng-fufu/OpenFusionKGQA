"""Extraction cache for LLM candidate extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExtractionCache:
    """Small JSON-backed cache keyed by a stable extraction fingerprint."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory: dict[str, Any] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        if key in self._memory:
            return self._memory[key]
        if self.cache_dir is None:
            return None
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return None
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        self._memory[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        self._memory[key] = value
        if self.cache_dir is None:
            return
        cache_path = self._cache_path(key)
        try:
            cache_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"  # type: ignore[operator]
