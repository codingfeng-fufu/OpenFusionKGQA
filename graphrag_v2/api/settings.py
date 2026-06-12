"""Runtime settings for the production API foundation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AnswererName = Literal["mock", "llm"]
DEFAULT_MAX_QUESTION_CHARS = 2048


@dataclass(frozen=True)
class ApiRuntimeSettings:
    """Resolved process settings for the API runtime."""

    index_path: Path
    config_path: Path | None = None
    answerer: AnswererName = "mock"
    strict_neo4j: bool = False
    auth_token: str | None = None
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS

    @classmethod
    def from_env(cls) -> "ApiRuntimeSettings":
        index_value = (os.getenv("KGQA_API_INDEX_PATH") or "").strip()
        index_path = Path(index_value) if index_value else Path("artifacts/demo")
        config_value = (os.getenv("KGQA_API_CONFIG") or "").strip()
        answerer = _answerer_name(os.getenv("KGQA_API_ANSWERER") or "mock")
        auth_token = (os.getenv("KGQA_API_AUTH_TOKEN") or "").strip() or None
        return cls(
            index_path=index_path,
            config_path=Path(config_value) if config_value else None,
            answerer=answerer,
            strict_neo4j=_truthy(os.getenv("KGQA_API_STRICT_NEO4J")),
            auth_token=auth_token,
            max_question_chars=_positive_int(
                os.getenv("KGQA_API_MAX_QUESTION_CHARS"),
                default=DEFAULT_MAX_QUESTION_CHARS,
                name="KGQA_API_MAX_QUESTION_CHARS",
            ),
        )


def _answerer_name(value: str) -> AnswererName:
    normalized = value.strip().lower()
    if normalized not in {"mock", "llm"}:
        raise ValueError("KGQA_API_ANSWERER must be 'mock' or 'llm'.")
    return normalized  # type: ignore[return-value]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: str | None, *, default: int, name: str) -> int:
    normalized = str(value or "").strip()
    if not normalized:
        return default
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed
