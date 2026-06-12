"""Run observability artifact helpers."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


EVENT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
RUN_EVENTS_FILENAME = "run_events.jsonl"
RUN_SUMMARY_FILENAME = "run_summary.json"
REDACTION_TOKEN = "***"
SENSITIVE_ENV_NAMES = (
    "NEO4J_PASSWORD",
    "ZHIPUAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KGQA_REAL_LLM_API_KEY",
    "GRAPHRAG_API_KEY",
    "GRAPHRAG_EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_LLM_API_KEY",
    "KGQA_API_AUTH_TOKEN",
)


@dataclass
class RunObserver:
    output_path: Path
    run_id: str
    index_id: str
    event_count: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    failed_stage: str | None = None
    _stage_starts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        output_path: str | Path,
        metadata: dict[str, Any],
    ) -> "RunObserver":
        run_id = metadata.get("run_id")
        if not run_id:
            raise ValueError("index metadata is missing required run_id")
        index_id = metadata.get("index_id")
        if not index_id:
            raise ValueError("index metadata is missing required index_id")
        if metadata.get("run_event_count") is not None:
            event_count = int(metadata["run_event_count"])
        else:
            event_count = count_run_events(output_path)
        return cls(
            output_path=Path(output_path),
            run_id=str(run_id),
            index_id=str(index_id),
            event_count=event_count,
            stage_timings=dict(metadata.get("run_stage_timings") or {}),
            failed_stage=metadata.get("run_failed_stage"),
        )

    def run_start(self, mode: str, input_path: str | Path | None) -> dict[str, Any]:
        return self._append(
            stage="run",
            event="run_start",
            status="running",
            counts={
                "mode": mode,
                "input_path": (
                    str(Path(input_path).resolve())
                    if input_path is not None
                    else None
                ),
            },
        )

    def run_end(self) -> dict[str, Any]:
        return self._append(
            stage="run",
            event="run_end",
            status="succeeded",
            elapsed_seconds=_sum_elapsed(self.stage_timings),
        )

    def run_failed(self, exc: Exception) -> dict[str, Any]:
        return self._append(
            stage="run",
            event="run_failed",
            status="failed",
            elapsed_seconds=_sum_elapsed(self.stage_timings),
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )

    def stage_start(
        self,
        stage: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self._stage_starts[stage] = time.perf_counter()
        return self._append(
            stage=stage,
            event="stage_start",
            status="running",
            provider=provider,
            model=model,
        )

    def stage_end(
        self,
        stage: str,
        counts: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        elapsed_seconds = self._stage_elapsed(stage)
        return self._append(
            stage=stage,
            event="stage_end",
            status="succeeded",
            elapsed_seconds=elapsed_seconds,
            counts=counts,
            provider=provider,
            model=model,
        )

    def stage_failed(
        self,
        stage: str,
        exc: Exception,
        counts: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.failed_stage = stage
        elapsed_seconds = self._stage_elapsed(stage)
        return self._append(
            stage=stage,
            event="stage_failed",
            status="failed",
            elapsed_seconds=elapsed_seconds,
            counts=counts,
            provider=provider,
            model=model,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )

    def _stage_elapsed(self, stage: str) -> float | None:
        started_at = self._stage_starts.pop(stage, None)
        if started_at is None:
            return None
        elapsed_seconds = round(time.perf_counter() - started_at, 6)
        self.stage_timings[stage] = round(
            float(self.stage_timings.get(stage, 0.0) or 0.0) + elapsed_seconds,
            6,
        )
        return elapsed_seconds

    def _append(
        self,
        *,
        stage: str | None,
        event: str,
        status: str,
        elapsed_seconds: float | None = None,
        counts: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        record = append_run_event(
            self.output_path,
            run_id=self.run_id,
            index_id=self.index_id,
            stage=stage,
            event=event,
            status=status,
            elapsed_seconds=elapsed_seconds,
            counts=counts,
            provider=provider,
            model=model,
            error_type=error_type,
            error_message=error_message,
        )
        self.event_count += 1
        return record


def new_run_id(
    *,
    now: datetime | None = None,
    token: str | None = None,
) -> str:
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.astimezone(UTC)
    suffix = (token or uuid.uuid4().hex)[:8]
    return f"run_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{suffix}"


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        redacted = value
        for env_name in SENSITIVE_ENV_NAMES:
            secret = os.environ.get(env_name)
            if secret:
                redacted = redacted.replace(secret, REDACTION_TOKEN)
        return redacted
    if isinstance(value, dict):
        return {
            redact_secrets(key): redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def run_events_path(output_path: str | Path) -> Path:
    return Path(output_path) / RUN_EVENTS_FILENAME


def run_summary_path(output_path: str | Path) -> Path:
    return Path(output_path) / RUN_SUMMARY_FILENAME


def reset_run_observability_artifacts(output_path: str | Path) -> None:
    for artifact_path in (run_events_path(output_path), run_summary_path(output_path)):
        if artifact_path.exists():
            artifact_path.unlink()


def append_run_event(
    output_path: str | Path,
    *,
    run_id: str,
    index_id: str,
    stage: str | None,
    event: str,
    status: str,
    elapsed_seconds: float | None = None,
    counts: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    record = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "run_id": run_id,
        "index_id": index_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "stage": stage,
        "event": event,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "counts": counts or {},
        "provider": provider,
        "model": model,
        "error_type": error_type,
        "error_message": error_message,
    }
    record = redact_secrets(record)

    events_path = run_events_path(output_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    return record


def load_run_events(output_path: str | Path) -> list[dict[str, Any]]:
    events_path = run_events_path(output_path)
    if not events_path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{events_path}: malformed JSON in {RUN_EVENTS_FILENAME} "
                f"line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{events_path}: expected JSON object/dict in "
                f"{RUN_EVENTS_FILENAME} line {line_number}"
            )
        records.append(record)
    return records


def count_run_events(output_path: str | Path) -> int:
    events_path = run_events_path(output_path)
    if not events_path.exists():
        return 0
    with events_path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_index_metadata(output_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(output_path) / "index_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_run_summary(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    timings = metadata.get("run_stage_timings") or {}
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": metadata.get("run_id"),
        "index_id": metadata.get("index_id"),
        "status": metadata.get("run_status"),
        "mode": metadata.get("run_mode") or metadata.get("mode"),
        "input_path": metadata.get("input_path"),
        "output_path": metadata.get("output_path"),
        "started_at": metadata.get("run_started_at"),
        "finished_at": metadata.get("run_finished_at"),
        "elapsed_seconds": metadata.get("run_elapsed_seconds"),
        "failed_stage": _failed_stage(metadata, events),
        "document": _stage_summary(
            metadata,
            timings,
            "document",
            (
                "num_documents",
                "num_text_units",
                "num_input_files",
                "num_included_files",
                "num_ignored_files",
                "num_rejected_files",
                "num_empty_documents",
                "chunk_size",
                "chunk_overlap",
            ),
        ),
        "extraction": _stage_summary(
            metadata,
            timings,
            "extraction",
            (
                "extractor",
                "num_candidate_entities",
                "num_candidate_relationships",
                "num_candidate_triples",
                "llm_provider",
                "llm_model_id",
                "llm_model_name",
                "llm_total_calls",
                "llm_total_tokens",
                "llm_estimated_cost",
                "extraction_prompt_version",
                "extraction_parse_failures",
                "extraction_repair_attempts",
                "extraction_max_gleanings",
                "extraction_gleaning_attempts",
                "extraction_gleaning_failures",
                "extraction_gleaned_entities",
                "extraction_gleaned_relationships",
                "extraction_salvaged_entities",
                "extraction_salvaged_relationships",
                "extraction_dropped_entities",
                "extraction_dropped_relationships",
                "extraction_failed_chunks",
                "extraction_failed_chunk_ids",
                "extraction_budget_exceeded",
                "extraction_cache_enabled",
                "extraction_cache_hits",
                "extraction_cache_misses",
            ),
        ),
        "fusion": _stage_summary(
            metadata,
            timings,
            "fusion",
            (
                "num_entities",
                "num_relationships",
                "num_rejected_triples",
                "fusion_min_confidence",
                "fusion_parameters_version",
                "fusion_relation_schema_mode",
                "fusion_relation_schema_version",
                "fusion_scoring_version",
                "fusion_scoring_weights",
                "fusion_num_accepted_triples",
                "fusion_entity_override_count",
                "fusion_relation_override_count",
            ),
        ),
        "graph_store": _graph_store_summary(metadata, timings),
        "community": _stage_summary(
            metadata,
            timings,
            "community",
            (
                "num_communities",
                "num_community_reports",
                "community_algorithm",
                "community_reporter",
            ),
        ),
        "errors": _summary_errors(metadata, events),
    }
    return redact_secrets(summary)


def write_run_summary(output_path: str | Path) -> dict[str, Any]:
    summary = build_run_summary(
        read_index_metadata(output_path),
        load_run_events(output_path),
    )
    summary_path = run_summary_path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def load_run_summary(output_path: str | Path) -> dict[str, Any]:
    return redact_secrets(
        json.loads(run_summary_path(output_path).read_text(encoding="utf-8"))
    )


def inspect_run_summary(output_path: str | Path) -> tuple[dict[str, Any], str]:
    if run_summary_path(output_path).exists():
        return load_run_summary(output_path), "present"
    return (
        build_run_summary(read_index_metadata(output_path), load_run_events(output_path)),
        "missing",
    )


def format_run_report(
    output_path: str | Path,
    summary: dict[str, Any],
    summary_status: str,
) -> str:
    summary = redact_secrets(summary)
    output_dir = Path(output_path)
    lines = [
        "Run:",
        f"  summary_status: {summary_status}",
        f"  run_id: {summary.get('run_id')}",
        f"  index_id: {summary.get('index_id')}",
        f"  status: {summary.get('status')}",
        f"  mode: {summary.get('mode')}",
        f"  elapsed_seconds: {summary.get('elapsed_seconds')}",
        f"  failed_stage: {summary.get('failed_stage')}",
        "Stages:",
    ]

    stage_fields = {
        "document": ("elapsed_seconds", "num_text_units"),
        "extraction": (
            "elapsed_seconds",
            "extractor",
            "num_candidate_entities",
        ),
        "fusion": ("elapsed_seconds", "num_entities", "num_relationships"),
        "graph_store": (
            "elapsed_seconds",
            "provider",
            "written",
            "health_status",
        ),
        "community": ("elapsed_seconds", "num_communities"),
    }
    for stage, fields in stage_fields.items():
        rendered = _format_stage_line(summary.get(stage) or {}, fields)
        if rendered:
            lines.append(f"  {stage}: {rendered}")

    lines.extend(
        [
            "Artifacts:",
            f"  metadata_path: {(output_dir / 'index_metadata.json').resolve()}",
            f"  run_summary_path: {run_summary_path(output_dir).resolve()}",
            f"  run_events_path: {run_events_path(output_dir).resolve()}",
        ]
    )

    errors = summary.get("errors") or []
    if errors:
        lines.append("Errors:")
        for error in errors:
            parts = [
                str(error.get("stage") or "run"),
                str(error.get("error_type") or "error"),
            ]
            if error.get("error_message"):
                parts.append(str(error["error_message"]))
            lines.append(f"  - {': '.join(parts)}")

    return "\n".join(lines)


def _stage_summary(
    metadata: dict[str, Any],
    timings: dict[str, Any],
    stage: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    summary = {"elapsed_seconds": timings.get(stage)}
    summary.update({field_name: metadata.get(field_name) for field_name in fields})
    return summary


def _graph_store_summary(
    metadata: dict[str, Any],
    timings: dict[str, Any],
) -> dict[str, Any]:
    field_map = {
        "provider": "graph_store_provider",
        "written": "graph_store_written",
        "error": "graph_store_error",
        "database": "graph_store_database",
        "index_id": "graph_store_index_id",
        "num_text_units": "graph_store_num_text_units",
        "num_entities": "graph_store_num_entities",
        "num_relationships": "graph_store_num_relationships",
        "health_status": "graph_store_health_status",
        "schema_ready": "graph_store_schema_ready",
        "schema_constraint_count": "graph_store_schema_constraint_count",
        "schema_index_count": "graph_store_schema_index_count",
        "schema_version": "graph_store_schema_version",
        "missing_schema_constraints": "graph_store_missing_schema_constraints",
        "missing_schema_indexes": "graph_store_missing_schema_indexes",
        "write_strategy": "graph_store_write_strategy",
        "staging_index_id": "graph_store_staging_index_id",
    }
    summary = {"elapsed_seconds": timings.get("graph_store")}
    summary.update(
        {
            summary_key: metadata.get(metadata_key)
            for summary_key, metadata_key in field_map.items()
        }
    )
    return summary


def _summary_errors(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    seen_errors = set()
    seen_stage_error_messages = set()

    def add_error(error: dict[str, Any]) -> None:
        key = (
            error.get("stage"),
            error.get("error_type"),
            error.get("error_message"),
        )
        message_key = (error.get("error_type"), error.get("error_message"))
        if (
            error.get("stage") == "run"
            and error.get("event") == "run_failed"
            and message_key in seen_stage_error_messages
        ):
            return
        if key in seen_errors:
            return
        seen_errors.add(key)
        if error.get("stage") != "run":
            seen_stage_error_messages.add(message_key)
        errors.append(error)

    if metadata.get("run_error_type") or metadata.get("run_error_message"):
        add_error(
            {
                "stage": metadata.get("run_failed_stage"),
                "event": "run_failed",
                "status": metadata.get("run_status"),
                "timestamp": metadata.get("run_finished_at"),
                "error_type": metadata.get("run_error_type"),
                "error_message": metadata.get("run_error_message"),
            }
        )
    if metadata.get("graph_store_error"):
        add_error(
            {
                "stage": "graph_store",
                "event": "graph_store_failed",
                "status": metadata.get("run_status"),
                "timestamp": metadata.get("run_finished_at"),
                "error_type": None,
                "error_message": metadata.get("graph_store_error"),
            }
        )
    for event in events:
        if event.get("error_type") or event.get("error_message"):
            add_error(
                {
                    "stage": event.get("stage"),
                    "event": event.get("event"),
                    "status": event.get("status"),
                    "timestamp": event.get("timestamp"),
                    "error_type": event.get("error_type"),
                    "error_message": event.get("error_message"),
                }
            )
    return errors


def _failed_stage(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> str | None:
    failed_stage = metadata.get("run_failed_stage")
    if failed_stage is not None:
        return failed_stage
    for event in events:
        if event.get("status") == "failed" and event.get("stage"):
            return event["stage"]
    return None


def _format_stage_line(
    stage_summary: dict[str, Any],
    fields: tuple[str, ...],
) -> str:
    parts = []
    for field_name in fields:
        value = stage_summary.get(field_name)
        if value is not None:
            parts.append(f"{field_name}={value}")
    return ", ".join(parts)


def _sum_elapsed(stage_timings: dict[str, float]) -> float | None:
    if not stage_timings:
        return None
    return round(sum(stage_timings.values()), 6)
