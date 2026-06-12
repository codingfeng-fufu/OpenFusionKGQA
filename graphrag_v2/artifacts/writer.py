"""Artifact writers for indexing outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from graphrag_v2.artifacts.run_observability import (
    count_run_events,
    new_run_id,
    redact_secrets,
    run_events_path,
    run_summary_path,
    write_run_summary,
)
from graphrag_v2.document.models import DocumentScanResult, SourceDocument, TextUnit
from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag_v2.community.models import Community, CommunityReport
from graphrag_v2.graph_fusion.models import FusionResult
from graphrag_v2.artifacts.contracts import (
    CANDIDATE_ENTITIES_COLUMNS,
    CANDIDATE_RELATIONSHIPS_COLUMNS,
    CANDIDATE_TRIPLES_COLUMNS,
    GENERATED_ARTIFACTS,
    METADATA_SCHEMA_VERSION,
    RELATIONSHIPS_COLUMNS,
    ENTITIES_COLUMNS,
    TEXT_UNITS_COLUMNS,
)
from graphrag_v2.artifacts.index_id import compute_index_id


STAGE_METADATA_KEYS = (
    "num_documents",
    "num_text_units",
    "num_input_files",
    "num_included_files",
    "num_ignored_files",
    "num_rejected_files",
    "num_empty_documents",
    "chunk_size",
    "chunk_overlap",
    "extractor",
    "num_candidate_entities",
    "num_candidate_relationships",
    "num_candidate_triples",
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
    "graph_store_provider",
    "graph_store_written",
    "graph_store_error",
    "graph_store_database",
    "graph_store_index_id",
    "graph_store_num_text_units",
    "graph_store_num_entities",
    "graph_store_num_relationships",
    "graph_store_health_status",
    "graph_store_schema_ready",
    "graph_store_schema_constraint_count",
    "graph_store_schema_index_count",
    "graph_store_schema_version",
    "graph_store_missing_schema_constraints",
    "graph_store_missing_schema_indexes",
    "graph_store_write_strategy",
    "graph_store_staging_index_id",
    "num_communities",
    "num_community_reports",
    "community_algorithm",
    "community_reporter",
)

STAGE_METADATA_PREFIXES = (
    "llm_",
    "extraction_",
)


def write_document_artifacts(
    output_path: str | Path,
    documents: list[SourceDocument],
    text_units: list[TextUnit],
    input_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    mode: str,
    document_scan: DocumentScanResult | None = None,
) -> dict:
    """Write text unit parquet and index metadata."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_units_df = pd.DataFrame(
        [asdict(text_unit) for text_unit in text_units],
        columns=list(TEXT_UNITS_COLUMNS),
    )
    text_units_df.to_parquet(output_dir / "text_units.parquet", index=False)
    if document_scan is not None:
        (output_dir / "document_scan.json").write_text(
            json.dumps(document_scan.to_manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metadata = _read_metadata(output_dir)
    if "created_at" not in metadata:
        metadata["created_at"] = datetime.now(UTC).isoformat()
    scan_metadata = _document_scan_metadata(document_scan, documents)
    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "index_id": compute_index_id(output_dir),
            "input_path": str(Path(input_path).resolve()),
            "output_path": str(output_dir.resolve()),
            "num_documents": len(documents),
            "num_text_units": len(text_units),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "mode": mode,
            **scan_metadata,
        }
    )
    _write_metadata(output_dir, metadata)

    return metadata


def start_run_metadata(
    output_path: str | Path,
    *,
    mode: str,
    input_path: str | Path | None = None,
) -> dict:
    """Create or update run-level metadata at the start of an indexing run."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_artifacts(output_dir)
    metadata = _read_metadata(output_dir)
    _clear_stage_metadata(metadata)
    started_at = _utc_now()
    run_id = new_run_id()
    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "index_id": compute_index_id(output_dir),
            "output_path": str(output_dir.resolve()),
            "run_id": run_id,
            "run_started_at": started_at.isoformat(),
            "run_finished_at": None,
            "run_elapsed_seconds": None,
            "run_status": "running",
            "run_mode": mode,
            "mode": mode,
            "run_failed_stage": None,
            "run_stage_timings": {},
            "run_event_count": 0,
            "run_summary_path": str(run_summary_path(output_dir).resolve()),
            "run_events_path": str(run_events_path(output_dir).resolve()),
        }
    )
    if input_path is not None:
        metadata["input_path"] = str(Path(input_path).resolve())
    _clear_run_error(metadata)
    _write_metadata(output_dir, metadata)
    return metadata


def finish_run_metadata(
    output_path: str | Path,
    *,
    status: str = "succeeded",
    stage_timings: dict[str, float] | None = None,
) -> dict:
    """Mark run-level metadata as finished."""
    output_dir = Path(output_path)
    metadata = _read_metadata(output_dir)
    finished_at = _utc_now()
    started_at = _parse_datetime(metadata.get("run_started_at"))
    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "run_finished_at": finished_at.isoformat(),
            "run_elapsed_seconds": _elapsed_seconds(started_at, finished_at),
            "run_status": status,
            "run_failed_stage": (
                None if status == "succeeded" else metadata.get("run_failed_stage")
            ),
            "run_stage_timings": (
                stage_timings
                if stage_timings is not None
                else metadata.get("run_stage_timings", {})
            ),
            "run_event_count": count_run_events(output_dir),
            "run_summary_path": str(run_summary_path(output_dir).resolve()),
            "run_events_path": str(run_events_path(output_dir).resolve()),
        }
    )
    if status == "succeeded":
        _clear_run_error(metadata)
    _write_metadata(output_dir, metadata)
    write_run_summary(output_dir)
    return metadata


def fail_run_metadata(
    output_path: str | Path,
    error: Exception | None = None,
    *,
    exc: Exception | None = None,
    status: str = "failed",
    failed_stage: str | None = None,
    stage_timings: dict[str, float] | None = None,
) -> dict:
    """Best-effort failure metadata writer for indexing runs."""
    error = error or exc
    if error is None:
        raise TypeError("fail_run_metadata() missing required error")
    output_dir = Path(output_path)
    metadata = _read_metadata(output_dir)
    finished_at = _utc_now()
    started_at = _parse_datetime(metadata.get("run_started_at"))
    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "run_finished_at": finished_at.isoformat(),
            "run_elapsed_seconds": _elapsed_seconds(started_at, finished_at),
            "run_status": status,
            "run_failed_stage": failed_stage or metadata.get("run_failed_stage"),
            "run_stage_timings": (
                stage_timings
                if stage_timings is not None
                else metadata.get("run_stage_timings", {})
            ),
            "run_event_count": count_run_events(output_dir),
            "run_summary_path": str(run_summary_path(output_dir).resolve()),
            "run_events_path": str(run_events_path(output_dir).resolve()),
            "run_error_type": error.__class__.__name__,
            "run_error_message": _safe_error_message(error),
        }
    )
    _write_metadata(output_dir, metadata)
    write_run_summary(output_dir)
    return metadata


def write_extraction_artifacts(
    output_path: str | Path,
    candidate_entities: list[ExtractedEntity],
    candidate_relationships: list[ExtractedRelationship],
    candidate_triples: list[CandidateTriple],
    extractor: str,
    extraction_metadata: dict | None = None,
) -> dict:
    """Write candidate extraction artifacts and update index metadata."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [asdict(entity) for entity in candidate_entities],
        columns=list(CANDIDATE_ENTITIES_COLUMNS),
    ).to_parquet(
        output_dir / "candidate_entities.parquet",
        index=False,
    )
    pd.DataFrame(
        [asdict(rel) for rel in candidate_relationships],
        columns=list(CANDIDATE_RELATIONSHIPS_COLUMNS),
    ).to_parquet(
        output_dir / "candidate_relationships.parquet",
        index=False,
    )
    pd.DataFrame(
        [asdict(triple) for triple in candidate_triples],
        columns=list(CANDIDATE_TRIPLES_COLUMNS),
    ).to_parquet(
        output_dir / "candidate_triples.parquet",
        index=False,
    )

    metadata = _read_metadata(output_dir)

    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "extractor": extractor,
            "num_candidate_entities": len(candidate_entities),
            "num_candidate_relationships": len(candidate_relationships),
            "num_candidate_triples": len(candidate_triples),
        }
    )
    if extraction_metadata:
        metadata.update(extraction_metadata)
    _write_metadata(output_dir, metadata)
    return metadata


def write_fusion_artifacts(
    output_path: str | Path,
    fusion_result: FusionResult,
    min_confidence: float,
    graph_store_provider: str | None = None,
    graph_store_written: bool | None = None,
    graph_store_error: str | None = None,
    graph_store_metadata: dict | None = None,
) -> dict:
    """Write fused graph artifacts and update index metadata."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [asdict(entity) for entity in fusion_result.entities],
        columns=list(ENTITIES_COLUMNS),
    ).to_parquet(
        output_dir / "entities.parquet",
        index=False,
    )
    pd.DataFrame(
        [asdict(relationship) for relationship in fusion_result.relationships],
        columns=list(RELATIONSHIPS_COLUMNS),
    ).to_parquet(output_dir / "relationships.parquet", index=False)
    rejected_df = pd.DataFrame(
        [asdict(triple) for triple in fusion_result.rejected_triples],
        columns=[field.name for field in fields(CandidateTriple)],
    )
    rejected_df.to_parquet(output_dir / "rejected_triples.parquet", index=False)
    (output_dir / "graph.json").write_text(
        json.dumps(fusion_result.graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = _read_metadata(output_dir)
    update = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "num_entities": len(fusion_result.entities),
        "num_relationships": len(fusion_result.relationships),
        "num_rejected_triples": len(fusion_result.rejected_triples),
        "fusion_min_confidence": min_confidence,
        **fusion_result.metadata,
    }
    if graph_store_provider is not None:
        update["graph_store_provider"] = graph_store_provider
    if graph_store_written is not None:
        update["graph_store_written"] = graph_store_written
    if graph_store_metadata:
        update.update(graph_store_metadata)
    if graph_store_error is not None:
        update["graph_store_error"] = str(redact_secrets(graph_store_error))
    else:
        metadata.pop("graph_store_error", None)
    metadata.update(update)
    _write_metadata(output_dir, metadata)
    return metadata


def write_community_artifacts(
    output_path: str | Path,
    communities: list[Community],
    reports: list[CommunityReport],
    algorithm: str,
    reporter: str,
) -> dict:
    """Write community artifacts and update index metadata."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [asdict(community) for community in communities],
        columns=[field.name for field in fields(Community)],
    ).to_parquet(output_dir / "communities.parquet", index=False)
    pd.DataFrame(
        [asdict(report) for report in reports],
        columns=[field.name for field in fields(CommunityReport)],
    ).to_parquet(output_dir / "community_reports.parquet", index=False)

    metadata = _read_metadata(output_dir)
    metadata.update(
        {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "num_communities": len(communities),
            "num_community_reports": len(reports),
            "community_algorithm": algorithm,
            "community_reporter": reporter,
        }
    )
    _write_metadata(output_dir, metadata)
    return metadata


def _read_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "index_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    redacted_metadata = redact_secrets(metadata)
    metadata.clear()
    metadata.update(redacted_metadata)
    (output_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _document_scan_metadata(
    document_scan: DocumentScanResult | None,
    documents: list[SourceDocument],
) -> dict[str, int]:
    if document_scan is None:
        return {
            "num_input_files": len(documents),
            "num_included_files": len(documents),
            "num_ignored_files": 0,
            "num_rejected_files": 0,
            "num_empty_documents": sum(
                1 for document in documents if not document.text.strip()
            ),
        }
    return {
        "num_input_files": document_scan.num_files,
        "num_included_files": document_scan.num_included_files,
        "num_ignored_files": document_scan.num_ignored_files,
        "num_rejected_files": document_scan.num_rejected_files,
        "num_empty_documents": document_scan.num_empty_documents,
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _elapsed_seconds(started_at: datetime | None, finished_at: datetime) -> float | None:
    if started_at is None:
        return None
    return round((finished_at - started_at).total_seconds(), 6)


def _safe_error_message(exc: Exception) -> str:
    return str(redact_secrets(str(exc)))


def _clear_run_error(metadata: dict[str, Any]) -> None:
    metadata.pop("run_error_type", None)
    metadata.pop("run_error_message", None)


def _clear_generated_artifacts(output_dir: Path) -> None:
    for artifact_name in GENERATED_ARTIFACTS:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()


def _clear_stage_metadata(metadata: dict[str, Any]) -> None:
    for key in STAGE_METADATA_KEYS:
        metadata.pop(key, None)
    for key in list(metadata):
        if any(key.startswith(prefix) for prefix in STAGE_METADATA_PREFIXES):
            metadata.pop(key, None)
