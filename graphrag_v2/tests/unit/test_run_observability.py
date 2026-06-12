"""Unit tests for run observability helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from graphrag_v2.artifacts.run_observability import (
    RunObserver,
    append_run_event,
    build_run_summary,
    format_run_report,
    inspect_run_summary,
    load_run_events,
    new_run_id,
    redact_secrets,
    write_run_summary,
)


def test_redact_secrets_redacts_all_operational_env_values(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-secret")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "zhipu-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("KGQA_REAL_LLM_API_KEY", "generic-llm-secret")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "graphrag-secret")
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-llm-secret")

    payload = {
        "message": (
            "neo4j-secret zhipu-secret graphrag-secret embedding-secret "
            "openai-secret local-llm-secret deepseek-secret generic-llm-secret"
        ),
        "nested": [
            "embedding-secret",
            {"token": "zhipu-secret openai-secret deepseek-secret"},
        ],
        "tupled": ("neo4j-secret", {"embedding": "embedding-secret"}),
        "local": {"api_key": "local-llm-secret", "generic": "generic-llm-secret"},
        "unchanged": 7,
    }

    redacted = redact_secrets(payload)
    serialized = json.dumps(redacted, ensure_ascii=False)

    assert "neo4j-secret" not in serialized
    assert "zhipu-secret" not in serialized
    assert "deepseek-secret" not in serialized
    assert "generic-llm-secret" not in serialized
    assert "graphrag-secret" not in serialized
    assert "embedding-secret" not in serialized
    assert "openai-secret" not in serialized
    assert "local-llm-secret" not in serialized
    assert serialized.count("***") == 16
    assert redacted["tupled"] == ("***", {"embedding": "***"})
    assert redacted["local"] == {"api_key": "***", "generic": "***"}
    assert redacted["unchanged"] == 7


def test_redact_secrets_redacts_dictionary_keys(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_KEY", "embedding-secret")

    redacted = redact_secrets(
        {
            "embedding-secret-key": {
                "nested-embedding-secret": "safe",
            },
        }
    )

    serialized = json.dumps(redacted, ensure_ascii=False)

    assert "embedding-secret" not in serialized
    assert redacted == {"***-key": {"nested-***": "safe"}}


def test_new_run_id_uses_utc_timestamp_and_short_token():
    run_id = new_run_id(
        now=datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC),
        token="abcdef123456",
    )

    assert run_id == "run_20260606T120000Z_abcdef12"


def test_append_run_event_writes_redacted_jsonl(monkeypatch, temp_dir):
    monkeypatch.setenv("GRAPHRAG_API_KEY", "artifact-secret")

    record = append_run_event(
        temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
        stage="extraction",
        event="stage_failed",
        status="failed",
        elapsed_seconds=0.5,
        counts={"num_candidate_entities": 2},
        provider="llm",
        model="glm-4",
        error_type="ValueError",
        error_message="bad artifact-secret response",
        timestamp="2026-06-06T12:00:00+00:00",
    )

    events_path = temp_dir / "run_events.jsonl"

    assert record["event_schema_version"] == 1
    assert record["timestamp"] == "2026-06-06T12:00:00+00:00"
    assert record["run_id"] == "run_20260606T120000Z_abcdef12"
    assert record["index_id"] == "kgqa_test"
    assert record["stage"] == "extraction"
    assert record["event"] == "stage_failed"
    assert record["status"] == "failed"
    assert record["elapsed_seconds"] == 0.5
    assert record["counts"] == {"num_candidate_entities": 2}
    assert record["provider"] == "llm"
    assert record["model"] == "glm-4"
    assert record["error_type"] == "ValueError"
    assert record["error_message"] == "bad *** response"
    assert events_path.exists()
    assert load_run_events(temp_dir) == [record]
    assert "artifact-secret" not in events_path.read_text(encoding="utf-8")


def test_run_observer_stage_elapsed_records_timing(monkeypatch, temp_dir):
    observer = RunObserver(
        output_path=temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
    )
    observer._stage_starts["document"] = 10.0
    monkeypatch.setattr(
        "graphrag_v2.artifacts.run_observability.time.perf_counter",
        lambda: 10.5,
    )

    elapsed_seconds = observer._stage_elapsed("document")

    assert elapsed_seconds == 0.5
    assert observer.stage_timings["document"] == 0.5
    assert "document" not in observer._stage_starts


def test_run_observer_stage_elapsed_accumulates_repeated_stage_timings(
    monkeypatch,
    temp_dir,
):
    observer = RunObserver(
        output_path=temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
    )
    perf_counter_values = iter([10.0, 10.2, 20.0, 20.3])
    monkeypatch.setattr(
        "graphrag_v2.artifacts.run_observability.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    observer.stage_start("graph_store")
    first_elapsed = observer._stage_elapsed("graph_store")
    observer.stage_start("graph_store")
    second_elapsed = observer._stage_elapsed("graph_store")

    assert first_elapsed == 0.2
    assert second_elapsed == 0.3
    assert observer.stage_timings["graph_store"] == 0.5
    assert "graph_store" not in observer._stage_starts


def test_run_observer_stage_elapsed_returns_none_for_unstarted_stage(temp_dir):
    observer = RunObserver(
        output_path=temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
    )

    elapsed_seconds = observer._stage_elapsed("document")

    assert elapsed_seconds is None
    assert "document" not in observer.stage_timings


def test_run_observer_run_events_use_run_stage_and_resolve_input_path(temp_dir):
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    unresolved_input_path = input_dir / ".." / "input"
    observer = RunObserver(
        output_path=temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
        stage_timings={"document": 1.0},
    )

    start_record = observer.run_start("full", unresolved_input_path)
    end_record = observer.run_end()
    failed_record = observer.run_failed(RuntimeError("boom"))

    assert start_record["stage"] == "run"
    assert start_record["event"] == "run_start"
    assert start_record["counts"]["input_path"] == str(input_dir.resolve())
    assert end_record["stage"] == "run"
    assert end_record["event"] == "run_end"
    assert failed_record["stage"] == "run"
    assert failed_record["event"] == "run_failed"


def test_run_observer_run_failed_stays_run_after_stage_failed(temp_dir):
    observer = RunObserver(
        output_path=temp_dir,
        run_id="run_20260606T120000Z_abcdef12",
        index_id="kgqa_test",
    )
    exc = RuntimeError("boom")

    observer.stage_failed("extraction", exc)
    observer.run_failed(exc)

    events = load_run_events(temp_dir)
    assert [(event["stage"], event["event"]) for event in events] == [
        ("extraction", "stage_failed"),
        ("run", "run_failed"),
    ]


def test_run_observer_from_metadata_rejects_missing_run_id(temp_dir):
    metadata = {"index_id": "kgqa_test"}

    with pytest.raises(ValueError, match="run_id"):
        RunObserver.from_metadata(temp_dir, metadata)


def test_run_observer_from_metadata_rejects_missing_index_id(temp_dir):
    metadata = {"run_id": "run_20260606T120000Z_abcdef12"}

    with pytest.raises(ValueError, match="index_id"):
        RunObserver.from_metadata(temp_dir, metadata)


def test_run_observer_from_metadata_preserves_identity_and_state(temp_dir):
    metadata = {
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "run_event_count": "7",
        "run_stage_timings": {"document": 1.25},
        "run_failed_stage": "extraction",
    }

    observer = RunObserver.from_metadata(temp_dir, metadata)

    assert observer.output_path == temp_dir
    assert observer.run_id == "run_20260606T120000Z_abcdef12"
    assert observer.index_id == "kgqa_test"
    assert observer.event_count == 7
    assert observer.stage_timings == {"document": 1.25}
    assert observer.failed_stage == "extraction"


def test_load_run_events_reports_malformed_line_with_path_and_line(temp_dir):
    (temp_dir / "run_events.jsonl").write_text(
        '{"event": "run_start"}\n{bad-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_run_events(temp_dir)

    message = str(exc_info.value)
    assert "run_events.jsonl" in message
    assert "line 2" in message


def test_load_run_events_rejects_non_dict_json_line(temp_dir):
    (temp_dir / "run_events.jsonl").write_text(
        '{"event": "run_start"}\n42\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_run_events(temp_dir)

    message = str(exc_info.value)
    assert "run_events.jsonl" in message
    assert "line 2" in message
    assert "object" in message or "dict" in message


def test_build_run_summary_deduplicates_matching_errors():
    metadata = {
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "run_status": "failed",
        "run_failed_stage": "extraction",
        "run_finished_at": "2026-06-06T12:00:01+00:00",
        "run_error_type": "ValueError",
        "run_error_message": "bad response",
    }
    events = [
        {
            "stage": "extraction",
            "event": "stage_failed",
            "status": "failed",
            "timestamp": "2026-06-06T12:00:00+00:00",
            "error_type": "ValueError",
            "error_message": "bad response",
        },
        {
            "stage": "run",
            "event": "run_failed",
            "status": "failed",
            "timestamp": "2026-06-06T12:00:01+00:00",
            "error_type": "ValueError",
            "error_message": "bad response",
        },
    ]

    summary = build_run_summary(metadata, events)

    matching_errors = [
        error
        for error in summary["errors"]
        if (
            error["stage"],
            error["error_type"],
            error["error_message"],
        )
        == ("extraction", "ValueError", "bad response")
    ]
    assert len(matching_errors) == 1
    assert all(
        not (
            error["stage"] == "run"
            and error["error_type"] == "ValueError"
            and error["error_message"] == "bad response"
        )
        for error in summary["errors"]
    )


def test_write_run_summary_maps_metadata_and_stage_timings(temp_dir):
    metadata = {
        "metadata_schema_version": 1,
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "run_status": "succeeded",
        "run_mode": "fusion-only",
        "input_path": "/tmp/input",
        "output_path": str(temp_dir.resolve()),
        "run_started_at": "2026-06-06T12:00:00+00:00",
        "run_finished_at": "2026-06-06T12:00:04+00:00",
        "run_elapsed_seconds": 4.0,
        "run_failed_stage": None,
        "run_stage_timings": {
            "document": 1.0,
            "extraction": 1.5,
            "fusion": 1.0,
            "graph_store": 0.5,
        },
        "num_documents": 1,
        "num_text_units": 2,
        "num_input_files": 1,
        "num_included_files": 1,
        "num_ignored_files": 0,
        "num_rejected_files": 0,
        "num_empty_documents": 0,
        "extractor": "llm",
        "num_candidate_entities": 3,
        "num_candidate_relationships": 2,
        "num_candidate_triples": 2,
        "llm_model_id": "default_chat_model",
        "llm_model_name": "glm-test",
        "extraction_repair_attempts": 2,
        "extraction_failed_chunks": 1,
        "extraction_failed_chunk_ids": ["chunk_7"],
        "extraction_salvaged_entities": 1,
        "extraction_salvaged_relationships": 1,
        "extraction_dropped_entities": 3,
        "extraction_budget_exceeded": False,
        "extraction_cache_enabled": True,
        "extraction_cache_hits": 4,
        "extraction_cache_misses": 2,
        "num_entities": 2,
        "num_relationships": 1,
        "num_rejected_triples": 0,
        "fusion_min_confidence": 0.8,
        "graph_store_provider": "json",
        "graph_store_written": True,
        "graph_store_health_status": "ready",
    }
    (temp_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = write_run_summary(temp_dir)

    assert summary["summary_schema_version"] == 1
    assert summary["run_id"] == "run_20260606T120000Z_abcdef12"
    assert summary["status"] == "succeeded"
    assert summary["failed_stage"] is None
    assert summary["document"]["num_text_units"] == 2
    assert summary["document"]["elapsed_seconds"] == 1.0
    assert summary["extraction"]["extractor"] == "llm"
    assert summary["extraction"]["llm_model_id"] == "default_chat_model"
    assert summary["extraction"]["llm_model_name"] == "glm-test"
    assert summary["extraction"]["extraction_repair_attempts"] == 2
    assert summary["extraction"]["extraction_failed_chunks"] == 1
    assert summary["extraction"]["extraction_failed_chunk_ids"] == ["chunk_7"]
    assert summary["extraction"]["extraction_salvaged_entities"] == 1
    assert summary["extraction"]["extraction_cache_hits"] == 4
    assert summary["fusion"]["num_entities"] == 2
    assert summary["graph_store"]["provider"] == "json"
    assert summary["graph_store"]["elapsed_seconds"] == 0.5
    assert summary["community"]["num_communities"] is None
    assert summary["errors"] == []
    assert (temp_dir / "run_summary.json").exists()


def test_format_run_report_marks_missing_summary(temp_dir):
    metadata = {
        "metadata_schema_version": 1,
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "run_status": "succeeded",
        "run_mode": "documents-only",
        "input_path": "/tmp/input",
        "output_path": str(temp_dir.resolve()),
        "run_started_at": "2026-06-06T12:00:00+00:00",
        "run_finished_at": "2026-06-06T12:00:01+00:00",
        "run_elapsed_seconds": 1.0,
        "run_failed_stage": None,
        "run_stage_timings": {"document": 1.0},
        "num_documents": 1,
        "num_text_units": 1,
    }
    (temp_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary, summary_status = inspect_run_summary(temp_dir)
    report = format_run_report(
        temp_dir,
        summary=summary,
        summary_status=summary_status,
    )

    assert summary_status == "missing"
    assert "Run:" in report
    assert "summary_status: missing" in report
    assert "run_20260606T120000Z_abcdef12" in report
    assert "  failed_stage: None" in report
    assert "Stages:" in report
    assert "document: elapsed_seconds=1.0, num_text_units=1" in report
    assert "Artifacts:" in report
    assert f"  metadata_path: {(temp_dir / 'index_metadata.json').resolve()}" in report
    assert f"  run_summary_path: {(temp_dir / 'run_summary.json').resolve()}" in report
    assert f"  run_events_path: {(temp_dir / 'run_events.jsonl').resolve()}" in report


def test_existing_run_summary_and_report_are_redacted(monkeypatch, temp_dir):
    monkeypatch.setenv("GRAPHRAG_API_KEY", "summary-secret")
    summary_payload = {
        "summary_schema_version": 1,
        "run_id": "run_20260606T120000Z_abcdef12",
        "index_id": "kgqa_test",
        "status": "failed",
        "mode": "full",
        "elapsed_seconds": 1.0,
        "failed_stage": "extraction",
        "document": {},
        "extraction": {},
        "fusion": {},
        "graph_store": {},
        "community": {},
        "errors": [
            {
                "stage": "extraction",
                "event": "stage_failed",
                "status": "failed",
                "timestamp": "2026-06-06T12:00:00+00:00",
                "error_type": "ValueError",
                "error_message": "bad summary-secret response",
            }
        ],
    }
    (temp_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary, summary_status = inspect_run_summary(temp_dir)
    report = format_run_report(temp_dir, summary, summary_status)
    serialized_summary = json.dumps(summary, ensure_ascii=False)

    assert summary_status == "present"
    assert "summary-secret" not in serialized_summary
    assert "summary-secret" not in report
    assert "***" in serialized_summary
    assert "***" in report
