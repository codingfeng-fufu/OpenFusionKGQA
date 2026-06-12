"""Tests for benchmark suite orchestration and report summaries."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_benchmark_suite import build_benchmark_commands, is_successful_exit
from scripts.summarize_benchmark_reports import (
    classify_case_failure,
    main as summarize_main,
    passes_thresholds,
    summarize_report,
)


def test_build_benchmark_commands_keeps_logs_outside_reset_outputs(tmp_path):
    commands = build_benchmark_commands(
        hotpotqa_input=Path("/data/dev.json"),
        output_root=tmp_path / "bench",
        config_path=Path("settings.local.real-llm.yaml"),
        run_real_llm=True,
    )

    log_paths = [item.log_path for item in commands]
    output_paths = [item.output_path for item in commands]

    assert commands[0].name == "offline_demo"
    assert any(item.name == "hotpotqa_isolated_real20" for item in commands)
    assert all("logs" in str(path) for path in log_paths)
    assert all(log_path.parent not in output_paths for log_path in log_paths)


def test_build_benchmark_commands_runs_offline_qa_eval(tmp_path):
    output_root = tmp_path / "bench"

    commands = build_benchmark_commands(
        hotpotqa_input=Path("/data/dev.json"),
        output_root=output_root,
        config_path=Path("settings.local.real-llm.yaml"),
        run_real_llm=False,
    )

    names = [item.name for item in commands]
    assert names[:2] == ["offline_demo", "offline_qa_eval"]
    eval_command = commands[1]
    assert "scripts/evaluate_qa.py" in eval_command.argv
    assert "--format" in eval_command.argv
    assert "markdown" in eval_command.argv
    assert str(output_root / "offline-demo" / "qa-eval.md") in eval_command.argv


def test_build_benchmark_commands_includes_mini_real_llm_smoke(tmp_path):
    commands = build_benchmark_commands(
        hotpotqa_input=Path("/data/dev.json"),
        output_root=tmp_path / "bench",
        config_path=Path("settings.local.real-llm.yaml"),
        run_real_llm=True,
    )

    mini_real = next(item for item in commands if item.name == "hotpotqa_mini_real3")
    assert "--real-llm-smoke-size" in mini_real.argv
    assert "3" in mini_real.argv
    assert "--config" in mini_real.argv
    assert "settings.local.real-llm.yaml" in mini_real.argv


def test_hotpotqa_mini_allows_diagnostic_quality_failure_exit_code(tmp_path):
    commands = build_benchmark_commands(
        hotpotqa_input=Path("/data/dev.json"),
        output_root=tmp_path / "bench",
        config_path=Path("settings.local.real-llm.yaml"),
        run_real_llm=False,
    )

    mini = next(item for item in commands if item.name == "hotpotqa_mini_25")
    assert is_successful_exit(mini, 1)
    assert not is_successful_exit(mini, 2)


def test_classify_case_failure_detects_answer_selection_miss():
    case = {
        "passed": False,
        "all_required_citations_found": True,
        "citation_recall": 1.0,
        "answer_contains_gold": False,
        "error": None,
        "answer": "Republic of Genoa",
    }

    assert classify_case_failure(case) == "answer_selection_miss"


def test_classify_case_failure_detects_retrieval_miss():
    case = {
        "passed": False,
        "retrieval_hit": False,
        "all_required_citations_found": False,
        "citation_recall": 0.0,
        "answer_contains_gold": False,
        "error": None,
    }

    assert classify_case_failure(case) == "retrieval_miss"


def test_classify_case_failure_detects_retrieval_or_citation_miss():
    case = {
        "passed": False,
        "all_required_citations_found": False,
        "citation_recall": 0.5,
        "answer_contains_gold": False,
        "error": None,
        "answer": "Sardinia",
    }

    assert classify_case_failure(case) == "citation_miss"


def test_classify_case_failure_detects_runtime_error():
    case = {"passed": False, "error": "ValueError: bad response"}

    assert classify_case_failure(case) == "runtime_error"


def test_passes_thresholds_accepts_beta_gate_metrics():
    summary = {
        "exact_match": 0.65,
        "average_token_f1": 0.68,
        "supporting_citation_recall": 0.99,
        "errors": 0,
    }

    assert passes_thresholds(summary, min_em=0.6, min_f1=0.65, min_support_recall=0.95)


def test_passes_thresholds_rejects_low_answer_quality():
    summary = {
        "exact_match": 0.53,
        "average_token_f1": 0.61,
        "supporting_citation_recall": 1.0,
        "errors": 0,
    }

    assert not passes_thresholds(summary, min_em=0.6, min_f1=0.65, min_support_recall=0.95)


def test_summarize_report_keeps_real_llm_smoke_section():
    report = {
        "benchmark": {"name": "HotpotQA Mini"},
        "offline_qa": {
            "summary": {"total": 10, "passed": 10, "failed": 0},
            "cases": [{"passed": True}],
        },
        "real_llm_smoke": {
            "enabled": True,
            "sample_size": 3,
            "summary": {"total": 3, "passed": 2, "failed": 1},
            "cases": [
                {
                    "passed": False,
                    "all_required_citations_found": True,
                    "answer_contains_gold": False,
                    "answer": "wrong",
                }
            ],
        },
    }

    summary = summarize_report(report)

    real_section = next(
        section for section in summary["sections"] if section["name"] == "real_llm_smoke"
    )
    assert real_section["summary"]["total"] == 3
    assert real_section["failure_taxonomy"] == {"answer_selection_miss": 1}


def test_summary_cli_returns_nonzero_when_threshold_gate_fails(tmp_path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "summary.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark": {"name": "HotpotQA Isolated"},
                "summary": {
                    "exact_match": 0.53,
                    "average_token_f1": 0.61,
                    "supporting_citation_recall": 1.0,
                    "errors": 0,
                },
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    status = summarize_main(
        [
            str(report_path),
            "--output",
            str(output_path),
            "--min-em",
            "0.6",
            "--min-f1",
            "0.65",
            "--min-support-recall",
            "0.95",
        ]
    )

    assert status == 1
