#!/usr/bin/env python3
"""Run HotpotQA with one gold-support graph index per question."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_hotpotqa_mini import (  # noqa: E402
    answer_token_f1,
    build_sample_manifest,
    evaluate_hotpotqa_questions,
    load_hotpotqa_records,
    prepare_hotpotqa_mini,
    select_hotpotqa_records,
)
from graphrag_v2.config import load_config  # noqa: E402
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID  # noqa: E402
from graphrag_v2.llm import create_chat_provider  # noqa: E402
from graphrag_v2.qa import LLMAnswerer  # noqa: E402
from graphrag_v2.qa.prompts import QA_ANSWER_PROMPT_VERSION  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("artifacts/hotpotqa-isolated")
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42


def run_isolated_hotpotqa_benchmark(
    *,
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    answerer_name: str = "mock",
    config_path: Path | None = None,
    reset_output: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    records = load_hotpotqa_records(input_path)
    selected = select_hotpotqa_records(records, sample_size=sample_size, seed=seed)
    if not selected:
        raise ValueError("No HotpotQA records selected.")
    if reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest_path = output_dir / "sample_manifest.json"
    sample_manifest = build_sample_manifest(
        records,
        sample_sizes=sorted({50, 100, 200, sample_size}),
        seed=seed,
        source_file=input_path,
    )
    sample_manifest["active_sample"] = {
        "requested_size": sample_size,
        "actual_size": len(selected),
        "ids": [str(record.get("_id") or record.get("id") or "") for record in selected],
    }
    sample_manifest_path.write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    llm_client = None
    answerer = None
    if answerer_name == "llm":
        if config_path is None:
            raise ValueError("--config is required with --answerer llm")
        config = load_config(config_path)
        model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
        llm_client = create_chat_provider(
            provider=config.extraction.llm_provider,
            model_config=model_config,
            require_real=True,
        )
        answerer = LLMAnswerer(llm_client=llm_client)

    cases = []
    for ordinal, record in enumerate(selected, start=1):
        hotpotqa_id = str(record.get("_id") or record.get("id") or ordinal)
        case_dir = cases_dir / f"{ordinal:03d}-{_slug(hotpotqa_id)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_started_at = time.perf_counter()
        prepared = None
        question = None
        try:
            case_input_path = case_dir / "input.json"
            case_input_path.write_text(
                json.dumps([record], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            prepared = prepare_hotpotqa_mini(
                input_path=case_input_path,
                output_dir=case_dir,
                sample_size=1,
                seed=seed,
                reset_output=False,
            )
            questions = _read_jsonl(prepared.questions_path)
            if len(questions) != 1:
                raise ValueError(f"Expected one question for {hotpotqa_id}, got {len(questions)}")
            question = questions[0]
            report = evaluate_hotpotqa_questions(
                index_path=prepared.index_path,
                questions=questions,
                answerer=answerer,
            )
            case = dict(report["cases"][0])
            case.update(
                {
                    "ordinal": ordinal,
                    "hotpotqa_id": hotpotqa_id,
                    "type": str(question.get("type") or record.get("type") or "unknown"),
                    "level": str((question.get("metadata") or {}).get("level") or record.get("level") or "unknown"),
                    "error": None,
                    "case_index_path": str(prepared.index_path),
                    "answer_prompt_version": QA_ANSWER_PROMPT_VERSION,
                }
            )
        except Exception as exc:
            case = _error_case(
                record=record,
                question=question,
                ordinal=ordinal,
                hotpotqa_id=hotpotqa_id,
                case_index_path=prepared.index_path if prepared else case_dir / "index",
                error=f"{exc.__class__.__name__}: {exc}",
                latency_seconds=round(time.perf_counter() - case_started_at, 6),
            )
        cases.append(case)

    summary = summarize_cases(cases, elapsed_seconds=time.perf_counter() - started_at)
    report = {
        "benchmark": {
            "name": "HotpotQA Isolated",
            "source_file": str(input_path),
            "sample_size": len(selected),
            "seed": seed,
            "selection": "fixed-seed sample from queryable HotpotQA dev records",
            "index_mode": "one gold HotpotQA context index per question",
            "answerer": "real_llm" if answerer_name == "llm" else "mock",
            "provider": str(getattr(llm_client, "provider_name", "")) if llm_client else None,
            "model": str(
                getattr(llm_client, "model_name", None)
                or getattr(llm_client, "model", "")
            )
            if llm_client
            else None,
            "mock_mode": bool(getattr(llm_client, "mock_mode", False)) if llm_client else None,
            "answer_prompt_version": QA_ANSWER_PROMPT_VERSION,
            "sample_manifest_path": str(sample_manifest_path),
        },
        "summary": summary,
        "llm_stats": _client_stats(llm_client),
        "cases": cases,
    }
    write_reports(output_dir, report)
    return report


def summarize_cases(cases: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    total = len(cases)
    return {
        "total": total,
        "passed": sum(1 for case in cases if case["passed"]),
        "failed": sum(1 for case in cases if not case["passed"]),
        "errors": sum(1 for case in cases if case.get("error")),
        "citation_coverage": _ratio(
            sum(1 for case in cases if case["all_required_citations_found"]),
            total,
        ),
        "supporting_citation_recall": _ratio(
            sum(len(set(case["required_citations"]) & set(case["citations"])) for case in cases),
            sum(len(case["required_citations"]) for case in cases),
        ),
        "answer_contains_gold_rate": _ratio(
            sum(1 for case in cases if case["answer_contains_gold"]),
            total,
        ),
        "average_token_f1": _mean([case["answer_token_f1"] for case in cases]),
        "exact_match": _ratio(
            sum(
                1
                for case in cases
                if answer_token_f1(case["gold_answer"], case["answer"]) == 1.0
                and _normalized(case["gold_answer"]) == _normalized(case["answer"])
            ),
            total,
        ),
        "average_latency_seconds": _mean([case["latency_seconds"] for case in cases]),
        "run_seconds": round(elapsed_seconds, 6),
    }


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "hotpotqa-isolated-report.json"
    markdown_path = output_dir / "hotpotqa-isolated-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(format_markdown_report(report), encoding="utf-8")


def _error_case(
    *,
    record: dict[str, Any],
    question: dict[str, Any] | None,
    ordinal: int,
    hotpotqa_id: str,
    case_index_path: Path,
    error: str,
    latency_seconds: float,
) -> dict[str, Any]:
    required_citations = [
        str(item) for item in (question or {}).get("required_citations", [])
    ]
    return {
        "id": str((question or {}).get("id") or f"hotpotqa_{hotpotqa_id}"),
        "question": str((question or {}).get("question") or record.get("question") or ""),
        "gold_answer": str((question or {}).get("answer") or record.get("answer") or ""),
        "answer": "",
        "route": None,
        "refused": False,
        "latency_seconds": latency_seconds,
        "required_citations": required_citations,
        "citations": [],
        "citation_hits": [],
        "missing_required_citations": required_citations,
        "citation_recall": 0.0,
        "all_required_citations_found": False,
        "answer_contains_gold": False,
        "answer_token_f1": 0.0,
        "query_trace": {},
        "linked_entities": [],
        "retrieved_relationships": [],
        "retrieved_relationship_hops": [],
        "retrieved_text_chunks": [],
        "adaptive_enabled": True,
        "adaptive_triggered": False,
        "matched_adaptive_cues": [],
        "hop_plan": [],
        "relationship_count_by_hop": {},
        "max_retrieved_hop": 0,
        "passed": False,
        "ordinal": ordinal,
        "hotpotqa_id": hotpotqa_id,
        "type": str((question or {}).get("type") or record.get("type") or "unknown"),
        "level": str(((question or {}).get("metadata") or {}).get("level") or record.get("level") or "unknown"),
        "error": error,
        "case_index_path": str(case_index_path),
        "answer_prompt_version": QA_ANSWER_PROMPT_VERSION,
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    summary = report["summary"]
    lines = [
        "# HotpotQA Isolated Benchmark Report",
        "",
        f"- Source: `{benchmark['source_file']}`",
        f"- Sample size: `{benchmark['sample_size']}`",
        f"- Seed: `{benchmark['seed']}`",
        f"- Mode: {benchmark['index_mode']}",
        f"- Answerer: {benchmark['answerer']}",
        f"- Answer prompt version: `{benchmark['answer_prompt_version']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "total",
        "passed",
        "failed",
        "errors",
        "citation_coverage",
        "supporting_citation_recall",
        "answer_contains_gold_rate",
        "average_token_f1",
        "exact_match",
        "average_latency_seconds",
        "run_seconds",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| # | Passed | Citation Recall | Answer Hit | Token F1 | Question |",
            "| ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["ordinal"]),
                    str(case["passed"]),
                    str(case["citation_recall"]),
                    str(case["answer_contains_gold"]),
                    str(case["answer_token_f1"]),
                    _md_cell(str(case["question"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated HotpotQA benchmark with one index per question.",
    )
    parser.add_argument("--input", required=True, help="HotpotQA JSON/JSONL path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--answerer", choices=["mock", "llm"], default="mock")
    parser.add_argument("--config", default=None, help="Settings YAML required for llm answerer.")
    parser.add_argument("--no-reset-output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_isolated_hotpotqa_benchmark(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            sample_size=args.sample_size,
            seed=args.seed,
            answerer_name=args.answerer,
            config_path=Path(args.config) if args.config else None,
            reset_output=not args.no_reset_output,
        )
    except Exception as exc:
        print(f"HotpotQA isolated benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _client_stats(llm_client: Any | None) -> dict[str, Any]:
    if llm_client is None:
        return {}
    get_stats = getattr(llm_client, "get_stats", None)
    if callable(get_stats):
        stats = get_stats()
        if isinstance(stats, dict):
            return stats
    return {}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _normalized(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("_") or "case"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:180]


if __name__ == "__main__":
    sys.exit(main())
