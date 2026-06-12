#!/usr/bin/env python3
"""Summarize KGQA benchmark reports and classify failure modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def classify_case_failure(case: dict[str, Any]) -> str:
    if case.get("passed"):
        return "passed"
    if case.get("error"):
        return "runtime_error"
    if _case_indicates_retrieval_miss(case):
        return "retrieval_miss"
    if not case.get("all_required_citations_found", False):
        return "citation_miss"
    if case.get("refused") or str(case.get("answer", "")).startswith("证据不足"):
        return "refusal_miss"
    if not case.get("answer_contains_gold", False):
        return "answer_selection_miss"
    return "metric_miss"


def _case_indicates_retrieval_miss(case: dict[str, Any]) -> bool:
    if case.get("retrieval_hit") is False:
        return True
    checks = case.get("checks")
    if isinstance(checks, dict):
        retrieval = checks.get("retrieval")
        if isinstance(retrieval, dict) and retrieval.get("passed") is False:
            return True
    return False


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    sections = _report_sections(report)
    failure_counts: Counter[str] = Counter()
    for section in sections:
        failure_counts.update(section["failure_taxonomy"])
    return {
        "benchmark": report.get("benchmark", {}).get("name", "unknown"),
        "summary": _primary_summary(report),
        "failure_taxonomy": dict(sorted(failure_counts.items())),
        "sections": sections,
    }


def _primary_summary(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("summary"), dict):
        return report["summary"]
    offline_qa = report.get("offline_qa")
    if isinstance(offline_qa, dict) and isinstance(offline_qa.get("summary"), dict):
        return offline_qa["summary"]
    real_llm_smoke = report.get("real_llm_smoke")
    if isinstance(real_llm_smoke, dict) and isinstance(
        real_llm_smoke.get("summary"),
        dict,
    ):
        return real_llm_smoke["summary"]
    return {}


def _report_sections(report: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    if isinstance(report.get("summary"), dict) or isinstance(report.get("cases"), list):
        sections.append(
            _summarize_section(
                "top_level",
                report.get("summary") if isinstance(report.get("summary"), dict) else {},
                report.get("cases") if isinstance(report.get("cases"), list) else [],
            )
        )
    for name in ("offline_qa", "real_llm_smoke"):
        section = report.get(name)
        if not isinstance(section, dict):
            continue
        if not isinstance(section.get("summary"), dict) and not isinstance(
            section.get("cases"),
            list,
        ):
            continue
        sections.append(
            _summarize_section(
                name,
                section.get("summary") if isinstance(section.get("summary"), dict) else {},
                section.get("cases") if isinstance(section.get("cases"), list) else [],
            )
        )
    return sections


def _summarize_section(
    name: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_counts = Counter(classify_case_failure(case) for case in cases)
    return {
        "name": name,
        "summary": summary,
        "failure_taxonomy": dict(sorted(failure_counts.items())),
    }


def passes_thresholds(
    summary: dict[str, Any],
    *,
    min_em: float,
    min_f1: float,
    min_support_recall: float,
) -> bool:
    return (
        float(summary.get("exact_match", 0.0)) >= min_em
        and float(summary.get("average_token_f1", 0.0)) >= min_f1
        and float(summary.get("supporting_citation_recall", 0.0)) >= min_support_recall
        and int(summary.get("errors", 0) or 0) == 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark reports.")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-em", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument("--min-support-recall", type=float, default=None)
    args = parser.parse_args(argv)

    summaries = []
    failed_thresholds = False
    for path_text in args.reports:
        path = Path(path_text)
        summary = summarize_report(json.loads(path.read_text(encoding="utf-8")))
        if _threshold_gate_requested(args):
            gate = _threshold_gate(summary["summary"], args)
            summary["threshold_gate"] = gate
            failed_thresholds = failed_thresholds or not gate["passed"]
        summaries.append(summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if failed_thresholds else 0


def _threshold_gate_requested(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (args.min_em, args.min_f1, args.min_support_recall)
    )


def _threshold_gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    minimums = {
        "exact_match": args.min_em if args.min_em is not None else 0.0,
        "average_token_f1": args.min_f1 if args.min_f1 is not None else 0.0,
        "supporting_citation_recall": (
            args.min_support_recall if args.min_support_recall is not None else 0.0
        ),
    }
    passed = passes_thresholds(
        summary,
        min_em=minimums["exact_match"],
        min_f1=minimums["average_token_f1"],
        min_support_recall=minimums["supporting_citation_recall"],
    )
    return {
        "passed": passed,
        "minimums": minimums,
        "actual": {
            "exact_match": summary.get("exact_match", 0.0),
            "average_token_f1": summary.get("average_token_f1", 0.0),
            "supporting_citation_recall": summary.get(
                "supporting_citation_recall",
                0.0,
            ),
            "errors": summary.get("errors", 0),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
