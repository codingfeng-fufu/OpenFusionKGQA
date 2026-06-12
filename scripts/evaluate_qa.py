#!/usr/bin/env python
"""Evaluate graph-grounded QA behavior against a small question set."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from graphrag_v2.config import GraphRagConfig, load_config
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID
from graphrag_v2.llm import create_chat_provider
from graphrag_v2.qa import GraphGroundedQA, LLMAnswerer, MockAnswerer


THRESHOLD_FIELDS = {
    "route_accuracy": "min_route_accuracy",
    "retrieval_hit_rate": "min_retrieval_hit_rate",
    "citation_coverage": "min_citation_coverage",
    "refusal_accuracy": "min_refusal_accuracy",
    "citation_grounding_rate": "min_citation_grounding_rate",
    "entity_recall": "min_entity_recall",
    "relationship_recall": "min_relationship_recall",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate QA answers for a KGQA index.",
    )
    parser.add_argument("--index", required=True, help="Index/artifact directory.")
    parser.add_argument(
        "--questions",
        required=True,
        help="QA evaluation questions JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the report.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional settings YAML/JSON path for Neo4j and LLM answerer config.",
    )
    parser.add_argument(
        "--strict-neo4j",
        action="store_true",
        help="Fail instead of falling back to local artifacts when metadata expects Neo4j.",
    )
    parser.add_argument(
        "--answerer",
        choices=["mock", "llm"],
        default="mock",
        help="Answer generation provider. Defaults to mock.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Report format for stdout and --output.",
    )
    _add_threshold_arguments(parser)
    args = parser.parse_args(argv)

    try:
        thresholds = _thresholds_from_args(args)
    except ValueError as exc:
        print(f"QA evaluation failed: {exc}", file=sys.stderr)
        return 2

    report = evaluate(
        index_path=Path(args.index),
        questions_path=Path(args.questions),
        config_path=Path(args.config) if args.config else None,
        strict_neo4j=args.strict_neo4j,
        answerer_name=args.answerer,
    )
    if thresholds:
        report["thresholds"] = _evaluate_thresholds(report["summary"], thresholds)
        report["passed"] = report["passed"] and all(
            item["passed"] for item in report["thresholds"].values()
        )
    payload = (
        _format_markdown(report)
        if args.format == "markdown"
        else json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def _add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    for metric, dest in THRESHOLD_FIELDS.items():
        parser.add_argument(
            f"--min-{metric.replace('_', '-')}",
            dest=dest,
            type=float,
            default=None,
            help=f"Minimum required {metric} for a passing report.",
        )


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    thresholds = {}
    for metric, dest in THRESHOLD_FIELDS.items():
        value = getattr(args, dest)
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"Threshold for {metric} must be >= 0.")
        thresholds[metric] = value
    return thresholds


def _evaluate_thresholds(
    summary: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    return {
        metric: {
            "minimum": minimum,
            "actual": summary.get(metric),
            "passed": float(summary.get(metric, 0.0)) >= minimum,
        }
        for metric, minimum in thresholds.items()
    }


def evaluate(
    index_path: Path,
    questions_path: Path,
    config_path: Path | None = None,
    strict_neo4j: bool = False,
    answerer_name: str = "mock",
) -> dict[str, Any]:
    config = load_config(config_path) if config_path else GraphRagConfig()
    qa = _create_qa(
        index_path=index_path,
        config=config,
        strict_neo4j=strict_neo4j,
        answerer_name=answerer_name,
    )
    cases = []
    for spec in _read_jsonl(questions_path):
        cases.append(_evaluate_case(qa, spec))

    total = len(cases)
    passed = sum(1 for case in cases if case["passed"])
    route_cases = [case for case in cases if case["checks"]["route"]["expected"] is not None]
    refusal_cases = [
        case for case in cases if case["checks"]["refusal"]["expected"] is not None
    ]
    citation_cases = [
        case
        for case in cases
        if case["checks"]["citations"]["eligible"]
    ]
    cited_cases = [case for case in cases if case["citations"]]
    retrieval_required = sum(
        case["checks"]["retrieval"]["required"]
        for case in cases
    )
    retrieval_hits = sum(case["checks"]["retrieval"]["matched"] for case in cases)
    average_latency = (
        sum(case["latency_seconds"] for case in cases) / total if total else 0.0
    )

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "retrieval_hit_rate": _ratio(retrieval_hits, retrieval_required),
        "citation_coverage": _ratio(
            sum(1 for case in citation_cases if case["checks"]["citations"]["passed"]),
            len(citation_cases),
        ),
        "citation_grounding_rate": _ratio(
            sum(
                1
                for case in cited_cases
                if case["checks"]["citation_grounding"]["passed"]
            ),
            len(cited_cases),
        ),
        "refusal_accuracy": _ratio(
            sum(1 for case in refusal_cases if case["checks"]["refusal"]["passed"]),
            len(refusal_cases),
        ),
        "route_accuracy": _ratio(
            sum(1 for case in route_cases if case["checks"]["route"]["passed"]),
            len(route_cases),
        ),
        "entity_recall": _ratio(
            _sum_check(cases, "entity_ranking", "matched"),
            _sum_check(cases, "entity_ranking", "required"),
        ),
        "relationship_recall": _ratio(
            _sum_check(cases, "relationship_ranking", "matched"),
            _sum_check(cases, "relationship_ranking", "required"),
        ),
        "entity_mrr": _mean_check(cases, "entity_ranking", "mrr"),
        "relationship_mrr": _mean_check(cases, "relationship_ranking", "mrr"),
        "average_required_entity_rank": _mean_check(
            cases,
            "entity_ranking",
            "average_rank",
        ),
        "average_required_relationship_rank": _mean_check(
            cases,
            "relationship_ranking",
            "average_rank",
        ),
        "average_latency_seconds": round(average_latency, 6),
    }
    return {
        "passed": passed == total,
        "runtime": {
            "data_source_provider": qa.data_source.provider,
            "answerer": answerer_name,
            "strict_neo4j": strict_neo4j,
            "config_path": str(config_path) if config_path is not None else None,
        },
        "summary": summary,
        "cases": cases,
    }


def _create_qa(
    *,
    index_path: Path,
    config: GraphRagConfig,
    strict_neo4j: bool,
    answerer_name: str,
) -> GraphGroundedQA:
    return GraphGroundedQA.from_index(
        index_path,
        graph_store_config=config.graph_store,
        answerer=_create_answerer(answerer_name, config),
        allow_neo4j_fallback=not strict_neo4j,
    )


def _create_answerer(
    answerer_name: str,
    config: GraphRagConfig,
) -> MockAnswerer | LLMAnswerer:
    if answerer_name == "mock":
        return MockAnswerer()
    if answerer_name == "llm":
        model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
        llm_client = create_chat_provider(
            provider=config.extraction.llm_provider,
            model_config=model_config,
            require_real=True,
        )
        return LLMAnswerer(llm_client=llm_client)
    raise ValueError(f"Unsupported answerer: {answerer_name}")


def _evaluate_case(qa: GraphGroundedQA, spec: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    result = qa.ask(str(spec["question"]))
    latency = round(time.perf_counter() - started_at, 6)
    result_payload = result.to_dict()

    route_check = _check_route(result_payload, spec)
    refusal_check = _check_refusal(result_payload, spec)
    entity_check = _check_entities(result_payload, spec)
    relationship_check = _check_relationships(result_payload, spec)
    entity_ranking_check = _check_entity_ranking(result_payload, spec)
    relationship_ranking_check = _check_relationship_ranking(result_payload, spec)
    citation_check = _check_citations(result_payload, spec)
    citation_grounding_check = _check_citation_grounding(result_payload)
    answer_terms_check = _check_answer_terms(result_payload, spec)
    retrieval_check = {
        "passed": entity_check["passed"] and relationship_check["passed"],
        "required": entity_check["required"] + relationship_check["required"],
        "matched": entity_check["matched"] + relationship_check["matched"],
        "missing_entities": entity_check["missing"],
        "missing_relationships": relationship_check["missing"],
    }
    checks = {
        "route": route_check,
        "refusal": refusal_check,
        "entities": entity_check,
        "relationships": relationship_check,
        "entity_ranking": entity_ranking_check,
        "relationship_ranking": relationship_ranking_check,
        "retrieval": retrieval_check,
        "citations": citation_check,
        "citation_grounding": citation_grounding_check,
        "answer_terms": answer_terms_check,
    }
    passed = all(check["passed"] for check in checks.values())

    return {
        "id": str(spec.get("id", "")),
        "type": str(spec.get("type", "")),
        "question": str(spec["question"]),
        "passed": passed,
        "latency_seconds": latency,
        "route": result.route,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "citations": result.citations,
        "answer": result.answer,
        "checks": checks,
    }


def _check_route(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec.get("expected_route")
    actual = result.get("route")
    return {
        "passed": expected is None or actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _check_refusal(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec.get("expected_refused")
    actual = bool(result.get("refused"))
    return {
        "passed": expected is None or actual == bool(expected),
        "expected": expected,
        "actual": actual,
        "refusal_reason": result.get("refusal_reason"),
    }


def _check_entities(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_entities", []) or []
    actual_keys = _actual_entity_keys(result)
    answer_key = _text_key(str(result.get("answer", "")))
    missing = [
        _display_name(item)
        for item in required
        if not (_alias_keys(item, _text_key) & actual_keys)
        and not any(alias in answer_key for alias in _alias_keys(item, _text_key))
    ]
    return {
        "passed": not missing,
        "required": len(required),
        "matched": len(required) - len(missing),
        "missing": missing,
    }


def _check_relationships(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_relationships", []) or []
    actual = _actual_relationships(result)
    missing = [
        _display_relationship(item)
        for item in required
        if not _matches_relationship(item, actual)
    ]
    return {
        "passed": not missing,
        "required": len(required),
        "matched": len(required) - len(missing),
        "missing": missing,
    }


def _check_entity_ranking(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_entities", []) or []
    ranked = _ranked_entity_keys(result)
    ranks: dict[str, int] = {}
    missing = []
    reciprocal_sum = 0.0
    for item in required:
        display = _display_name(item)
        aliases = _alias_keys(item, _text_key)
        rank = _first_matching_rank(aliases, ranked)
        if rank is None:
            missing.append(display)
            continue
        ranks[display] = rank
        reciprocal_sum += 1.0 / rank
    matched = len(required) - len(missing)
    return {
        "passed": not missing,
        "required": len(required),
        "matched": matched,
        "missing": missing,
        "ranks": ranks,
        "mrr": round(reciprocal_sum / len(required), 4) if required else 1.0,
        "average_rank": round(sum(ranks.values()) / matched, 4) if matched else None,
    }


def _check_relationship_ranking(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_relationships", []) or []
    actual = _actual_relationships(result)
    ranks = []
    missing = []
    reciprocal_sum = 0.0
    for item in required:
        rank = _relationship_rank(item, actual)
        display = _display_relationship(item)
        if rank is None:
            missing.append(display)
            continue
        ranked_display = {**display, "rank": rank}
        ranks.append(ranked_display)
        reciprocal_sum += 1.0 / rank
    matched = len(required) - len(missing)
    return {
        "passed": not missing,
        "required": len(required),
        "matched": matched,
        "missing": missing,
        "ranks": ranks,
        "mrr": round(reciprocal_sum / len(required), 4) if required else 1.0,
        "average_rank": round(sum(item["rank"] for item in ranks) / matched, 4)
        if matched
        else None,
    }


def _check_citations(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    expected_refused = spec.get("expected_refused")
    expected = [str(item) for item in spec.get("required_citations", []) or []]
    expected_sources = [
        _normalize_citation_source_spec(item)
        for item in spec.get("required_citation_sources", []) or []
    ]
    actual = [str(item) for item in result.get("citations", []) or []]
    eligible = bool(expected or expected_sources) or expected_refused is not True
    missing_ids = [item for item in expected if item not in actual]
    missing_sources = [
        item
        for item in expected_sources
        if not _matches_citation_source(result, item)
    ]
    if expected or expected_sources:
        missing = [
            *missing_ids,
            *[_display_citation_source(item) for item in missing_sources],
        ]
        passed = not missing
    elif expected_refused is True:
        missing = []
        passed = True
    else:
        missing = ["<any citation>"] if not actual else []
        passed = bool(actual)
    return {
        "passed": passed,
        "eligible": eligible,
        "expected": expected,
        "expected_sources": expected_sources,
        "actual": actual,
        "missing": missing,
    }


def _check_citation_grounding(result: dict[str, Any]) -> dict[str, Any]:
    actual = [str(item) for item in result.get("citations", []) or []]
    retrieved = [
        str(item.get("chunk_id"))
        for item in result.get("text_evidence", []) or []
        if item.get("chunk_id")
    ]
    retrieved_set = set(retrieved)
    ungrounded = [citation for citation in actual if citation not in retrieved_set]
    return {
        "passed": not ungrounded,
        "actual": actual,
        "retrieved_text_chunks": retrieved,
        "uncited_or_unretrieved": ungrounded,
    }


def _normalize_citation_source_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"source_path_endswith": spec}
    if not isinstance(spec, dict):
        return {"source_path_endswith": str(spec)}
    normalized = {}
    for key in ("source_path_endswith", "source_path", "chunk_index"):
        if key in spec:
            normalized[key] = spec[key]
    return normalized


def _matches_citation_source(
    result: dict[str, Any],
    source_spec: dict[str, Any],
) -> bool:
    cited = {str(item) for item in result.get("citations", []) or []}
    if not cited:
        return False
    for text in result.get("text_evidence", []) or []:
        chunk_id = str(text.get("chunk_id", ""))
        if chunk_id not in cited:
            continue
        if _citation_source_matches_text(source_spec, text):
            return True
    return False


def _citation_source_matches_text(
    source_spec: dict[str, Any],
    text: dict[str, Any],
) -> bool:
    source_path = str(text.get("source_path", ""))
    expected_path = source_spec.get("source_path")
    if expected_path is not None and source_path != str(expected_path):
        return False
    suffix = source_spec.get("source_path_endswith")
    if suffix is not None and not source_path.endswith(str(suffix)):
        return False
    expected_chunk_index = source_spec.get("chunk_index")
    if expected_chunk_index is not None:
        try:
            actual_chunk_index = int(text.get("chunk_index"))
            expected_chunk_index = int(expected_chunk_index)
        except (TypeError, ValueError):
            return False
        if actual_chunk_index != expected_chunk_index:
            return False
    return True


def _display_citation_source(source_spec: dict[str, Any]) -> str:
    parts = []
    if "source_path" in source_spec:
        parts.append(f"source_path={source_spec['source_path']}")
    if "source_path_endswith" in source_spec:
        parts.append(f"source_path_endswith={source_spec['source_path_endswith']}")
    if "chunk_index" in source_spec:
        parts.append(f"chunk_index={source_spec['chunk_index']}")
    return "{" + ", ".join(parts) + "}"


def _check_answer_terms(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("answer_terms", []) or []
    answer_key = _text_key(str(result.get("answer", "")))
    missing = [
        _display_name(term)
        for term in required
        if not any(alias in answer_key for alias in _alias_keys(term, _text_key))
    ]
    return {
        "passed": not missing,
        "required": len(required),
        "matched": len(required) - len(missing),
        "missing": missing,
    }


def _actual_entity_keys(result: dict[str, Any]) -> set[str]:
    keys = set()
    for entity in result.get("graph_evidence", {}).get("linked_entities", []) or []:
        for value in [
            entity.get("id"),
            entity.get("name"),
            entity.get("canonical_name"),
            *(entity.get("aliases") or []),
        ]:
            if value:
                keys.add(_text_key(str(value)))
    for value in result.get("used_entities", []) or []:
        keys.add(_text_key(str(value)))
    return keys


def _ranked_entity_keys(result: dict[str, Any]) -> list[set[str]]:
    ranked = []
    for entity in result.get("graph_evidence", {}).get("linked_entities", []) or []:
        keys = set()
        for value in [
            entity.get("id"),
            entity.get("name"),
            entity.get("canonical_name"),
            *(entity.get("aliases") or []),
        ]:
            if value:
                keys.add(_text_key(str(value)))
        ranked.append(keys)
    return ranked


def _actual_relationships(result: dict[str, Any]) -> list[dict[str, str]]:
    relationships = []
    for item in result.get("graph_evidence", {}).get("relationships", []) or []:
        relationships.append(
            {
                "id": str(item.get("id", "")),
                "source": _text_key(str(item.get("source_name", ""))),
                "relation": _relation_key(str(item.get("relation", ""))),
                "target": _text_key(str(item.get("target_name", ""))),
            }
        )
    return relationships


def _matches_relationship(spec: dict[str, Any], actual: list[dict[str, str]]) -> bool:
    ids = _alias_keys(spec.get("id"), _text_key)
    sources = _alias_keys(spec.get("source"), _text_key)
    relations = _alias_keys(spec.get("relation"), _relation_key)
    targets = _alias_keys(spec.get("target"), _text_key)
    for item in actual:
        if ids and item["id"] in ids:
            return True
        if item["source"] in sources and item["relation"] in relations and item["target"] in targets:
            return True
    return False


def _relationship_rank(spec: dict[str, Any], actual: list[dict[str, str]]) -> int | None:
    ids = _alias_keys(spec.get("id"), _text_key)
    sources = _alias_keys(spec.get("source"), _text_key)
    relations = _alias_keys(spec.get("relation"), _relation_key)
    targets = _alias_keys(spec.get("target"), _text_key)
    for index, item in enumerate(actual, start=1):
        if ids and item["id"] in ids:
            return index
        if item["source"] in sources and item["relation"] in relations and item["target"] in targets:
            return index
    return None


def _first_matching_rank(required: set[str], ranked: list[set[str]]) -> int | None:
    for index, actual in enumerate(ranked, start=1):
        if required & actual:
            return index
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _alias_keys(spec: Any, normalizer) -> set[str]:
    return {normalizer(alias) for alias in _aliases(spec) if normalizer(alias)}


def _aliases(spec: Any) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        aliases = []
        for item in spec:
            aliases.extend(_aliases(item))
        return aliases
    if isinstance(spec, dict):
        aliases = []
        for key in ("id", "name", "value"):
            if spec.get(key):
                aliases.append(str(spec[key]))
        for key in ("aliases", "any_of"):
            aliases.extend(_aliases(spec.get(key)))
        return aliases
    return [str(spec)]


def _display_name(spec: Any) -> str:
    aliases = _aliases(spec)
    return aliases[0] if aliases else str(spec)


def _display_relationship(spec: dict[str, Any]) -> dict[str, str]:
    return {
        "source": _display_name(spec.get("source")),
        "relation": _display_name(spec.get("relation")),
        "target": _display_name(spec.get("target")),
    }


def _text_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _relation_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _sum_check(cases: list[dict[str, Any]], name: str, field: str) -> int:
    return sum(int(case["checks"][name][field]) for case in cases)


def _mean_check(cases: list[dict[str, Any]], name: str, field: str) -> float:
    values = [
        case["checks"][name][field]
        for case in cases
        if case["checks"][name][field] is not None
    ]
    if not values:
        return 1.0
    return round(sum(float(value) for value in values) / len(values), 4)


def _format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# QA Evaluation Report",
        "",
        f"Status: {'PASS' if report['passed'] else 'FAIL'}",
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
        "retrieval_hit_rate",
        "citation_coverage",
        "citation_grounding_rate",
        "refusal_accuracy",
        "route_accuracy",
        "entity_recall",
        "relationship_recall",
        "entity_mrr",
        "relationship_mrr",
        "average_required_entity_rank",
        "average_required_relationship_rank",
        "average_latency_seconds",
    ]:
        lines.append(f"| {_escape_markdown(key)} | {summary[key]} |")

    runtime = report.get("runtime", {})
    if runtime:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                "| Field | Value |",
                "| --- | --- |",
            ]
        )
        for key in [
            "data_source_provider",
            "answerer",
            "strict_neo4j",
            "config_path",
        ]:
            lines.append(
                f"| {_escape_markdown(key)} | "
                f"{_escape_markdown(str(runtime.get(key)))} |"
            )

    if report.get("thresholds"):
        lines.extend(
            [
                "",
                "## Thresholds",
                "",
                "| Metric | Minimum | Actual | Status |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for metric, threshold in report["thresholds"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(metric),
                        str(threshold["minimum"]),
                        str(threshold["actual"]),
                        "PASS" if threshold["passed"] else "FAIL",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| ID | Type | Status | Route | Refused | Citations | Latency (s) |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(case["id"]),
                    _escape_markdown(case["type"]),
                    status,
                    _escape_markdown(case["route"]),
                    str(case["refused"]),
                    str(len(case["citations"])),
                    str(case["latency_seconds"]),
                ]
            )
            + " |"
        )

    failed_cases = [case for case in report["cases"] if not case["passed"]]
    if failed_cases:
        lines.extend(["", "## Failures", ""])
        for case in failed_cases:
            lines.append(f"- `{_escape_backticks(case['id'])}`")
            for name, check in case["checks"].items():
                if not check["passed"]:
                    lines.append(
                        "  - "
                        f"{name}: expected={_display_json(check.get('expected'))}, "
                        f"actual={_display_json(check.get('actual'))}, "
                        f"missing={_display_json(check.get('missing'))}"
                    )
    return "\n".join(lines)


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _escape_backticks(value: Any) -> str:
    return str(value).replace("`", "\\`")


def _display_json(value: Any) -> str:
    if value is None:
        return "-"
    return _escape_markdown(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
