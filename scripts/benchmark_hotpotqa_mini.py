#!/usr/bin/env python3
"""Prepare and run a cost-controlled HotpotQA Mini benchmark."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import request

import pandas as pd

from graphrag_v2.config import load_config
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID
from graphrag_v2.llm import create_chat_provider
from graphrag_v2.qa import GraphGroundedQA, LLMAnswerer


DEFAULT_HOTPOTQA_DEV_DISTRACTOR_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/hotpotqa-mini")
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 42
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class PreparedHotpotQAMini:
    """Paths and sample metadata produced by the prepare step."""

    output_dir: Path
    docs_path: Path
    index_path: Path
    questions_path: Path
    sample_size: int
    seed: int


def prepare_hotpotqa_mini(
    *,
    input_path: Path,
    output_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    reset_output: bool = True,
) -> PreparedHotpotQAMini:
    """Convert HotpotQA examples into docs, QA specs, and gold local artifacts."""
    records = load_hotpotqa_records(input_path)
    selected = select_hotpotqa_records(records, sample_size=sample_size, seed=seed)
    if not selected:
        raise ValueError("No HotpotQA records selected.")

    docs_path = output_dir / "docs"
    index_path = output_dir / "index"
    questions_path = output_dir / "questions.jsonl"
    if reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)

    text_units: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []

    for record in selected:
        case = normalize_hotpotqa_record(record)
        doc_path = docs_path / f"{_slug(case['id'])}.md"
        doc_path.write_text(_format_case_document(case), encoding="utf-8")

        title_to_chunk: dict[str, str] = {}
        for title, sentences in case["context"]:
            chunk_id = stable_id("hotpot_chunk", case["id"], title)
            title_to_chunk[title] = chunk_id
            text = _paragraph_text(title, sentences)
            text_units.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": stable_id("hotpot_doc", case["id"]),
                    "source_path": str(doc_path),
                    "chunk_index": len(title_to_chunk) - 1,
                    "text": text,
                    "n_tokens": len(text.split()),
                    "metadata": {
                        "benchmark": "HotpotQA",
                        "hotpotqa_id": case["id"],
                        "title": title,
                        "supporting": title in case["supporting_titles"],
                    },
                }
            )
            entity_id = stable_id("hotpot_entity", case["id"], title)
            entities[entity_id] = {
                "id": entity_id,
                "name": title,
                "canonical_name": title.lower(),
                "type": "WikipediaPage",
                "description": sentences[0] if sentences else title,
                "aliases": [title, title.lower()],
                "evidence_chunk_ids": [chunk_id],
                "confidence": 1.0,
                "metadata": {
                    "benchmark": "HotpotQA",
                    "hotpotqa_id": case["id"],
                    "supporting": title in case["supporting_titles"],
                },
            }

        support_titles = [
            title for title in case["supporting_titles"] if title in title_to_chunk
        ]
        required_relationships = []
        for source, target in _support_title_pairs(support_titles):
            source_id = stable_id("hotpot_entity", case["id"], source)
            target_id = stable_id("hotpot_entity", case["id"], target)
            rel_id = stable_id("hotpot_rel", case["id"], source, target)
            evidence_chunk_ids = [title_to_chunk[source], title_to_chunk[target]]
            relationships[rel_id] = {
                "id": rel_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "source_name": source,
                "target_name": target,
                "relation": "supports_answer",
                "original_relations": ["supports_answer"],
                "description": (
                    f"{source} and {target} are HotpotQA supporting pages for "
                    f"question {case['id']}."
                ),
                "confidence": 1.0,
                "evidence_chunk_ids": evidence_chunk_ids,
                "extraction_count": 1,
                "metadata": {
                    "benchmark": "HotpotQA",
                    "hotpotqa_id": case["id"],
                    "gold_supporting_fact": True,
                },
            }
            required_relationships.append(
                {
                    "source": {"name": source},
                    "relation": {"name": "supports_answer"},
                    "target": {"name": target},
                }
            )

        required_citations = [title_to_chunk[title] for title in support_titles]
        questions.append(
            {
                "id": f"hotpotqa_{case['id']}",
                "type": f"hotpotqa_{case['type']}",
                "question": case["question"],
                "expected_route": "local",
                "expected_refused": False,
                "answer": case["answer"],
                "required_entities": [{"name": title} for title in support_titles],
                "required_relationships": required_relationships,
                "required_citations": required_citations,
                "answer_terms": [case["answer"]] if case["answer"] else [],
                "metadata": {
                    "benchmark": "HotpotQA",
                    "hotpotqa_id": case["id"],
                    "level": case["level"],
                    "supporting_titles": support_titles,
                },
            }
        )

    _write_parquet(index_path / "text_units.parquet", text_units)
    _write_parquet(index_path / "entities.parquet", list(entities.values()))
    _write_parquet(index_path / "relationships.parquet", list(relationships.values()))
    _write_parquet(index_path / "rejected_triples.parquet", [])
    _write_graph_json(index_path, list(entities.values()), list(relationships.values()))
    _write_index_metadata(
        index_path=index_path,
        input_path=input_path,
        output_dir=output_dir,
        sample_size=len(selected),
        seed=seed,
        text_unit_count=len(text_units),
        entity_count=len(entities),
        relationship_count=len(relationships),
    )
    questions_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in questions) + "\n",
        encoding="utf-8",
    )
    sample_manifest = build_sample_manifest(
        records,
        sample_sizes=[sample_size],
        seed=seed,
        source_file=input_path,
    )
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return PreparedHotpotQAMini(
        output_dir=output_dir,
        docs_path=docs_path,
        index_path=index_path,
        questions_path=questions_path,
        sample_size=len(selected),
        seed=seed,
    )


def run_hotpotqa_mini_benchmark(
    *,
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    real_llm_smoke_size: int = 0,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare HotpotQA Mini, run offline QA, and write JSON/Markdown reports."""
    prepared = prepare_hotpotqa_mini(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
    )
    questions = _read_jsonl(prepared.questions_path)
    offline_report = evaluate_hotpotqa_questions(
        index_path=prepared.index_path,
        questions=questions,
        answerer=None,
    )
    real_llm_report = run_real_llm_smoke(
        index_path=prepared.index_path,
        questions=questions,
        smoke_size=real_llm_smoke_size,
        config_path=config_path,
    )
    report = {
        "benchmark": {
            "name": "HotpotQA Mini",
            "source": "HotpotQA distractor-style records",
            "public_benchmark_url": "https://hotpotqa.github.io/",
            "sample_size": prepared.sample_size,
            "seed": prepared.seed,
            "selection": "fixed-seed sample from the provided HotpotQA JSON file",
            "cost_control": {
                "offline_gold_index": True,
                "real_llm_smoke_size": real_llm_smoke_size,
            },
        },
        "artifacts": {
            "docs_path": str(prepared.docs_path),
            "index_path": str(prepared.index_path),
            "questions_path": str(prepared.questions_path),
        },
        "offline_qa": offline_report,
        "answer_quality": offline_report["answer_quality"],
        "real_llm_smoke": real_llm_report,
    }
    _write_reports(output_dir, report)
    return report


def evaluate_hotpotqa_questions(
    *,
    index_path: Path,
    questions: list[dict[str, Any]],
    answerer: Any | None,
) -> dict[str, Any]:
    qa = GraphGroundedQA.from_index(index_path, answerer=answerer)
    cases = []
    for spec in questions:
        started_at = time.perf_counter()
        result = qa.ask(str(spec["question"]))
        latency = round(time.perf_counter() - started_at, 6)
        payload = result.to_dict()
        required_citations = [str(item) for item in spec.get("required_citations", [])]
        citations = [str(item) for item in payload.get("citations", [])]
        citation_hits = [item for item in required_citations if item in citations]
        missing_required_citations = [
            item for item in required_citations if item not in citations
        ]
        gold_answer = str(spec.get("answer", ""))
        answer = str(payload.get("answer", ""))
        token_f1 = answer_token_f1(gold_answer, answer)
        query_trace = _query_trace_from_payload(payload)
        retrieved_relationships = _as_list_of_dicts(
            query_trace.get("retrieved_relationships")
        )
        relationship_count_by_hop = query_trace.get("relationship_count_by_hop")
        if not isinstance(relationship_count_by_hop, dict):
            relationship_count_by_hop = _relationship_count_by_hop(retrieved_relationships)
        relationship_count_by_hop = {
            str(key): int(value) for key, value in relationship_count_by_hop.items()
        }
        max_retrieved_hop = _max_retrieved_hop(retrieved_relationships)
        trace_max_retrieved_hop = query_trace.get("max_retrieved_hop")
        if isinstance(trace_max_retrieved_hop, int):
            max_retrieved_hop = trace_max_retrieved_hop
        case = {
            "id": spec["id"],
            "question": spec["question"],
            "gold_answer": gold_answer,
            "answer": answer,
            "route": payload.get("route"),
            "refused": bool(payload.get("refused")),
            "latency_seconds": latency,
            "required_citations": required_citations,
            "citations": citations,
            "citation_hits": citation_hits,
            "missing_required_citations": missing_required_citations,
            "citation_recall": _ratio(len(citation_hits), len(required_citations)),
            "all_required_citations_found": len(citation_hits) == len(required_citations),
            "answer_contains_gold": _contains_answer(gold_answer, answer),
            "answer_token_f1": token_f1,
            "query_trace": query_trace,
            "linked_entities": _as_list_of_dicts(query_trace.get("linked_entities")),
            "retrieved_relationships": retrieved_relationships,
            "retrieved_relationship_hops": [
                int(relationship.get("hop") or 1)
                for relationship in retrieved_relationships
            ],
            "retrieved_text_chunks": _as_list_of_dicts(
                query_trace.get("retrieved_text_chunks")
            ),
            "adaptive_enabled": bool(query_trace.get("adaptive_enabled", True)),
            "adaptive_triggered": bool(
                query_trace.get("adaptive_triggered", max_retrieved_hop > 1)
            ),
            "matched_adaptive_cues": list(
                query_trace.get("matched_adaptive_cues")
                or _matched_adaptive_cues(str(spec["question"]))
            ),
            "hop_plan": list(query_trace.get("hop_plan") or []),
            "relationship_count_by_hop": relationship_count_by_hop,
            "max_retrieved_hop": max_retrieved_hop,
        }
        case["passed"] = (
            not case["refused"]
            and case["all_required_citations_found"]
            and (case["answer_contains_gold"] or token_f1 > 0.0)
        )
        cases.append(case)

    total = len(cases)
    summary = {
        "total": total,
        "passed": sum(1 for case in cases if case["passed"]),
        "failed": sum(1 for case in cases if not case["passed"]),
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
        "average_latency_seconds": _mean([case["latency_seconds"] for case in cases]),
    }
    return {
        "passed": summary["failed"] == 0,
        "summary": summary,
        "answer_quality": {
            "answer_contains_gold_rate": summary["answer_contains_gold_rate"],
            "average_token_f1": summary["average_token_f1"],
        },
        "cases": cases,
    }


def run_real_llm_smoke(
    *,
    index_path: Path,
    questions: list[dict[str, Any]],
    smoke_size: int,
    config_path: Path | None,
) -> dict[str, Any]:
    if smoke_size <= 0:
        return {"enabled": False, "sample_size": 0, "cases": []}
    if config_path is None:
        return {
            "enabled": True,
            "sample_size": smoke_size,
            "status": "blocked",
            "reason": "--config is required when --real-llm-smoke-size is > 0",
            "cases": [],
        }

    config = load_config(config_path)
    model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
    llm_client = create_chat_provider(
        provider=config.extraction.llm_provider,
        model_config=model_config,
        require_real=True,
    )
    answerer = LLMAnswerer(llm_client=llm_client)
    selected = questions[:smoke_size]
    try:
        report = evaluate_hotpotqa_questions(
            index_path=index_path,
            questions=selected,
            answerer=answerer,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "sample_size": len(selected),
            "status": "failed",
            "reason": f"{exc.__class__.__name__}: {exc}",
            "cases": [],
        }
    return {
        "enabled": True,
        "sample_size": len(selected),
        "status": "passed" if report["passed"] else "failed",
        "summary": report["summary"],
        "cases": report["cases"],
    }


def load_hotpotqa_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("HotpotQA JSON input must be a list of records.")
        return data
    records = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        records.append(item)
    return records


def select_hotpotqa_records(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0.")
    queryable = [
        record
        for record in records
        if _has_queryable_supporting_title(normalize_hotpotqa_record(record))
    ]
    pool = queryable or records
    if not pool:
        return []
    ordered = _deterministic_sample_order(pool, seed=seed)
    return ordered[: min(sample_size, len(ordered))]


def build_sample_manifest(
    records: list[dict[str, Any]],
    *,
    sample_sizes: list[int],
    seed: int,
    source_file: Path | None = None,
) -> dict[str, Any]:
    """Build a nested deterministic sample manifest from one seeded ordering."""
    if any(size <= 0 for size in sample_sizes):
        raise ValueError("sample_sizes must all be > 0.")
    queryable = [
        record
        for record in records
        if _has_queryable_supporting_title(normalize_hotpotqa_record(record))
    ]
    pool = queryable or records
    ordered = _deterministic_sample_order(pool, seed=seed)
    ordered_ids = [str(record.get("_id") or record.get("id") or "") for record in ordered]
    samples = {}
    for sample_size in sorted(set(sample_sizes)):
        prefix = ordered_ids[: min(sample_size, len(ordered_ids))]
        samples[f"dev{sample_size}"] = {
            "requested_size": sample_size,
            "actual_size": len(prefix),
            "ids": prefix,
        }
    return {
        "manifest_schema_version": 1,
        "benchmark": "HotpotQA",
        "source_file": str(source_file) if source_file else None,
        "seed": seed,
        "selection": "seeded_shuffle_prefix",
        "total_records": len(records),
        "queryable_records": len(queryable),
        "ordered_ids": ordered_ids,
        "samples": samples,
    }


def normalize_hotpotqa_record(record: dict[str, Any]) -> dict[str, Any]:
    case_id = str(record.get("_id") or record.get("id") or "")
    if not case_id:
        raise ValueError("HotpotQA record is missing _id.")
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    context = _normalize_context(record.get("context") or [])
    supporting_titles = []
    supporting_facts = record.get("supporting_facts") or []
    for fact in supporting_facts:
        if isinstance(fact, list) and fact:
            title = str(fact[0])
        elif isinstance(fact, dict):
            title = str(fact.get("title") or "")
        else:
            title = ""
        if title and title not in supporting_titles:
            supporting_titles.append(title)
    return {
        "id": case_id,
        "question": question,
        "answer": answer,
        "type": str(record.get("type") or "unknown"),
        "level": str(record.get("level") or "unknown"),
        "context": context,
        "supporting_titles": supporting_titles,
    }


def stable_id(prefix: str, *parts: str) -> str:
    digest = sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def answer_token_f1(gold: str, prediction: str) -> float:
    gold_tokens = _answer_tokens(gold)
    prediction_tokens = _answer_tokens(prediction)
    if not gold_tokens and not prediction_tokens:
        return 1.0
    if not gold_tokens or not prediction_tokens:
        return 0.0
    common = 0
    remaining = list(prediction_tokens)
    for token in gold_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    if not args.download:
        raise ValueError("Provide --input or opt in to --download.")
    cache_path = Path(args.cache)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(
            args.download_url,
            cache_path,
            timeout=float(args.download_timeout),
        )
    return cache_path


def download_file(url: str, output_path: Path, *, timeout: float) -> None:
    """Download a file with an explicit timeout to avoid hanging smoke runs."""
    with request.urlopen(url, timeout=timeout) as response:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a cost-controlled HotpotQA Mini benchmark.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Local HotpotQA JSON/JSONL file. Required unless --download is set.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download HotpotQA dev distractor data into --cache if --input is omitted.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_HOTPOTQA_DEV_DISTRACTOR_URL,
        help="HotpotQA download URL used only with --download.",
    )
    parser.add_argument(
        "--cache",
        default=".cache/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Local cache path used only with --download.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        help=f"Download timeout in seconds. Defaults to {DEFAULT_DOWNLOAD_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Fixed-seed sample size. Defaults to {DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed. Defaults to {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--real-llm-smoke-size",
        type=int,
        default=0,
        help="Optional real LLM answerer smoke size. Defaults to 0.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Settings file required when --real-llm-smoke-size is > 0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_path = resolve_input_path(args)
        report = run_hotpotqa_mini_benchmark(
            input_path=input_path,
            output_dir=Path(args.output),
            sample_size=args.sample_size,
            seed=args.seed,
            real_llm_smoke_size=args.real_llm_smoke_size,
            config_path=Path(args.config) if args.config else None,
        )
    except Exception as exc:
        print(f"HotpotQA Mini benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["offline_qa"]["passed"] else 1


def _normalize_context(raw_context: list[Any]) -> list[tuple[str, list[str]]]:
    context = []
    for item in raw_context:
        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0])
            sentences = [str(sentence) for sentence in (item[1] or [])]
            context.append((title, sentences))
    return context


def _has_queryable_supporting_title(case: dict[str, Any]) -> bool:
    question_key = _normalize_text(case["question"])
    return any(_normalize_text(title) in question_key for title in case["supporting_titles"])


def _deterministic_sample_order(
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: str(item.get("_id") or item.get("id") or ""))
    random.Random(seed).shuffle(ordered)
    return ordered


def _support_title_pairs(titles: list[str]) -> list[tuple[str, str]]:
    if len(titles) < 2:
        return []
    pairs = []
    for index in range(len(titles) - 1):
        pairs.append((titles[index], titles[index + 1]))
    return pairs


def _paragraph_text(title: str, sentences: list[str]) -> str:
    body = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return f"# {title}\n\n{body}\n"


def _format_case_document(case: dict[str, Any]) -> str:
    lines = [
        f"# HotpotQA Case {case['id']}",
        "",
        f"Question: {case['question']}",
        f"Answer: {case['answer']}",
        "",
    ]
    for title, sentences in case["context"]:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(sentence.strip() for sentence in sentences if sentence.strip())
        lines.append("")
    return "\n".join(lines)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_graph_json(
    index_path: Path,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    graph = {
        "created_at": _utc_now(),
        "nodes": entities,
        "edges": relationships,
        "statistics": {
            "num_nodes": len(entities),
            "num_edges": len(relationships),
        },
    }
    (index_path / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_index_metadata(
    *,
    index_path: Path,
    input_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    text_unit_count: int,
    entity_count: int,
    relationship_count: int,
) -> None:
    metadata = {
        "metadata_schema_version": 1,
        "benchmark": "HotpotQA Mini",
        "benchmark_source": "https://hotpotqa.github.io/",
        "sample_seed": seed,
        "sample_size": sample_size,
        "index_id": stable_id("hotpot_index", str(input_path), str(seed), str(sample_size)),
        "output_path": str(output_dir),
        "run_status": "succeeded",
        "run_mode": "hotpotqa-mini-gold",
        "input_path": str(input_path),
        "created_at": _utc_now(),
        "num_documents": sample_size,
        "num_text_units": text_unit_count,
        "num_entities": entity_count,
        "num_relationships": relationship_count,
        "num_rejected_triples": 0,
        "extractor": "hotpotqa_gold_adapter",
        "graph_store_provider": "json",
        "graph_store_written": True,
        "graph_store_health_status": "ready",
    }
    (index_path / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _query_trace_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    query_trace = metadata.get("query_trace")
    if not isinstance(query_trace, dict):
        return {}
    return query_trace


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _relationship_count_by_hop(
    relationships: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relationship in relationships:
        hop = str(int(relationship.get("hop") or 1))
        counts[hop] = counts.get(hop, 0) + 1
    return counts


def _max_retrieved_hop(relationships: list[dict[str, Any]]) -> int:
    if not relationships:
        return 0
    return max(int(relationship.get("hop") or 1) for relationship in relationships)


_ADAPTIVE_QUERY_CUES = (
    "same",
    "older",
    "younger",
    "born first",
    "born later",
    "died first",
    "died later",
    "released first",
    "came out first",
    "more recently",
    "director of",
    "founder of",
    "father of",
    "mother of",
    "husband of",
    "wife of",
    "composer of",
)


def _matched_adaptive_cues(question: str) -> list[str]:
    normalized = " ".join(question.lower().split())
    return [cue for cue in _ADAPTIVE_QUERY_CUES if cue in normalized]


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "hotpotqa-mini-report.json"
    markdown_path = output_dir / "hotpotqa-mini-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(format_markdown_report(report), encoding="utf-8")


def format_markdown_report(report: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    offline = report["offline_qa"]
    summary = offline["summary"]
    real_llm = report["real_llm_smoke"]
    lines = [
        "# HotpotQA Mini Benchmark Report",
        "",
        "## Benchmark",
        "",
        f"- Source: HotpotQA public benchmark ({benchmark['public_benchmark_url']})",
        f"- Sample: `{benchmark['sample_size']}` questions, fixed seed `{benchmark['seed']}`",
        "- Mode: offline gold index for retrieval/QA path evaluation",
        "- Cost control: real LLM calls are disabled unless explicitly requested",
        "",
        "## Offline QA Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "total",
        "passed",
        "failed",
        "citation_coverage",
        "supporting_citation_recall",
        "answer_contains_gold_rate",
        "average_token_f1",
        "average_latency_seconds",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(["", "## Real LLM Smoke", ""])
    if not real_llm.get("enabled"):
        lines.append("Real LLM smoke: disabled")
    else:
        lines.append(f"Real LLM smoke: {real_llm.get('status')} ({real_llm.get('sample_size')} cases)")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Docs: `{report['artifacts']['docs_path']}`",
            f"- Index: `{report['artifacts']['index_path']}`",
            f"- Questions: `{report['artifacts']['questions_path']}`",
            "",
            "## Case Results",
            "",
            "| ID | Passed | Citation Recall | Answer F1 | Latency (s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in offline["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["id"]),
                    str(case["passed"]),
                    str(case["citation_recall"]),
                    str(case["answer_token_f1"]),
                    str(case["latency_seconds"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contains_answer(gold: str, answer: str) -> bool:
    if not gold:
        return True
    return _normalize_text(gold) in _normalize_text(answer)


def _answer_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split(" ") if token]


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return slug or stable_id("hotpot_case", value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
