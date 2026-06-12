#!/usr/bin/env python3
"""Prepare support-only QA datasets and run real-extraction controls.

The adapter keeps only gold supporting pages, writes a deterministic gold index
for retrieval/QA diagnostics, and can optionally build a real LLM-extracted
index from the same support-only documents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_hotpotqa_mini import evaluate_hotpotqa_questions  # noqa: E402
from graphrag_v2.config import load_config  # noqa: E402
from graphrag_v2.indexing import index_fusion_only  # noqa: E402
from graphrag_v2.qa import LLMAnswerer  # noqa: E402
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID  # noqa: E402
from graphrag_v2.llm import create_chat_provider  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("artifacts/support-real-extract")
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PreparedSupportOnlyDataset:
    """Paths produced by support-only dataset preparation."""

    output_dir: Path
    docs_path: Path
    questions_path: Path
    gold_index_path: Path
    sample_manifest_path: Path
    sample_size: int
    seed: int
    benchmark_name: str


def prepare_support_only_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    benchmark_name: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    reset_output: bool = True,
) -> PreparedSupportOnlyDataset:
    """Write docs, QA specs, and a gold graph index from support-only pages."""
    records = load_records(input_path)
    selected = select_records(records, sample_size=sample_size, seed=seed)
    if not selected:
        raise ValueError("No support-only records selected.")

    adapter_dir = output_dir / "adapter"
    docs_path = adapter_dir / "docs"
    questions_path = adapter_dir / "questions.jsonl"
    gold_index_path = adapter_dir / "index"
    sample_manifest_path = adapter_dir / "sample_manifest.json"
    if reset_output and adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    gold_index_path.mkdir(parents=True, exist_ok=True)

    text_units: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []

    benchmark_key = _benchmark_key(benchmark_name)
    for record in selected:
        case = normalize_record(record, benchmark_name=benchmark_name)
        doc_path = docs_path / f"{_slug(case['id'])}.md"
        doc_path.write_text(_format_case_document(case), encoding="utf-8")

        title_to_chunk: dict[str, str] = {}
        for title in case["supporting_titles"]:
            sentences = case["context_by_title"].get(title, [])
            chunk_id = stable_id(f"{benchmark_key}_chunk", case["id"], title)
            title_to_chunk[title] = chunk_id
            text = _paragraph_text(title, sentences)
            text_units.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": stable_id(f"{benchmark_key}_doc", case["id"]),
                    "source_path": str(doc_path),
                    "chunk_index": len(title_to_chunk) - 1,
                    "text": text,
                    "n_tokens": len(text.split()),
                    "metadata": {
                        "benchmark": benchmark_name,
                        "record_id": case["id"],
                        "title": title,
                        "supporting": True,
                    },
                }
            )
            entity_id = stable_id(f"{benchmark_key}_entity", case["id"], title)
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
                    "benchmark": benchmark_name,
                    "record_id": case["id"],
                    "supporting": True,
                },
            }

        required_relationships = []
        for source, target in _support_title_pairs(case["supporting_titles"]):
            source_id = stable_id(f"{benchmark_key}_entity", case["id"], source)
            target_id = stable_id(f"{benchmark_key}_entity", case["id"], target)
            rel_id = stable_id(f"{benchmark_key}_rel", case["id"], source, target)
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
                    f"{source} and {target} are {benchmark_name} supporting pages "
                    f"for question {case['id']}."
                ),
                "confidence": 1.0,
                "evidence_chunk_ids": evidence_chunk_ids,
                "extraction_count": 1,
                "metadata": {
                    "benchmark": benchmark_name,
                    "record_id": case["id"],
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

        required_citations = [
            title_to_chunk[title]
            for title in case["supporting_titles"]
            if title in title_to_chunk
        ]
        questions.append(
            {
                "id": f"{benchmark_key}_{case['id']}",
                "type": f"{benchmark_key}_{case['type']}",
                "question": case["question"],
                "expected_route": "local",
                "expected_refused": False,
                "answer": case["answer"],
                "required_entities": [
                    {"name": title} for title in case["supporting_titles"]
                ],
                "required_relationships": required_relationships,
                "required_citations": required_citations,
                "answer_terms": [case["answer"]] if case["answer"] else [],
                "metadata": {
                    "benchmark": benchmark_name,
                    "record_id": case["id"],
                    "level": case["level"],
                    "supporting_titles": case["supporting_titles"],
                },
            }
        )

    _write_parquet(gold_index_path / "text_units.parquet", text_units)
    _write_parquet(gold_index_path / "entities.parquet", list(entities.values()))
    _write_parquet(gold_index_path / "relationships.parquet", list(relationships.values()))
    _write_parquet(gold_index_path / "rejected_triples.parquet", [])
    _write_graph_json(gold_index_path, list(entities.values()), list(relationships.values()))
    _write_index_metadata(
        index_path=gold_index_path,
        input_path=input_path,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
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
        benchmark_name=benchmark_name,
        sample_sizes=sorted({5, 10, sample_size}),
        seed=seed,
        source_file=input_path,
    )
    sample_manifest_path.write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return PreparedSupportOnlyDataset(
        output_dir=output_dir,
        docs_path=docs_path,
        questions_path=questions_path,
        gold_index_path=gold_index_path,
        sample_manifest_path=sample_manifest_path,
        sample_size=len(selected),
        seed=seed,
        benchmark_name=benchmark_name,
    )


def run_support_real_extract_benchmark(
    *,
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    benchmark_name: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    config_path: Path | None = None,
    answerer_name: str = "llm",
    run_real_extraction: bool = True,
    reset_output: bool = True,
) -> dict[str, Any]:
    """Prepare support-only data, optionally index with real extraction, then QA."""
    started_at = time.perf_counter()
    prepared = prepare_support_only_dataset(
        input_path=input_path,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
        sample_size=sample_size,
        seed=seed,
        reset_output=reset_output,
    )
    questions = _read_jsonl(prepared.questions_path)
    index_path = output_dir / "index"
    real_index_metadata: dict[str, Any] | None = None
    if run_real_extraction:
        if config_path is None:
            raise ValueError("--config is required unless --skip-real-extraction is set.")
        config = load_config(config_path)
        config.extraction.fail_on_invalid_chunk = False
        real_index_metadata = asyncio.run(
            index_fusion_only(
                input_path=prepared.docs_path,
                output_path=index_path,
                config=config,
                extractor_name="llm",
                mode="support-real-extraction",
                graph_store_provider="json",
            )
        )
    else:
        index_path = prepared.gold_index_path

    if run_real_extraction:
        questions = remap_required_citations_to_real_index_chunks(
            questions,
            index_path,
        )
    answerer = _create_answerer(answerer_name, config_path)
    qa_report = evaluate_hotpotqa_questions(
        index_path=index_path,
        questions=questions,
        answerer=answerer,
    )
    report = {
        "benchmark": {
            "name": f"{benchmark_name} support-only real LLM extraction graph benchmark",
            "real_extraction": bool(run_real_extraction),
            "index_mode": (
                "real_llm_extracted_graph_support_only"
                if run_real_extraction
                else "gold_support_only"
            ),
            "sample_size": prepared.sample_size,
            "seed": seed,
            "source_file": str(input_path),
            "adapter_docs_path": str(prepared.docs_path),
            "questions_path": str(prepared.questions_path),
            "gold_index_path": str(prepared.gold_index_path),
            "index_path": str(index_path),
            "sample_manifest_path": str(prepared.sample_manifest_path),
            "fail_on_invalid_chunk": False if run_real_extraction else None,
        },
        "summary": qa_report["summary"],
        "real_index_metadata": real_index_metadata or {},
        "cases": qa_report["cases"],
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
    }
    _write_reports(output_dir, benchmark_name, report)
    return report


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON array or JSONL benchmark records."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("Benchmark JSON input must be a list of records.")
        return [item for item in data if isinstance(item, dict)]
    records = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def select_records(
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
        if _has_supporting_context(normalize_record(record, benchmark_name="Generic"))
    ]
    pool = queryable or records
    ordered = _deterministic_sample_order(pool, seed=seed)
    return ordered[: min(sample_size, len(ordered))]


def build_sample_manifest(
    records: list[dict[str, Any]],
    *,
    benchmark_name: str,
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
        if _has_supporting_context(normalize_record(record, benchmark_name=benchmark_name))
    ]
    pool = queryable or records
    ordered = _deterministic_sample_order(pool, seed=seed)
    ordered_ids = [_record_id(record) for record in ordered]
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
        "benchmark": benchmark_name,
        "source_file": str(source_file) if source_file else None,
        "seed": seed,
        "selection": "seeded_shuffle_prefix",
        "total_records": len(records),
        "queryable_records": len(queryable),
        "ordered_ids": ordered_ids,
        "samples": samples,
    }


def normalize_record(record: dict[str, Any], *, benchmark_name: str) -> dict[str, Any]:
    case_id = _record_id(record)
    if not case_id:
        raise ValueError("Benchmark record is missing _id/id.")
    context = _normalize_context(record.get("context") or [])
    context_by_title = {title: sentences for title, sentences in context}
    supporting_titles = _supporting_titles(record)
    supporting_titles = [title for title in supporting_titles if title in context_by_title]
    return {
        "id": case_id,
        "question": str(record.get("question") or "").strip(),
        "answer": str(record.get("answer") or "").strip(),
        "type": str(record.get("type") or "unknown"),
        "level": str(record.get("level") or "unknown"),
        "benchmark": benchmark_name,
        "context": context,
        "context_by_title": context_by_title,
        "supporting_titles": supporting_titles,
    }


def remap_required_citations_to_real_index_chunks(
    questions: list[dict[str, Any]],
    index_path: Path,
) -> list[dict[str, Any]]:
    """Map gold support citation ids onto real extraction text-unit chunk ids."""
    text_units_path = index_path / "text_units.parquet"
    if not text_units_path.exists():
        return [dict(question) for question in questions]
    frame = pd.read_parquet(text_units_path)
    rows = frame.to_dict(orient="records")
    rows_by_record_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        record_id = _record_id_from_source_path(str(row.get("source_path") or ""))
        if not record_id:
            continue
        rows_by_record_id.setdefault(record_id, []).append(row)

    remapped_questions = []
    for question in questions:
        remapped = dict(question)
        metadata = dict(remapped.get("metadata") or {})
        record_id = str(metadata.get("record_id") or "")
        candidate_rows = rows_by_record_id.get(record_id, [])
        required_mapping: dict[str, list[str]] = {}
        required_chunk_ids: list[str] = []
        for title in [str(item) for item in metadata.get("supporting_titles", [])]:
            title_chunk_ids = _matching_real_chunk_ids(title, candidate_rows)
            required_mapping[title] = title_chunk_ids
            required_chunk_ids.extend(title_chunk_ids)
        if required_chunk_ids:
            metadata["gold_required_citations"] = [
                str(item) for item in remapped.get("required_citations", [])
            ]
            metadata["required_citation_mapping"] = required_mapping
            remapped["required_citations"] = _unique(required_chunk_ids)
            remapped["metadata"] = metadata
        remapped_questions.append(remapped)
    return remapped_questions


def stable_id(prefix: str, *parts: str) -> str:
    digest = sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run support-only real-extraction QA benchmark.",
    )
    parser.add_argument("--input", required=True, help="Benchmark JSON/JSONL path.")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark label, for example HotpotQA or 2Wiki.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", default=None, help="Settings YAML for real LLM.")
    parser.add_argument(
        "--answerer",
        choices=["mock", "llm"],
        default="llm",
        help="QA answerer used after indexing.",
    )
    parser.add_argument(
        "--skip-real-extraction",
        action="store_true",
        help="Evaluate the gold support-only index instead of indexing with real LLM extraction.",
    )
    parser.add_argument("--no-reset-output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_support_real_extract_benchmark(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            benchmark_name=args.benchmark,
            sample_size=args.sample_size,
            seed=args.seed,
            config_path=Path(args.config) if args.config else None,
            answerer_name=args.answerer,
            run_real_extraction=not args.skip_real_extraction,
            reset_output=not args.no_reset_output,
        )
    except Exception as exc:
        print(f"Support real-extraction benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _create_answerer(answerer_name: str, config_path: Path | None):
    if answerer_name == "mock":
        return None
    if config_path is None:
        raise ValueError("--config is required with --answerer llm")
    config = load_config(config_path)
    model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
    llm_client = create_chat_provider(
        provider=config.extraction.llm_provider,
        model_config=model_config,
        require_real=True,
    )
    return LLMAnswerer(llm_client=llm_client)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("_id") or record.get("id") or "")


def _record_id_from_source_path(source_path: str) -> str:
    if not source_path:
        return ""
    return Path(source_path).stem


def _normalize_context(raw_context: Any) -> list[tuple[str, list[str]]]:
    context = []
    if isinstance(raw_context, dict):
        titles = raw_context.get("title") or raw_context.get("titles") or []
        sentences_by_title = (
            raw_context.get("sentences")
            or raw_context.get("sentence")
            or raw_context.get("text")
            or []
        )
        if isinstance(titles, list) and isinstance(sentences_by_title, list):
            for title, sentences in zip(titles, sentences_by_title, strict=False):
                context.append((str(title), _as_sentence_list(sentences)))
        return context
    for item in raw_context:
        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0])
            sentences = _as_sentence_list(item[1])
            context.append((title, sentences))
    return context


def _supporting_titles(record: dict[str, Any]) -> list[str]:
    titles = []
    supporting_facts = record.get("supporting_facts") or []
    if isinstance(supporting_facts, dict):
        supporting_facts = supporting_facts.get("title") or []
    for fact in supporting_facts:
        if isinstance(fact, list) and fact:
            title = str(fact[0])
        elif isinstance(fact, dict):
            title = str(fact.get("title") or "")
        else:
            title = str(fact)
        if title and title not in titles:
            titles.append(title)
    return titles


def _as_sentence_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(sentence) for sentence in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _has_supporting_context(case: dict[str, Any]) -> bool:
    return bool(case["question"] and case["answer"] and case["supporting_titles"])


def _deterministic_sample_order(
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=_record_id)
    random.Random(seed).shuffle(ordered)
    return ordered


def _support_title_pairs(titles: list[str]) -> list[tuple[str, str]]:
    return [(titles[index], titles[index + 1]) for index in range(len(titles) - 1)]


def _paragraph_text(title: str, sentences: list[str]) -> str:
    body = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return f"# {title}\n\n{body}\n"


def _matching_real_chunk_ids(
    title: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    normalized_title = _normalize_match_text(title)
    matches = []
    fallback_matches = []
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        text = str(row.get("text") or "")
        normalized_text = _normalize_match_text(text)
        if f"## {title}" in text or f"# {title}" in text:
            matches.append(chunk_id)
        elif normalized_title and normalized_title in normalized_text:
            fallback_matches.append(chunk_id)
    return _unique(matches or fallback_matches)


def _normalize_match_text(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _format_case_document(case: dict[str, Any]) -> str:
    lines = [
        f"# {case['benchmark']} Case {case['id']}",
        "",
        f"Question: {case['question']}",
        f"Answer: {case['answer']}",
        "",
    ]
    for title in case["supporting_titles"]:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(
            sentence.strip()
            for sentence in case["context_by_title"].get(title, [])
            if sentence.strip()
        )
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
    benchmark_name: str,
    sample_size: int,
    seed: int,
    text_unit_count: int,
    entity_count: int,
    relationship_count: int,
) -> None:
    metadata = {
        "metadata_schema_version": 1,
        "benchmark": f"{benchmark_name} Support-Only",
        "sample_seed": seed,
        "sample_size": sample_size,
        "index_id": stable_id("support_index", benchmark_name, str(input_path), str(seed), str(sample_size)),
        "output_path": str(output_dir),
        "run_status": "succeeded",
        "run_mode": "support-only-gold",
        "input_path": str(input_path),
        "created_at": _utc_now(),
        "num_documents": sample_size,
        "num_text_units": text_unit_count,
        "num_entities": entity_count,
        "num_relationships": relationship_count,
        "num_rejected_triples": 0,
        "extractor": "support_only_gold_adapter",
        "graph_store_provider": "json",
        "graph_store_written": True,
        "graph_store_health_status": "ready",
    }
    (index_path / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_reports(output_dir: Path, benchmark_name: str, report: dict[str, Any]) -> None:
    benchmark_key = _benchmark_key(benchmark_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{benchmark_key}-real-extract-qa-report.json"
    md_path = output_dir / f"{benchmark_key}-real-extract-qa-report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_format_markdown_report(report), encoding="utf-8")


def _format_markdown_report(report: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    summary = report["summary"]
    lines = [
        f"# {benchmark['name']}",
        "",
        f"- Source: `{benchmark['source_file']}`",
        f"- Sample size: `{benchmark['sample_size']}`",
        f"- Seed: `{benchmark['seed']}`",
        f"- Index mode: `{benchmark['index_mode']}`",
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
        "citation_coverage",
        "supporting_citation_recall",
        "answer_contains_gold_rate",
        "average_token_f1",
        "average_latency_seconds",
    ]:
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    return "\n".join(lines) + "\n"


def _benchmark_key(benchmark_name: str) -> str:
    normalized = benchmark_name.strip().lower()
    if normalized in {"2wiki", "2wikimultihopqa", "2wikimultihop"}:
        return "2wiki"
    if normalized in {"hotpotqa", "hotpot"}:
        return "hotpotqa"
    return "".join(char if char.isalnum() else "_" for char in normalized).strip("_")


def _slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return slug.strip("_") or "case"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
