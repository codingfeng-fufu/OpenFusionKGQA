"""Tests for the graph-grounded QA evaluation utility."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from graphrag_v2.config import GraphRagConfig
from graphrag_v2.indexing import index_fusion_only


def test_evaluate_qa_passes_offline_fixture(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_graphrag",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
                "answer_terms": ["GraphRAG"],
            },
            {
                "id": "no_answer",
                "question": "不存在的概念是什么？",
                "expected_route": "local",
                "expected_refused": True,
            },
        ],
    )

    result = _run_eval(index_path, questions_path)

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["retrieval_hit_rate"] == 1.0
    assert payload["summary"]["citation_coverage"] == 1.0
    assert payload["summary"]["refusal_accuracy"] == 1.0
    assert payload["runtime"]["data_source_provider"] == "json"
    assert payload["runtime"]["answerer"] == "mock"
    assert payload["runtime"]["strict_neo4j"] is False


def test_evaluate_qa_accepts_config_strict_neo4j_and_answerer_flags(temp_dir: Path):
    index_path = _write_index(temp_dir)
    config_path = temp_dir / "settings.yaml"
    config_path.write_text(
        """
graph_store:
  provider: "json"
  fallback: "json"
""",
        encoding="utf-8",
    )
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_graphrag",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
            }
        ],
    )

    result = _run_eval(
        index_path,
        questions_path,
        config_path=config_path,
        strict_neo4j=True,
        answerer="mock",
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["runtime"] == {
        "data_source_provider": "json",
        "answerer": "mock",
        "strict_neo4j": True,
        "config_path": str(config_path),
    }


def test_evaluate_qa_fails_for_missing_required_citation(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "missing_citation",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
                "required_citations": ["missing_chunk"],
            }
        ],
    )

    result = _run_eval(index_path, questions_path)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["summary"]["citation_coverage"] == 0.0
    assert payload["cases"][0]["checks"]["citations"]["passed"] is False
    assert payload["cases"][0]["checks"]["citations"]["missing"] == ["missing_chunk"]


def test_citation_check_matches_required_source_spec_without_stable_chunk_id():
    evaluate_qa = _load_eval_module()
    result = {
        "citations": ["chunk_runtime_hash"],
        "text_evidence": [
            {
                "chunk_id": "chunk_runtime_hash",
                "source_path": "/tmp/workspace/examples/docs/neo4j.txt",
                "chunk_index": 0,
            }
        ],
    }
    spec = {
        "expected_refused": False,
        "required_citation_sources": [
            {
                "source_path_endswith": "examples/docs/neo4j.txt",
                "chunk_index": 0,
            }
        ],
    }

    check = evaluate_qa._check_citations(result, spec)

    assert check["passed"] is True
    assert check["expected_sources"] == [
        {
            "source_path_endswith": "examples/docs/neo4j.txt",
            "chunk_index": 0,
        }
    ]
    assert check["missing"] == []


def test_evaluate_qa_fails_for_wrong_route(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "wrong_route",
                "question": "GraphRAG 是什么？",
                "expected_route": "global",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
            }
        ],
    )

    result = _run_eval(index_path, questions_path)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["summary"]["route_accuracy"] == 0.0
    assert payload["cases"][0]["checks"]["route"] == {
        "passed": False,
        "expected": "global",
        "actual": "local",
    }


def test_evaluate_qa_reports_entity_and_relationship_ranking_metrics(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "ranked_relationship",
                "question": "Neo4j 和 Graph Database 的关系是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "Neo4j"}, {"name": "Graph Database"}],
                "required_relationships": [
                    {
                        "source": {"name": "Neo4j"},
                        "relation": {"name": "is_a"},
                        "target": {"name": "Graph Database"},
                    }
                ],
            }
        ],
    )

    result = _run_eval(index_path, questions_path)

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["summary"]["entity_recall"] == 1.0
    assert payload["summary"]["relationship_recall"] == 1.0
    assert payload["summary"]["entity_mrr"] > 0
    assert payload["summary"]["relationship_mrr"] > 0
    case = payload["cases"][0]
    assert case["checks"]["entity_ranking"]["matched"] == 2
    assert case["checks"]["relationship_ranking"]["matched"] == 1
    assert case["checks"]["entity_ranking"]["ranks"]["Neo4j"] >= 1
    assert case["checks"]["relationship_ranking"]["ranks"][0]["rank"] >= 1


def test_evaluate_qa_reports_grounded_citations_for_retrieved_text_evidence(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "grounded_citation",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
            }
        ],
    )

    result = _run_eval(index_path, questions_path)

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["summary"]["citation_grounding_rate"] == 1.0
    grounding = payload["cases"][0]["checks"]["citation_grounding"]
    assert grounding["passed"] is True
    assert grounding["uncited_or_unretrieved"] == []


def test_citation_grounding_check_reports_unretrieved_citation():
    evaluate_qa = _load_eval_module()

    result = {
        "citations": ["chunk_missing"],
        "text_evidence": [{"chunk_id": "chunk_present"}],
    }

    check = evaluate_qa._check_citation_grounding(result)

    assert check == {
        "passed": False,
        "actual": ["chunk_missing"],
        "retrieved_text_chunks": ["chunk_present"],
        "uncited_or_unretrieved": ["chunk_missing"],
    }


def test_answer_terms_support_alias_specs():
    evaluate_qa = _load_eval_module()
    result = {"answer": "Neo4j 是生产图存储，JSON 用于离线演示。"}
    spec = {"answer_terms": [{"any_of": ["Graph", "图存储"]}]}

    check = evaluate_qa._check_answer_terms(result, spec)

    assert check == {
        "passed": True,
        "required": 1,
        "matched": 1,
        "missing": [],
    }


def test_evaluate_qa_writes_output_report(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_neo4j",
                "question": "Neo4j 在项目里起什么作用？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "Neo4j"}],
                "answer_terms": ["Neo4j"],
            }
        ],
    )
    output_path = temp_dir / "qa-report.json"

    result = _run_eval(index_path, questions_path, output_path=output_path)

    assert result.returncode == 0, result.stdout
    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload == stdout_payload
    assert file_payload["summary"]["total"] == 1


def test_evaluate_qa_writes_markdown_report(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_neo4j",
                "type": "local",
                "question": "Neo4j 在项目里起什么作用？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "Neo4j"}],
            }
        ],
    )
    output_path = temp_dir / "reports" / "qa-report.md"

    result = _run_eval(
        index_path,
        questions_path,
        output_path=output_path,
        report_format="markdown",
    )

    assert result.returncode == 0, result.stdout
    report = output_path.read_text(encoding="utf-8")
    assert report == result.stdout
    assert "# QA Evaluation Report" in report
    assert "Status: PASS" in report
    assert "| retrieval_hit_rate | 1.0 |" in report
    assert "| entity_recall |" in report
    assert "| citation_grounding_rate |" in report
    assert "| local_neo4j | local | PASS | local | False |" in report


def test_evaluate_qa_thresholds_can_fail_otherwise_passing_report(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_graphrag",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
            }
        ],
    )

    result = _run_eval(
        index_path,
        questions_path,
        extra_args=["--min-relationship-recall", "1.1"],
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["thresholds"]["relationship_recall"] == {
        "minimum": 1.1,
        "actual": payload["summary"]["relationship_recall"],
        "passed": False,
    }


def test_evaluate_qa_invalid_threshold_returns_cli_error(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_graphrag",
                "question": "GraphRAG 是什么？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "GraphRAG"}],
            }
        ],
    )

    result = _run_eval(
        index_path,
        questions_path,
        extra_args=["--min-route-accuracy", "-0.1"],
    )

    assert result.returncode == 2
    assert "Threshold for route_accuracy must be >= 0." in result.stderr


def test_evaluate_qa_markdown_includes_threshold_table(temp_dir: Path):
    index_path = _write_index(temp_dir)
    questions_path = _write_questions(
        temp_dir,
        [
            {
                "id": "local_neo4j",
                "question": "Neo4j 在项目里起什么作用？",
                "expected_route": "local",
                "expected_refused": False,
                "required_entities": [{"name": "Neo4j"}],
            }
        ],
    )

    result = _run_eval(
        index_path,
        questions_path,
        report_format="markdown",
        extra_args=["--min-route-accuracy", "1.0"],
    )

    assert result.returncode == 0, result.stdout
    assert "## Thresholds" in result.stdout
    assert "| route_accuracy | 1.0 | 1.0 | PASS |" in result.stdout


def _write_index(temp_dir: Path) -> Path:
    docs_dir = temp_dir / "docs"
    index_path = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=index_path,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    return index_path


def _write_questions(temp_dir: Path, records: list[dict]) -> Path:
    path = temp_dir / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "evaluate_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_qa", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_eval(
    index_path: Path,
    questions_path: Path,
    output_path: Path | None = None,
    report_format: str = "json",
    config_path: Path | None = None,
    strict_neo4j: bool = False,
    answerer: str = "mock",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_qa.py"),
        "--index",
        str(index_path),
        "--questions",
        str(questions_path),
        "--format",
        report_format,
        "--answerer",
        answerer,
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if strict_neo4j:
        command.append("--strict-neo4j")
    if output_path is not None:
        command.extend(["--output", str(output_path)])
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
