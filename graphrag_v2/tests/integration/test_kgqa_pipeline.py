"""Integration tests for graph-grounded QA."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from graphrag_v2.config import GraphRagConfig
from graphrag_v2.community import Community, CommunityPipelineResult, CommunityReport
from graphrag_v2.artifacts import (
    GRAPH_EVIDENCE_KEYS,
    LINKED_ENTITY_KEYS,
    LLM_EXTRACTION_METADATA_KEYS,
    QA_RESULT_KEYS,
    RELATIONSHIP_EVIDENCE_KEYS,
    TEXT_EVIDENCE_KEYS,
    write_community_artifacts,
)
from graphrag_v2.cli.main import main as kgqa_main
from graphrag_v2.graph_store import GraphStoreError, GraphStoreStats
from graphrag_v2.indexing import index_fusion_only
from graphrag_v2.qa import GraphGroundedQA
from graphrag_v2.qa.prompts import QA_ANSWER_PROMPT_VERSION


def test_kgqa_index_default_full_mode_creates_artifacts(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Indexed graph:" in captured.out
    assert "Artifacts written to:" in captured.out
    assert (output_dir / "text_units.parquet").exists()
    assert (output_dir / "candidate_entities.parquet").exists()
    assert (output_dir / "candidate_relationships.parquet").exists()
    assert (output_dir / "candidate_triples.parquet").exists()
    assert (output_dir / "entities.parquet").exists()
    assert (output_dir / "relationships.parquet").exists()
    assert (output_dir / "rejected_triples.parquet").exists()
    assert (output_dir / "graph.json").exists()

    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert metadata["mode"] == "full"
    assert metadata["graph_store_provider"] == "json"
    assert metadata["graph_store_written"] is True
    assert metadata["metadata_schema_version"] == 1
    assert metadata["run_status"] == "succeeded"
    assert metadata["run_mode"] == "full"
    assert metadata["run_started_at"]
    assert metadata["run_finished_at"]
    assert metadata["run_elapsed_seconds"] >= 0
    assert metadata["num_text_units"] > 0
    assert metadata["num_candidate_entities"] > 0
    assert metadata["num_entities"] > 0


def test_kgqa_index_default_full_mode_records_stage_events(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Indexed graph:" in captured.out
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "run_summary.json").read_text())

    event_pairs = [(event["stage"], event["event"]) for event in events]
    assert event_pairs == [
        ("run", "run_start"),
        ("document", "stage_start"),
        ("document", "stage_end"),
        ("extraction", "stage_start"),
        ("extraction", "stage_end"),
        ("fusion", "stage_start"),
        ("fusion", "stage_end"),
        ("graph_store", "stage_start"),
        ("graph_store", "stage_end"),
        ("run", "run_end"),
    ]
    assert metadata["run_event_count"] == len(events)
    assert set(metadata["run_stage_timings"]) == {
        "document",
        "extraction",
        "fusion",
        "graph_store",
    }
    assert events[4]["counts"]["num_candidate_entities"] > 0
    assert events[6]["counts"]["num_entities"] > 0
    assert events[8]["provider"] == "json"
    assert summary["graph_store"]["provider"] == "json"


def test_kgqa_index_cli_strict_neo4j_rejects_json_store(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "strict-json"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--graph-store",
            "json",
            "--strict-neo4j",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--strict-neo4j requires --graph-store neo4j" in captured.err
    assert not output_dir.exists()


class StubChatProvider:
    provider_name = "stub"
    model_name = "stub-model"
    mock_mode = False
    total_errors = 0

    def __init__(self):
        self.calls = []

    def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
        self.calls.append(messages)
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "GraphRAG",
                        "type": "Technology",
                        "description": "GraphRAG combines graph evidence and RAG.",
                        "confidence": 0.9,
                    },
                    {
                        "name": "Knowledge Graph",
                        "type": "Technology",
                        "description": "Knowledge Graph stores structured evidence.",
                        "confidence": 0.86,
                    },
                ],
                "relationships": [
                    {
                        "source": "GraphRAG",
                        "target": "Knowledge Graph",
                        "relation": "uses",
                        "description": "GraphRAG uses knowledge graph evidence.",
                        "confidence": 0.84,
                    }
                ],
            }
        )

    def get_stats(self):
        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_seconds": 0.0,
            "max_latency_seconds": 0.0,
            "average_latency_seconds": 0.0,
            "estimated_cost": None,
        }


def test_kgqa_index_llm_full_mode_creates_artifacts(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j stores graph data.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_chat_provider",
        lambda *args, **kwargs: StubChatProvider(),
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--extractor",
            "llm",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Indexed graph:" in captured.out
    assert (output_dir / "candidate_entities.parquet").exists()
    assert (output_dir / "entities.parquet").exists()
    assert (output_dir / "graph.json").exists()

    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert metadata["extractor"] == "llm"
    assert metadata["num_entities"] > 0
    assert metadata["graph_store_provider"] == "json"
    assert metadata["llm_provider"] == "deepseek"
    assert metadata["llm_model_id"] == "default_chat_model"
    assert metadata["llm_mock_mode"] is False
    assert metadata["llm_total_calls"] >= 1
    assert metadata["llm_total_tokens"] == 0
    assert metadata["llm_prompt_tokens"] == 0
    assert metadata["llm_completion_tokens"] == 0
    assert metadata["llm_estimated_cost"] is None
    assert metadata["extraction_max_gleanings"] == 1
    assert metadata["extraction_gleaning_attempts"] == metadata["num_text_units"]
    assert metadata["extraction_gleaning_failures"] == 0
    assert metadata["extraction_failed_chunks"] == 0
    _assert_keys_present(metadata, LLM_EXTRACTION_METADATA_KEYS)


def test_kgqa_index_uses_extractor_provider_from_config_when_cli_omits_extractor(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    config_path = temp_dir / "settings.yaml"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )
    config_path.write_text(
        "extraction:\n"
        "  extractor_provider: llm\n"
        "  max_gleanings: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_chat_provider",
        lambda *args, **kwargs: StubChatProvider(),
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    assert exit_code == 0
    assert "Indexed graph:" in captured.out
    assert metadata["extractor"] == "llm"
    assert metadata["llm_provider"] == "deepseek"


def test_kgqa_index_llm_extraction_failure_outputs_error(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    class FailingExtractor:
        async def extract(self, text_unit):
            raise ValueError(
                f"bad llm response embedding-secret for {text_unit.chunk_id}"
            )

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_chat_provider",
        lambda *args, **kwargs: StubChatProvider(),
    )
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setattr(
        "graphrag_v2.indexing.LLMExtractor",
        lambda *args, **kwargs: FailingExtractor(),
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--extractor",
            "llm",
            "--mode",
            "extraction-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Indexing failed:" in captured.err
    assert "bad llm response" in captured.err
    assert "embedding-secret" not in captured.err
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "run_summary.json").read_text())
    metadata_blob = json.dumps(metadata, ensure_ascii=False)
    events_blob = json.dumps(events, ensure_ascii=False)
    summary_blob = json.dumps(summary, ensure_ascii=False)

    assert metadata["run_status"] == "failed"
    assert metadata["run_failed_stage"] == "extraction"
    assert metadata["run_error_type"] == "ValueError"
    assert "bad llm response" in metadata["run_error_message"]
    assert "embedding-secret" not in metadata_blob
    assert "embedding-secret" not in events_blob
    assert "embedding-secret" not in summary_blob
    assert any(
        event["stage"] == "extraction"
        and event["event"] == "stage_failed"
        and event["error_type"] == "ValueError"
        for event in events
    )
    assert any(event["event"] == "run_failed" for event in events)
    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "extraction"
    assert summary["errors"][0]["error_type"] == "ValueError"


def test_kgqa_index_community_requires_neo4j(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )

    exit_code = kgqa_main(
        [
            "index",
            str(docs_dir),
            "--output",
            str(output_dir),
            "--community",
            "--graph-store",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--community requires --graph-store neo4j" in captured.err
    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    community_events = [
        event for event in events if event["stage"] == "community"
    ]

    assert metadata["run_status"] == "failed"
    assert metadata["run_failed_stage"] == "community"
    assert metadata["run_error_type"] == "GraphStoreError"
    assert [
        (event["stage"], event["event"]) for event in community_events
    ] == [
        ("community", "stage_start"),
        ("community", "stage_failed"),
    ]
    assert all(event["provider"] == "json" for event in community_events)


def test_kgqa_index_strict_neo4j_preflight_failure_records_graph_store_stage(
    temp_dir: Path,
    monkeypatch,
):
    class FailingPreflightGraphStore:
        provider = "neo4j"

        def preflight(self):
            raise GraphStoreError("preflight failed")

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_graph_store",
        lambda *args, **kwargs: FailingPreflightGraphStore(),
    )

    with pytest.raises(GraphStoreError, match="preflight failed"):
        asyncio.run(
            index_fusion_only(
                input_path=docs_dir,
                output_path=output_dir,
                config=GraphRagConfig(),
                extractor_name="mock",
                graph_store_provider="neo4j",
                strict_neo4j=True,
            )
        )

    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [(event["stage"], event["event"]) for event in events] == [
        ("run", "run_start"),
        ("graph_store", "stage_start"),
        ("graph_store", "stage_failed"),
        ("run", "run_failed"),
    ]
    assert metadata["run_failed_stage"] == "graph_store"
    assert metadata["run_stage_timings"]["graph_store"] >= 0
    assert events[2]["elapsed_seconds"] is not None


def test_kgqa_index_optional_community_records_provider_in_stage_events(
    temp_dir: Path,
    monkeypatch,
):
    class FakeNeo4jGraphStore:
        provider = "neo4j"

        def write_graph(self, fusion_result):
            return GraphStoreStats(
                provider="neo4j",
                num_text_units=1,
                num_entities=len(fusion_result.entities),
                num_relationships=len(fusion_result.relationships),
                num_rejected_triples=len(fusion_result.rejected_triples),
                health_status="ready",
            )

    def fake_run_community_pipeline(*args, **kwargs):
        return CommunityPipelineResult(
            communities=[
                Community(
                    id="community_1",
                    level=0,
                    title="GraphRAG",
                    summary="GraphRAG community.",
                    entity_ids=[],
                    relationship_ids=[],
                    text_unit_ids=[],
                    size=1,
                    rank=1.0,
                    metadata={"source": "test"},
                )
            ],
            reports=[
                CommunityReport(
                    id="report_1",
                    community_id="community_1",
                    title="GraphRAG",
                    summary="GraphRAG community.",
                    full_content="GraphRAG community report.",
                    findings=[],
                    key_entities=[],
                    key_relationships=[],
                    evidence_chunk_ids=[],
                    rank=1.0,
                    metadata={"source": "test"},
                )
            ],
        )

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_graph_store",
        lambda *args, **kwargs: FakeNeo4jGraphStore(),
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.run_community_pipeline",
        fake_run_community_pipeline,
    )

    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="neo4j",
            enable_community=True,
        )
    )

    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    community_events = [
        event for event in events if event["stage"] == "community"
    ]

    assert [
        (event["stage"], event["event"]) for event in community_events
    ] == [
        ("community", "stage_start"),
        ("community", "stage_end"),
    ]
    assert all(event["provider"] == "neo4j" for event in community_events)


def test_kgqa_index_optional_community_failure_records_provider_in_stage_failed(
    temp_dir: Path,
    monkeypatch,
):
    class FakeNeo4jGraphStore:
        provider = "neo4j"

        def write_graph(self, fusion_result):
            return GraphStoreStats(
                provider="neo4j",
                num_text_units=1,
                num_entities=len(fusion_result.entities),
                num_relationships=len(fusion_result.relationships),
                num_rejected_triples=len(fusion_result.rejected_triples),
                health_status="ready",
            )

    def fake_run_community_pipeline(*args, **kwargs):
        raise GraphStoreError("community failed")

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.create_graph_store",
        lambda *args, **kwargs: FakeNeo4jGraphStore(),
    )
    monkeypatch.setattr(
        "graphrag_v2.indexing.run_community_pipeline",
        fake_run_community_pipeline,
    )

    with pytest.raises(GraphStoreError, match="community failed"):
        asyncio.run(
            index_fusion_only(
                input_path=docs_dir,
                output_path=output_dir,
                config=GraphRagConfig(),
                extractor_name="mock",
                graph_store_provider="neo4j",
                enable_community=True,
            )
        )

    metadata = json.loads((output_dir / "index_metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "run_summary.json").read_text())
    community_events = [
        event for event in events if event["stage"] == "community"
    ]

    assert metadata["run_status"] == "failed"
    assert metadata["run_failed_stage"] == "community"
    assert [
        (event["stage"], event["event"]) for event in community_events
    ] == [
        ("community", "stage_start"),
        ("community", "stage_failed"),
    ]
    assert all(event["provider"] == "neo4j" for event in community_events)
    assert any(event["event"] == "run_failed" for event in events)
    assert summary["failed_stage"] == "community"


def test_graph_grounded_qa_local_and_global(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    write_community_artifacts(
        output_path=output_dir,
        communities=[
            Community(
                id="community_1",
                level=0,
                title="GraphRAG 技术社区",
                summary="GraphRAG 与知识图谱、Neo4j 和社区检测相关。",
                entity_ids=entities["id"].tolist()[:2],
                relationship_ids=relationships["id"].tolist()[:1],
                text_unit_ids=text_units["chunk_id"].tolist(),
                size=2,
                rank=0.92,
                metadata={"source": "test"},
            )
        ],
        reports=[
            CommunityReport(
                id="report_1",
                community_id="community_1",
                title="GraphRAG 技术社区",
                summary="GraphRAG 与知识图谱、Neo4j 和社区检测相关。",
                full_content="GraphRAG uses knowledge graphs and Neo4j supports graph storage.",
                findings=[
                    "GraphRAG uses knowledge graphs",
                    "Neo4j stores the graph",
                ],
                key_entities=entities["name"].tolist()[:2],
                key_relationships=relationships["id"].tolist()[:1],
                evidence_chunk_ids=text_units["chunk_id"].tolist(),
                rank=0.92,
                metadata={"source": "test"},
            )
        ],
        algorithm="louvain",
        reporter="mock",
    )

    qa = GraphGroundedQA.from_index(output_dir)

    local_result = qa.ask("GraphRAG 是什么？")
    assert local_result.route == "local"
    assert local_result.graph_evidence.relationships
    assert local_result.citations
    assert "GraphRAG" in local_result.answer

    global_result = qa.ask("这批文档主要讲了哪些主题？")
    assert global_result.route == "global"
    assert global_result.community_evidence
    assert "GraphRAG 技术社区" in global_result.answer
    assert global_result.citations


def test_kgqa_ask_cli_outputs_sections(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Graph Evidence:" in captured.out
    assert "Citations:" in captured.out


def test_kgqa_ask_cli_outputs_json_contract(temp_dir: Path, capsys):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert set(payload) == set(QA_RESULT_KEYS)
    assert payload["question"] == "GraphRAG 是什么？"
    assert payload["route"] == "local"
    assert payload["refused"] is False
    assert payload["refusal_reason"] is None
    assert payload["citations"]
    assert payload["graph_evidence"]["linked_entities"]
    assert set(payload["graph_evidence"]) == set(GRAPH_EVIDENCE_KEYS)
    assert set(payload["graph_evidence"]["linked_entities"][0]) == set(
        LINKED_ENTITY_KEYS
    )
    assert set(payload["graph_evidence"]["relationships"][0]) == set(
        RELATIONSHIP_EVIDENCE_KEYS
    )
    assert set(payload["text_evidence"][0]) == set(TEXT_EVIDENCE_KEYS)
    assert payload["metadata"]["source_provider"] == "json"
    assert payload["metadata"]["answer_prompt_version"] == QA_ANSWER_PROMPT_VERSION
    trace = payload["metadata"]["query_trace"]
    assert trace["route"] == payload["route"]
    assert trace["linked_entities"]
    assert trace["retrieved_text_chunks"]


def test_kgqa_ask_cli_json_marks_neo4j_fallback(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    _mark_index_as_neo4j(output_dir)

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["source_provider"] == "json"
    assert payload["metadata"]["source_provider"] == "json"
    assert payload["metadata"]["qa_fallback_from_provider"] == "neo4j"
    assert "Neo4j" in payload["metadata"]["qa_fallback_reason"]


def test_kgqa_ask_cli_strict_neo4j_fails_without_connection(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    _mark_index_as_neo4j(output_dir)

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
            "--strict-neo4j",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Question answering failed:" in captured.err
    assert "Neo4j" in captured.err


def test_kgqa_ask_cli_llm_answerer_outputs_stubbed_answer(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    class StubGLMClient:
        mock_mode = False
        total_errors = 0

        def __init__(self, *args, **kwargs):
            pass

        def chat_completion(self, messages, temperature=0.2):
            assert messages[-1]["role"] == "user"
            assert "Text Evidence:" in messages[-1]["content"]
            return json.dumps(
                {
                    "candidate_id": "cand_1",
                    "answer_text": "GraphRAG",
                    "supported": True,
                    "reason": "selected first candidate",
                }
            )

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    monkeypatch.setattr(
        "graphrag_v2.cli.main.create_chat_provider",
        lambda *args, **kwargs: StubGLMClient(),
    )

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
            "--answerer",
            "llm",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["answer"] == "GraphRAG"
    assert payload["refused"] is False
    assert payload["citations"]


def test_kgqa_ask_cli_llm_answerer_requires_real_client(
    temp_dir: Path,
    capsys,
    monkeypatch,
):
    class StubMockGLMClient:
        mock_mode = True

        def __init__(self, *args, **kwargs):
            pass

        def chat_completion(self, messages, temperature=0.2):
            raise AssertionError("mock-mode LLM client should not be called")

    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )
    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    monkeypatch.setattr(
        "graphrag_v2.cli.main.create_chat_provider",
        lambda *args, **kwargs: StubMockGLMClient(),
    )

    exit_code = kgqa_main(
        [
            "ask",
            "GraphRAG 是什么？",
            "--index",
            str(output_dir),
            "--answerer",
            "llm",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "LLM answerer requires a configured real LLM client" in captured.err


def test_graph_grounded_qa_refuses_without_source_evidence(temp_dir: Path):
    docs_dir = temp_dir / "docs"
    output_dir = temp_dir / "index"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "GraphRAG uses Knowledge Graph evidence. Neo4j is a Graph Database.",
        encoding="utf-8",
    )

    asyncio.run(
        index_fusion_only(
            input_path=docs_dir,
            output_path=output_dir,
            config=GraphRagConfig(),
            extractor_name="mock",
            graph_store_provider="json",
        )
    )
    (output_dir / "text_units.parquet").unlink()

    result = GraphGroundedQA.from_index(output_dir).ask("GraphRAG 是什么？")

    assert result.refused is True
    assert result.refusal_reason == "no_source_evidence"
    assert result.citations == []
    assert result.confidence == 0.0
    assert "证据不足" in result.answer


def _mark_index_as_neo4j(output_dir: Path) -> None:
    metadata_path = output_dir / "index_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["graph_store_provider"] = "neo4j"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _assert_keys_present(payload: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    assert missing == []
