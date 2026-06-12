"""Tests for the HotpotQA Mini benchmark adapter."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_hotpotqa_benchmark():
    module_path = REPO_ROOT / "scripts" / "benchmark_hotpotqa_mini.py"
    spec = importlib.util.spec_from_file_location("benchmark_hotpotqa_mini", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_hotpotqa_fixture(path: Path) -> None:
    records = [
        {
            "_id": "case_2",
            "question": "Which city hosted Example Games where Ada Lovelace performed?",
            "answer": "Oxford",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Example Games", 0], ["Ada Lovelace", 0]],
            "context": [
                [
                    "Example Games",
                    [
                        "Example Games were hosted in Oxford.",
                        "The games included several exhibitions.",
                    ],
                ],
                [
                    "Ada Lovelace",
                    [
                        "Ada Lovelace performed at Example Games.",
                        "She also wrote about machines.",
                    ],
                ],
                ["Distractor Page", ["Distractor Page is unrelated."]],
            ],
        },
        {
            "_id": "case_1",
            "question": "What instrument did Grace Hopper use in Example Orchestra?",
            "answer": "violin",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Example Orchestra", 0], ["Grace Hopper", 0]],
            "context": [
                [
                    "Example Orchestra",
                    [
                        "Example Orchestra featured a violin section.",
                        "It toured in 1950.",
                    ],
                ],
                [
                    "Grace Hopper",
                    [
                        "Grace Hopper used a violin in Example Orchestra.",
                        "She later worked with computers.",
                    ],
                ],
            ],
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_prepare_hotpotqa_mini_writes_docs_questions_and_gold_index(tmp_path: Path):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    output_dir = tmp_path / "hotpotqa-mini"
    _write_hotpotqa_fixture(input_path)

    prepared = benchmark.prepare_hotpotqa_mini(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=1,
        seed=7,
    )

    assert prepared.sample_size == 1
    assert prepared.questions_path == output_dir / "questions.jsonl"
    assert prepared.index_path == output_dir / "index"
    assert (output_dir / "docs" / "case_1.md").exists()

    questions = [
        json.loads(line)
        for line in prepared.questions_path.read_text(encoding="utf-8").splitlines()
    ]
    expected_orchestra_chunk = benchmark.stable_id(
        "hotpot_chunk",
        "case_1",
        "Example Orchestra",
    )
    expected_hopper_chunk = benchmark.stable_id(
        "hotpot_chunk",
        "case_1",
        "Grace Hopper",
    )
    assert questions == [
        {
            "id": "hotpotqa_case_1",
            "type": "hotpotqa_bridge",
            "question": "What instrument did Grace Hopper use in Example Orchestra?",
            "expected_route": "local",
            "expected_refused": False,
            "answer": "violin",
            "required_entities": [
                {"name": "Example Orchestra"},
                {"name": "Grace Hopper"},
            ],
            "required_relationships": [
                {
                    "source": {"name": "Example Orchestra"},
                    "relation": {"name": "supports_answer"},
                    "target": {"name": "Grace Hopper"},
                }
            ],
            "required_citations": [
                expected_orchestra_chunk,
                expected_hopper_chunk,
            ],
            "answer_terms": ["violin"],
            "metadata": {
                "benchmark": "HotpotQA",
                "hotpotqa_id": "case_1",
                "level": "medium",
                "supporting_titles": ["Example Orchestra", "Grace Hopper"],
            },
        }
    ]

    metadata = json.loads((prepared.index_path / "index_metadata.json").read_text())
    assert metadata["benchmark"] == "HotpotQA Mini"
    assert metadata["sample_seed"] == 7
    assert metadata["sample_size"] == 1
    assert metadata["graph_store_provider"] == "json"


def test_hotpotqa_sample_manifest_uses_nested_deterministic_prefixes(tmp_path: Path):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    _write_hotpotqa_fixture(input_path)
    records = benchmark.load_hotpotqa_records(input_path)

    manifest = benchmark.build_sample_manifest(
        records,
        sample_sizes=[1, 2],
        seed=7,
        source_file=input_path,
    )

    assert manifest["seed"] == 7
    assert manifest["selection"] == "seeded_shuffle_prefix"
    assert manifest["samples"]["dev1"]["ids"] == manifest["ordered_ids"][:1]
    assert manifest["samples"]["dev2"]["ids"] == manifest["ordered_ids"][:2]
    assert manifest["samples"]["dev1"]["ids"] == manifest["samples"]["dev2"]["ids"][:1]


def test_select_hotpotqa_records_is_nested_for_larger_samples(tmp_path: Path):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    _write_hotpotqa_fixture(input_path)
    records = benchmark.load_hotpotqa_records(input_path)

    small = benchmark.select_hotpotqa_records(records, sample_size=1, seed=7)
    large = benchmark.select_hotpotqa_records(records, sample_size=2, seed=7)

    assert [item["_id"] for item in small] == [item["_id"] for item in large[:1]]


def test_run_hotpotqa_mini_benchmark_writes_portfolio_reports(tmp_path: Path):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    output_dir = tmp_path / "hotpotqa-mini"
    _write_hotpotqa_fixture(input_path)

    report = benchmark.run_hotpotqa_mini_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=1,
        seed=7,
        real_llm_smoke_size=0,
    )

    assert report["benchmark"]["name"] == "HotpotQA Mini"
    assert report["benchmark"]["sample_size"] == 1
    assert report["offline_qa"]["summary"]["total"] == 1
    assert report["offline_qa"]["summary"]["citation_coverage"] == 1.0
    assert report["answer_quality"]["average_token_f1"] > 0
    assert report["real_llm_smoke"]["enabled"] is False
    assert (output_dir / "hotpotqa-mini-report.json").exists()

    markdown = (output_dir / "hotpotqa-mini-report.md").read_text(encoding="utf-8")
    assert "# HotpotQA Mini Benchmark Report" in markdown
    assert "fixed seed `7`" in markdown
    assert "Real LLM smoke: disabled" in markdown


def test_hotpotqa_qa_cases_include_retrieval_diagnostics(tmp_path: Path):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    output_dir = tmp_path / "hotpotqa-mini"
    _write_hotpotqa_fixture(input_path)
    prepared = benchmark.prepare_hotpotqa_mini(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=1,
        seed=7,
    )
    questions = [
        json.loads(line)
        for line in prepared.questions_path.read_text(encoding="utf-8").splitlines()
    ]

    report = benchmark.evaluate_hotpotqa_questions(
        index_path=prepared.index_path,
        questions=questions,
        answerer=None,
    )

    case = report["cases"][0]
    assert case["citation_hits"] == case["required_citations"]
    assert case["missing_required_citations"] == []
    assert case["adaptive_enabled"] is True
    assert case["adaptive_triggered"] is False
    assert case["matched_adaptive_cues"] == []
    assert isinstance(case["query_trace"], dict)
    assert case["linked_entities"]
    assert case["retrieved_relationships"]
    assert case["retrieved_text_chunks"]
    assert case["relationship_count_by_hop"] == {"1": 1}
    assert case["max_retrieved_hop"] == 1


def test_run_hotpotqa_mini_benchmark_supports_real_llm_smoke_with_fake_client(
    tmp_path: Path,
    monkeypatch,
):
    benchmark = _load_hotpotqa_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    config_path = tmp_path / "settings.local.real-llm.yaml"
    output_dir = tmp_path / "hotpotqa-mini"
    _write_hotpotqa_fixture(input_path)
    config_path.write_text("models: {}\n", encoding="utf-8")

    class FakeConfig:
        class extraction:
            llm_provider = "deepseek"

        def get_language_model_config(self, _model_id):
            return object()

    class FakeLLMClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.7, max_tokens=None, stream=False):
            assert messages
            return json.dumps(
                {
                    "candidate_id": "cand_1",
                    "answer_text": "violin",
                    "supported": True,
                    "reason": "selected instrument candidate",
                }
            )

    monkeypatch.setattr(benchmark, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(
        benchmark,
        "create_chat_provider",
        lambda provider, model_config, require_real: FakeLLMClient(),
    )

    report = benchmark.run_hotpotqa_mini_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=1,
        seed=7,
        real_llm_smoke_size=1,
        config_path=config_path,
    )

    assert report["real_llm_smoke"]["enabled"] is True
    assert report["real_llm_smoke"]["sample_size"] == 1
    assert report["real_llm_smoke"]["status"] == "passed"
    assert report["real_llm_smoke"]["summary"]["total"] == 1
    assert report["real_llm_smoke"]["cases"][0]["answer"] == "violin"


def test_resolve_input_path_uses_download_cache_without_network(tmp_path: Path, monkeypatch):
    benchmark = _load_hotpotqa_benchmark()
    cache_path = tmp_path / "hotpot_dev_distractor_v1.json"
    cache_path.write_text("[]", encoding="utf-8")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("download should not run when cache exists")

    monkeypatch.setattr(benchmark, "download_file", fail_download)

    resolved = benchmark.resolve_input_path(
        Namespace(
            input=None,
            download=True,
            cache=str(cache_path),
            download_url="https://example.invalid/hotpot.json",
            download_timeout=0.1,
        )
    )

    assert resolved == cache_path
