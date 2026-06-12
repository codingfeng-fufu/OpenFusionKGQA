"""Tests for the isolated HotpotQA benchmark runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from graphrag_v2.tests.unit.test_hotpotqa_mini_benchmark import _write_hotpotqa_fixture


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_isolated_benchmark():
    module_path = REPO_ROOT / "scripts" / "run_hotpotqa_isolated_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "run_hotpotqa_isolated_benchmark",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_isolated_hotpotqa_benchmark_writes_per_case_indexes(tmp_path: Path):
    benchmark = _load_isolated_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    output_dir = tmp_path / "isolated"
    _write_hotpotqa_fixture(input_path)

    report = benchmark.run_isolated_hotpotqa_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=2,
        seed=7,
        answerer_name="mock",
        config_path=None,
    )

    assert report["benchmark"]["name"] == "HotpotQA Isolated"
    assert report["summary"]["total"] == 2
    assert len(report["cases"]) == 2
    assert (output_dir / "hotpotqa-isolated-report.json").exists()
    assert (output_dir / "hotpotqa-isolated-report.md").exists()
    manifest = json.loads((output_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"]["dev50"]["ids"] == manifest["ordered_ids"][:50]
    assert manifest["samples"]["dev100"]["ids"] == manifest["ordered_ids"][:100]
    assert manifest["samples"]["dev200"]["ids"] == manifest["ordered_ids"][:200]
    assert report["benchmark"]["sample_manifest_path"] == str(output_dir / "sample_manifest.json")
    for case in report["cases"]:
        case_index = Path(case["case_index_path"])
        assert case_index.exists()
        assert (case_index / "text_units.parquet").exists()
        assert case["hotpotqa_id"]
        assert case["answer_prompt_version"]


def test_isolated_benchmark_supports_fake_llm_client(tmp_path: Path, monkeypatch):
    benchmark = _load_isolated_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    config_path = tmp_path / "settings.yaml"
    output_dir = tmp_path / "isolated"
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
        provider_name = "fake"
        model_name = "fake-model"

        def __init__(self):
            self.total_calls = 0

        def chat_completion(self, messages, temperature=0.7, max_tokens=None, stream=False):
            self.total_calls += 1
            return json.dumps(
                {
                    "candidate_id": "cand_1",
                    "answer_text": "violin",
                    "supported": True,
                    "reason": "selected instrument candidate",
                }
            )

        def get_stats(self):
            return {"total_calls": self.total_calls}

    monkeypatch.setattr(benchmark, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(
        benchmark,
        "create_chat_provider",
        lambda provider, model_config, require_real: FakeLLMClient(),
    )

    report = benchmark.run_isolated_hotpotqa_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=1,
        seed=7,
        answerer_name="llm",
        config_path=config_path,
    )

    assert report["benchmark"]["answerer"] == "real_llm"
    assert report["llm_stats"]["total_calls"] == 1
    assert report["cases"][0]["answer"] == "violin"


def test_isolated_benchmark_records_case_error_and_continues(tmp_path: Path, monkeypatch):
    benchmark = _load_isolated_benchmark()
    input_path = tmp_path / "hotpot_dev.json"
    config_path = tmp_path / "settings.yaml"
    output_dir = tmp_path / "isolated"
    _write_hotpotqa_fixture(input_path)
    config_path.write_text("models: {}\n", encoding="utf-8")

    class FakeConfig:
        class extraction:
            llm_provider = "deepseek"

        def get_language_model_config(self, _model_id):
            return object()

    class FlakyLLMClient:
        mock_mode = False
        total_errors = 0
        provider_name = "fake"
        model_name = "fake-model"

        def __init__(self):
            self.total_calls = 0

        def chat_completion(self, messages, temperature=0.7, max_tokens=None, stream=False):
            self.total_calls += 1
            if self.total_calls == 1:
                return ""
            content = messages[-1]["content"]
            answer = "violin" if '"answer_text": "violin"' in content else "Oxford"
            return json.dumps(
                {
                    "candidate_id": "",
                    "answer_text": answer,
                    "supported": True,
                    "reason": "selected gold candidate",
                }
            )

        def get_stats(self):
            return {"total_calls": self.total_calls}

    monkeypatch.setattr(benchmark, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(
        benchmark,
        "create_chat_provider",
        lambda provider, model_config, require_real: FlakyLLMClient(),
    )

    report = benchmark.run_isolated_hotpotqa_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=2,
        seed=7,
        answerer_name="llm",
        config_path=config_path,
    )

    assert report["summary"]["total"] == 2
    assert report["summary"]["errors"] == 1
    assert report["cases"][0]["passed"] is False
    assert "empty response" in report["cases"][0]["error"]
    assert report["cases"][1]["error"] is None
    assert report["cases"][1]["answer"] == report["cases"][1]["gold_answer"]
