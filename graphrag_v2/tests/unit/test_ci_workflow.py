"""Tests for GitHub Actions CI policy."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _workflow() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
    )


def _normalize_expression(value: str) -> str:
    return " ".join(value.split())


def _env_has_external_secret_or_service(env: dict | None) -> bool:
    if not env:
        return False
    external_prefixes = (
        "NEO4J_",
        "ZHIPUAI_",
        "DEEPSEEK_",
        "KGQA_REAL_LLM_SMOKE",
        "KGQA_REAL_LLM_CONFIG",
        "KGQA_REAL_LLM_API_",
    )
    return any(key.startswith(external_prefixes) for key in env)


def test_ci_workflow_has_manual_optional_inputs():
    workflow = _workflow()
    dispatch = workflow[True]["workflow_dispatch"]
    inputs = dispatch["inputs"]

    assert set(inputs) >= {"run_neo4j", "run_real_llm", "run_docker"}
    for input_name in ("run_neo4j", "run_real_llm", "run_docker"):
        assert inputs[input_name]["type"] == "boolean"
        assert inputs[input_name]["default"] is False


def test_default_ci_job_is_offline_only():
    test_job = _workflow()["jobs"]["test"]

    assert "services" not in test_job
    assert not _env_has_external_secret_or_service(test_job.get("env"))

    run_steps = [step.get("run", "") for step in test_job["steps"]]
    assert "python -m pytest graphrag_v2/tests -q" in run_steps

    for step in test_job["steps"]:
        assert not _env_has_external_secret_or_service(step.get("env"))
        run = step.get("run", "").lower()
        uses = step.get("uses", "").lower()
        assert "neo4j" not in run
        assert "zhipuai_api_key" not in run
        assert "docker " not in run
        assert not uses.startswith("docker/")


def test_optional_jobs_are_workflow_dispatch_gated():
    jobs = _workflow()["jobs"]
    expectations = {
        "neo4j": "inputs.run_neo4j",
        "real-llm-smoke": "inputs.run_real_llm",
        "docker-build": "inputs.run_docker",
    }

    for job_name, input_guard in expectations.items():
        guard = _normalize_expression(jobs[job_name]["if"])
        assert guard == f"github.event_name == 'workflow_dispatch' && {input_guard}"


def test_real_llm_secret_is_scoped_to_smoke_steps():
    job = _workflow()["jobs"]["real-llm-smoke"]

    assert "DEEPSEEK_API_KEY" not in job.get("env", {})
    assert job["env"]["KGQA_REAL_LLM_CONFIG"] == "settings.local.real-llm.yaml"

    steps_with_key = [
        step.get("name")
        for step in job["steps"]
        if "DEEPSEEK_API_KEY" in step.get("env", {})
    ]
    assert steps_with_key == ["Write real DeepSeek config"]

    steps = {step["name"]: step for step in job["steps"]}
    assert (
        steps["Write real DeepSeek config"]["env"]["DEEPSEEK_API_KEY"]
        == "${{ secrets.DEEPSEEK_API_KEY }}"
    )
    write_config_run = steps["Write real DeepSeek config"]["run"]
    assert "deepseek-v4-flash" in write_config_run
    assert '"api_key": os.environ["DEEPSEEK_API_KEY"]' in write_config_run
    assert "yaml.safe_dump" in write_config_run
    assert "DEEPSEEK_API_KEY" not in steps["Run real DeepSeek smoke test"].get("env", {})
