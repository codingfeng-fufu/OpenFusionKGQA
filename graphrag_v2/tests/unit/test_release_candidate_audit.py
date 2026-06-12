"""Tests for service-backed release candidate audit tooling."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_audit():
    module_path = REPO_ROOT / "scripts" / "audit_release_candidate.py"
    spec = importlib.util.spec_from_file_location("audit_release_candidate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_overall_status_is_failed_when_any_gate_fails():
    audit = _load_audit()
    gates = [
        audit.GateResult(name="offline", status="passed", required=True),
        audit.GateResult(name="neo4j", status="failed", required=True),
        audit.GateResult(name="docker", status="blocked", required=False),
    ]

    assert audit.compute_overall_status(gates) == "failed"


def test_overall_status_is_blocked_when_optional_gates_are_blocked():
    audit = _load_audit()
    gates = [
        audit.GateResult(name="offline", status="passed", required=True),
        audit.GateResult(name="neo4j", status="passed", required=True),
        audit.GateResult(name="docker", status="blocked", required=False),
        audit.GateResult(name="real_llm", status="blocked", required=False),
    ]

    assert audit.compute_overall_status(gates) == "blocked"


def test_docker_gate_records_blocked_when_binary_missing(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.setattr(audit.shutil, "which", lambda command: None)

    result = audit.run_docker_gate(tmp_path, dry_run=False, include=True)

    assert result.status == "blocked"
    assert result.reason == "docker binary is not available"
    assert "docker build -t openfusion-kgqa:rc-audit ." in result.commands


def test_docker_gate_is_skipped_by_default_for_current_release_scope(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.setattr(audit.shutil, "which", lambda command: None)

    result = audit.run_docker_gate(tmp_path, dry_run=False)

    assert result.status == "skipped"
    assert result.reason == "excluded from current release scope"
    assert result.required is False


def test_offline_gate_disables_real_llm_opt_in_for_release_verification(
    tmp_path, monkeypatch
):
    audit = _load_audit()
    captured = {}

    def fake_run_command(command, *, cwd, env=None, display_env=None):
        captured["command"] = command
        captured["env"] = dict(env or {})
        return audit.CommandResult(command=audit.command_to_text(command), returncode=0)

    monkeypatch.setenv("KGQA_REAL_LLM_SMOKE", "1")
    monkeypatch.setattr(audit, "run_command", fake_run_command)

    result = audit.run_offline_gate(tmp_path, dry_run=False)

    assert result.status == "passed"
    assert captured["command"] == ["scripts/verify_release.sh"]
    assert captured["env"] == {"KGQA_REAL_LLM_SMOKE": "0"}


def test_api_runtime_gate_runs_targeted_contract_tests(tmp_path, monkeypatch):
    audit = _load_audit()
    captured = {}

    def fake_run_command(command, *, cwd, env=None, display_env=None):
        captured["command"] = command
        captured["env"] = dict(env or {})
        return audit.CommandResult(
            command=audit.command_to_text(command),
            returncode=0,
            stdout_tail="12 passed",
        )

    monkeypatch.setenv("KGQA_REAL_LLM_SMOKE", "1")
    monkeypatch.setattr(audit, "run_command", fake_run_command)

    result = audit.run_api_runtime_gate(tmp_path, dry_run=False)

    assert result.name == "api_runtime"
    assert result.status == "passed"
    assert result.required is True
    assert "metrics" in result.metadata["contracts"]
    assert captured["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "graphrag_v2/tests/unit/test_api_runtime.py",
        "-q",
    ]
    assert captured["env"] == {"KGQA_REAL_LLM_SMOKE": "0"}


def test_api_runtime_gate_can_be_skipped_but_remains_required(tmp_path):
    audit = _load_audit()

    result = audit.run_api_runtime_gate(tmp_path, dry_run=False, skip=True)

    assert result.status == "skipped"
    assert result.required is True
    assert result.reason == "disabled by --skip-api-runtime"


def test_neo4j_gate_starts_fresh_instance_without_inherited_home(
    tmp_path, monkeypatch
):
    audit = _load_audit()
    helper = tmp_path / "scripts" / "start_fresh_neo4j.sh"
    helper.parent.mkdir()
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    env_file = tmp_path / "kgqa-neo4j.env"
    env_file.write_text(
        "\n".join(
            [
                'export NEO4J_URI="bolt://127.0.0.1:7699"',
                'export NEO4J_USERNAME="neo4j"',
                'export NEO4J_PASSWORD="dummy-password"',
                'export NEO4J_DATABASE="neo4j"',
                f'export KGQA_NEO4J_HOME="{tmp_path / "fresh-home"}"',
            ]
        ),
        encoding="utf-8",
    )
    captured = []

    def fake_run_command(command, *, cwd, env=None, display_env=None):
        captured.append((command, dict(env or {})))
        if command == ["scripts/start_fresh_neo4j.sh"]:
            return audit.CommandResult(
                command=audit.command_to_text(command),
                returncode=0,
                stdout_tail=f"env_file: {env_file}\n",
            )
        return audit.CommandResult(
            command=audit.command_to_text(command),
            returncode=0,
            stdout_tail="ok",
        )

    monkeypatch.setenv("KGQA_NEO4J_HOME", str(tmp_path / "already-running"))
    monkeypatch.setenv("KGQA_NEO4J_INSTANCE_NAME", "already-running")
    monkeypatch.setattr(audit, "run_command", fake_run_command)

    result = audit.run_neo4j_gate(tmp_path, dry_run=False)

    assert result.status == "passed"
    start_command, start_env = captured[0]
    assert start_command == ["scripts/start_fresh_neo4j.sh"]
    assert start_env == {
        "KGQA_NEO4J_HOME": "",
        "KGQA_NEO4J_INSTANCE_NAME": "",
        "KGQA_NEO4J_BOLT_PORT": "",
        "KGQA_NEO4J_HTTP_PORT": "",
    }


def test_real_llm_gate_records_blocked_without_explicit_opt_in(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.delenv("KGQA_REAL_LLM_SMOKE", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = audit.run_real_llm_gate(tmp_path, dry_run=False)

    assert result.status == "blocked"
    assert result.reason == "KGQA_REAL_LLM_SMOKE=1 is not set"
    assert "test_real_llm_smoke.py" in result.commands[0]


def test_real_llm_gate_records_blocked_without_api_key(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.setenv("KGQA_REAL_LLM_SMOKE", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("KGQA_REAL_LLM_API_KEY", raising=False)

    result = audit.run_real_llm_gate(tmp_path, dry_run=False)

    assert result.status == "blocked"
    assert result.reason == "DEEPSEEK_API_KEY or KGQA_REAL_LLM_API_KEY is not set"


def test_real_llm_gate_reports_default_deepseek_metadata(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.delenv("KGQA_REAL_LLM_SMOKE", raising=False)
    monkeypatch.delenv("KGQA_REAL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KGQA_REAL_LLM_MODEL", raising=False)

    result = audit.run_real_llm_gate(tmp_path, dry_run=False)

    assert result.status == "blocked"
    assert result.metadata == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key": "unset",
    }


def test_real_llm_gate_uses_local_config_metadata_without_leaking_key(tmp_path, monkeypatch):
    audit = _load_audit()
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: deepseek-v4-flash
    model_provider: deepseek
    api_base: https://api.deepseek.com
    api_key: config-secret
extraction:
  llm_provider: deepseek
  llm_model_id: default_chat_model
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("KGQA_REAL_LLM_SMOKE", raising=False)
    monkeypatch.setenv("KGQA_REAL_LLM_CONFIG", str(config_path))

    result = audit.run_real_llm_gate(tmp_path, dry_run=False)
    report = audit.build_report(
        [result],
        audit.detect_environment(tmp_path),
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
    )
    paths = audit.write_reports(report, tmp_path / "audit")
    report_json = paths["json"].read_text(encoding="utf-8")
    report_markdown = paths["markdown"].read_text(encoding="utf-8")

    assert result.status == "blocked"
    assert result.metadata == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "KGQA_REAL_LLM_CONFIG",
        "api_key": "set",
        "config_path": str(config_path),
    }
    assert report["environment"]["real_llm_config_path"] == str(config_path)
    assert report["environment"]["real_llm_api_key"] == "set"
    assert "config-secret" not in report_json
    assert "config-secret" not in report_markdown


def test_real_llm_gate_passes_config_key_to_subprocess_for_redaction(
    tmp_path, monkeypatch
):
    audit = _load_audit()
    config_path = tmp_path / "settings.local.real-llm.yaml"
    config_path.write_text(
        """
models:
  default_chat_model:
    type: chat
    model: deepseek-v4-flash
    model_provider: deepseek
    api_base: https://api.deepseek.com
    api_key: config-secret
extraction:
  llm_provider: deepseek
  llm_model_id: default_chat_model
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_command(command, *, cwd, env=None, display_env=None):
        captured["env"] = dict(env or {})
        captured["display_env"] = dict(display_env or {})
        return audit.CommandResult(
            command=audit.command_to_text(command, display_env),
            returncode=0,
            stdout_tail="ok",
        )

    monkeypatch.setenv("KGQA_REAL_LLM_SMOKE", "1")
    monkeypatch.setenv("KGQA_REAL_LLM_CONFIG", str(config_path))
    monkeypatch.setattr(audit, "run_command", fake_run_command)

    result = audit.run_real_llm_gate(tmp_path, dry_run=False)

    assert result.status == "passed"
    assert captured["env"] == {"KGQA_REAL_LLM_API_KEY": "config-secret"}
    assert captured["display_env"] == {
        "KGQA_REAL_LLM_SMOKE": "1",
        "KGQA_REAL_LLM_PROVIDER": "deepseek",
        "KGQA_REAL_LLM_MODEL": "deepseek-v4-flash",
        "KGQA_REAL_LLM_API_BASE": "https://api.deepseek.com",
        "KGQA_REAL_LLM_CONFIG": str(config_path),
    }
    assert "config-secret" not in result.commands[0]


def test_report_json_and_markdown_include_blocked_gates(tmp_path):
    audit = _load_audit()
    gates = [
        audit.GateResult(
            name="offline",
            status="passed",
            required=True,
            commands=["scripts/verify_release.sh"],
            elapsed_seconds=1.25,
        ),
        audit.GateResult(
            name="docker",
            status="blocked",
            required=False,
            reason="docker binary is not available",
            commands=["docker build -t openfusion-kgqa:rc-audit ."],
        ),
    ]

    report = audit.build_report(
        gates,
        environment={"docker": "missing"},
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:10Z",
    )
    paths = audit.write_reports(report, tmp_path)

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert data["audit_started_at"] == "2026-06-07T00:00:00Z"
    assert data["audit_finished_at"] == "2026-06-07T00:00:10Z"
    assert data["overall_status"] == "blocked"
    assert data["gates"][1]["status"] == "blocked"
    assert data["environment"]["docker"] == "missing"
    assert "## Gate Summary" in markdown
    assert "| docker | blocked | no | docker binary is not available |" in markdown


def test_command_output_redaction_masks_sensitive_values():
    audit = _load_audit()

    redacted = audit.redact_text(
        (
            "DEEPSEEK_API_KEY=deepseek-value "
            "KGQA_REAL_LLM_API_KEY=generic-value "
            "KGQA_API_AUTH_TOKEN=api-token "
            "ZHIPUAI_API_KEY=secret-value "
            "NEO4J_PASSWORD=pass-value token"
        ),
        sensitive_values={
            "deepseek-value",
            "generic-value",
            "secret-value",
            "pass-value",
        },
    )

    assert "deepseek-value" not in redacted
    assert "generic-value" not in redacted
    assert "api-token" not in redacted
    assert "secret-value" not in redacted
    assert "pass-value" not in redacted
    assert "[REDACTED]" in redacted


def test_dry_run_cli_writes_report_without_running_heavy_gates(tmp_path):
    audit = _load_audit()

    exit_code = audit.main(["--dry-run", "--output", str(tmp_path)])

    assert exit_code == 0
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["overall_status"] == "blocked"
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert set(gates) == {"offline", "api_runtime", "neo4j", "docker", "real_llm"}
    assert gates["offline"]["status"] == "blocked"
    assert gates["api_runtime"]["status"] == "blocked"
    assert gates["api_runtime"]["reason"] == "dry-run does not execute audit gates"
    assert gates["neo4j"]["status"] == "blocked"
    assert gates["docker"]["status"] == "skipped"
    assert gates["docker"]["reason"] == "excluded from current release scope"
    assert gates["real_llm"]["status"] == "blocked"


def test_include_docker_cli_evaluates_docker_gate(tmp_path, monkeypatch):
    audit = _load_audit()
    monkeypatch.setattr(audit.shutil, "which", lambda command: None)

    exit_code = audit.main(
        [
            "--dry-run",
            "--include-docker",
            "--skip-offline",
            "--skip-api-runtime",
            "--skip-neo4j",
            "--skip-real-llm",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["docker"]["status"] == "blocked"
    assert gates["docker"]["reason"] == "dry-run does not execute heavyweight gates"
