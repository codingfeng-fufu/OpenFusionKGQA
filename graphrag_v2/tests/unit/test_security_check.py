"""Tests for the offline release security gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_security_check():
    module_path = REPO_ROOT / "scripts" / "security_check.py"
    spec = importlib.util.spec_from_file_location("security_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_security_repo(root: Path) -> None:
    (root / ".gitignore").write_text(
        "\n".join(
            [
                ".env",
                ".env.*",
                "!.env.example",
                "settings.local.yaml",
                "settings.local.yml",
                "settings.local.json",
                "settings.local.*.yaml",
                "settings.local.*.yml",
                "settings.local.*.json",
                "/artifacts/",
                "/cache/",
                "/logs/",
                "*.log",
            ]
        ),
        encoding="utf-8",
    )
    package_dir = root / "graphrag_v2" / "artifacts"
    package_dir.mkdir(parents=True)
    (package_dir / "run_observability.py").write_text(
        "SENSITIVE_ENV_NAMES = ("
        "'NEO4J_PASSWORD', "
        "'ZHIPUAI_API_KEY', "
        "'DEEPSEEK_API_KEY', "
        "'KGQA_REAL_LLM_API_KEY', "
        "'GRAPHRAG_API_KEY', "
        "'GRAPHRAG_EMBEDDING_API_KEY', "
        "'OPENAI_API_KEY', "
        "'LOCAL_LLM_API_KEY'"
        ")\n",
        encoding="utf-8",
    )
    docs_dir = root / "docs"
    docs_dir.mkdir()
    (docs_dir / "operator_guide.md").write_text(
        "artifact retention\nmodel output storage\ndependency vulnerability review\nlicense review\n",
        encoding="utf-8",
    )
    (docs_dir / "artifacts.md").write_text(
        "generated artifacts may contain source text\nartifact retention\npython scripts/security_check.py\n",
        encoding="utf-8",
    )
    (root / "DEPLOYMENT.md").write_text(
        "generated artifacts may contain source text\n"
        "model outputs require retention review\n"
        "do not index sensitive data without governance review\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("ZHIPUAI_API_KEY=your-zhipu-key\n", encoding="utf-8")


def test_security_check_passes_current_repository():
    security_check = _load_security_check()

    result = security_check.run_checks(REPO_ROOT)

    assert result.ok, result.format()


def test_security_check_requires_gitignore_patterns(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert ".gitignore" in result.format()
    assert "settings.local.yaml" in result.format()


def test_security_check_requires_redaction_env_names(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    redaction_file = tmp_path / "graphrag_v2" / "artifacts" / "run_observability.py"
    redaction_file.write_text("SENSITIVE_ENV_NAMES = ('NEO4J_PASSWORD',)\n", encoding="utf-8")

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert "redaction allowlist" in result.format()
    assert "OPENAI_API_KEY" in result.format()


def test_security_check_reads_actual_redaction_tuple_not_comments(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    redaction_file = tmp_path / "graphrag_v2" / "artifacts" / "run_observability.py"
    redaction_file.write_text(
        "# OPENAI_API_KEY LOCAL_LLM_API_KEY are documented here but not active.\n"
        "SENSITIVE_ENV_NAMES = ("
        "'NEO4J_PASSWORD', "
        "'ZHIPUAI_API_KEY', "
        "'GRAPHRAG_API_KEY', "
        "'GRAPHRAG_EMBEDDING_API_KEY'"
        ")\n",
        encoding="utf-8",
    )

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert "redaction allowlist" in result.format()
    assert "OPENAI_API_KEY" in result.format()
    assert "LOCAL_LLM_API_KEY" in result.format()


def test_security_check_rejects_committed_credential_assignments(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    env_name = "ZHIPUAI" + "_API_KEY"
    leaked_value = "sk-" + "real-looking-secret-token-" + "1234567890"
    (tmp_path / "settings.example.yaml").write_text(
        f'{env_name}="{leaked_value}"\n',
        encoding="utf-8",
    )

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert "credential" in result.format()
    assert "settings.example.yaml" in result.format()


def test_security_check_skips_ignored_local_settings_files(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    leaked_value = "sk-" + "AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    (tmp_path / "settings.local.real-llm.yaml").write_text(
        f"api_key: {leaked_value}\n",
        encoding="utf-8",
    )

    result = security_check.run_checks(tmp_path)

    assert result.ok, result.format()


def test_security_check_scans_env_example_files(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    env_name = "OPENAI" + "_API_KEY"
    leaked_value = "sk-" + "leaked-secret-token-" + "1234567890"
    (tmp_path / ".env.example").write_text(
        f'{env_name}="{leaked_value}"\n',
        encoding="utf-8",
    )

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert "credential" in result.format()
    assert ".env.example" in result.format()


def test_security_check_requires_governance_docs(tmp_path):
    security_check = _load_security_check()
    _write_security_repo(tmp_path)
    (tmp_path / "docs" / "operator_guide.md").write_text("operator guide\n", encoding="utf-8")

    result = security_check.run_checks(tmp_path)

    assert not result.ok
    assert "governance" in result.format()
    assert "artifact retention" in result.format()
