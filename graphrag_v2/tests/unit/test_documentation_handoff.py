"""Tests for documentation handoff completeness."""

from __future__ import annotations

from pathlib import Path

from graphrag_v2.api.app import create_app


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_readmes_publish_current_release_gate_baseline():
    for relative_path in ("README.md", "graphrag_v2/README.md"):
        text = _read(relative_path)

        assert "459 passed, 4 skipped" in text
        assert "scripts/security_check.py" in text
        assert "docs/operator_guide.md" in text


def test_operator_guide_covers_handoff_workflows():
    text = _read("docs/operator_guide.md")

    required_sections = (
        "## Installation",
        "## Configuration",
        "## Architecture And Artifact Lifecycle",
        "## Offline Indexing",
        "## QA",
        "## Neo4j Operations",
        "## QA Evaluation",
        "## Release Verification",
        "## Security And Data Governance",
        "## Troubleshooting",
        "## Known Limits",
    )
    for section in required_sections:
        assert section in text

    required_commands = (
        'python -m pip install -e ".[dev]"',
        "kgqa index examples/docs --output artifacts/demo",
        'kgqa ask "GraphRAG 是什么？" --index artifacts/demo',
        "kgqa inspect graph --index artifacts/demo",
        "kgqa inspect run --index artifacts/demo",
        "scripts/evaluate_qa.py",
        "python scripts/security_check.py",
        "scripts/verify_release.sh",
    )
    for command in required_commands:
        assert command in text

    assert "Documents -> chunks -> extraction -> fusion -> graph store -> QA" in text
    assert "Artifact lifecycle" in text


def test_examples_and_artifact_docs_cover_run_and_security_artifacts():
    examples = _read("examples/README.md")
    artifacts = _read("docs/artifacts.md")

    assert "kgqa inspect run --index artifacts/demo" in examples
    assert "scripts/security_check.py" in examples
    assert "kgqa inspect community-reports --index artifacts/neo4j-demo" in examples
    assert "--extractor llm" in examples
    assert "run_events.jsonl" in examples
    assert "run_summary.json" in examples

    assert "run_events.jsonl" in artifacts
    assert "run_summary.json" in artifacts
    assert "Index runs clear generated artifacts at run start" in artifacts
    assert "generated artifacts may contain source text" in artifacts.lower()
    assert "artifact retention" in artifacts.lower()


def test_release_candidate_audit_is_documented():
    operator = _read("docs/operator_guide.md")
    changelog = _read("CHANGELOG.md")

    for text in (operator, changelog):
        assert "scripts/audit_release_candidate.py" in text
        assert "report.json" in text
        assert "report.md" in text
        assert "api_runtime" in text
        assert "blocked" in text.lower()

    for text in (operator, changelog):
        assert "excluded from current release scope" in text
        assert "real LLM" in text or "real_llm" in text
        assert "deepseek-v4-flash" in text
        assert "KGQA_REAL_LLM_CONFIG" in text
        assert "settings.local.real-llm.yaml" in text
        assert "DEEPSEEK_API_KEY" in text


def test_beta_readiness_boundary_is_documented_in_public_docs():
    for relative_path in ("README.md", "docs/operator_guide.md", "CHANGELOG.md"):
        text = _read(relative_path)

        assert "not production-ready" in text
        assert "production-path beta" in text or "runnable beta" in text
        assert "deepseek-v4-flash" in text


def test_production_runtime_api_is_documented():
    readme = _read("README.md")
    operator = _read("docs/operator_guide.md")
    changelog = _read("CHANGELOG.md")

    for text in (readme, operator, changelog):
        assert "uvicorn graphrag_v2.api.app:app" in text
        assert "KGQA_API_INDEX_PATH" in text
        assert "KGQA_API_AUTH_TOKEN" in text
        assert "KGQA_API_MAX_QUESTION_CHARS" in text
        assert "/healthz" in text
        assert "/readyz" in text
        assert "/ask" in text
        assert "/metrics" in text
        assert "supervisor" in text.lower() or "monitoring" in text.lower()


def test_api_runtime_live_check_is_documented():
    for relative_path in (
        "docs/operator_guide.md",
        "CHANGELOG.md",
    ):
        text = _read(relative_path)

        assert "scripts/check_api_runtime.py" in text
        assert "--base-url" in text
        assert "live check" in text.lower() or "live-check" in text.lower()


def test_api_runtime_benchmark_is_documented():
    for relative_path in (
        "docs/operator_guide.md",
        "CHANGELOG.md",
    ):
        text = _read(relative_path)

        assert "scripts/benchmark_api_runtime.py" in text
        assert "--requests" in text
        assert "--concurrency" in text
        assert "--max-p95-ms" in text
        assert "latency" in text.lower()


def test_public_project_name_is_openfusionkgqa():
    previous_public_name = "".join(("K", "E", "T", "GraphRAG"))
    public_paths = (
        "README.md",
        "graphrag_v2/README.md",
        "docs/operator_guide.md",
        "CHANGELOG.md",
        "DEPLOYMENT.md",
        "examples/README.md",
    )
    for relative_path in public_paths:
        text = _read(relative_path)

        assert "OpenFusionKGQA" in text
        assert previous_public_name not in text

    assert create_app().title == "OpenFusionKGQA API"
