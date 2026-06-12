"""Tests for release verification policy scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_verify_release_inspects_run_observability_artifacts():
    script = (REPO_ROOT / "scripts" / "verify_release.sh").read_text(
        encoding="utf-8"
    )

    assert 'kgqa inspect run --index "$SMOKE_INDEX"' in script
    assert '"$SMOKE_INDEX/run_events.jsonl"' in script
    assert '"$SMOKE_INDEX/run_summary.json"' in script


def test_verify_release_runs_security_gate():
    script = (REPO_ROOT / "scripts" / "verify_release.sh").read_text(
        encoding="utf-8"
    )

    assert "python scripts/security_check.py" in script
