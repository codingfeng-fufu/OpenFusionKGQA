"""Tests for pytest process-level environment guards."""

from __future__ import annotations

import os
import subprocess
import sys


def test_pytest_limits_omp_threads_for_cli_subprocesses():
    assert os.environ.get("OMP_NUM_THREADS") == "1"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('OMP_NUM_THREADS', ''))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "1"
