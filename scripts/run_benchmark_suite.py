#!/usr/bin/env python3
"""Run the OpenFusionKGQA benchmark suite with stable logs and output paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCommand:
    name: str
    argv: list[str]
    output_path: Path
    log_path: Path
    acceptable_return_codes: tuple[int, ...] = (0,)


def build_benchmark_commands(
    *,
    hotpotqa_input: Path,
    output_root: Path,
    config_path: Path,
    run_real_llm: bool,
) -> list[BenchmarkCommand]:
    logs = output_root / "logs"
    return [
        BenchmarkCommand(
            name="offline_demo",
            argv=[
                sys.executable,
                "-m",
                "graphrag_v2.cli.main",
                "index",
                "examples/docs",
                "--output",
                str(output_root / "offline-demo"),
            ],
            output_path=output_root / "offline-demo",
            log_path=logs / "offline-demo.log",
        ),
        BenchmarkCommand(
            name="offline_qa_eval",
            argv=[
                sys.executable,
                "scripts/evaluate_qa.py",
                "--index",
                str(output_root / "offline-demo"),
                "--questions",
                "examples/eval/qa/questions.jsonl",
                "--format",
                "markdown",
                "--output",
                str(output_root / "offline-demo" / "qa-eval.md"),
            ],
            output_path=output_root / "offline-demo" / "qa-eval.md",
            log_path=logs / "offline-qa-eval.log",
        ),
        BenchmarkCommand(
            name="hotpotqa_mini_25",
            argv=[
                sys.executable,
                "scripts/benchmark_hotpotqa_mini.py",
                "--input",
                str(hotpotqa_input),
                "--output",
                str(output_root / "hotpotqa-mini-dev25"),
                "--sample-size",
                "25",
                "--seed",
                "42",
            ],
            output_path=output_root / "hotpotqa-mini-dev25",
            log_path=logs / "hotpotqa-mini-dev25.log",
            acceptable_return_codes=(0, 1),
        ),
        *(
            [
                BenchmarkCommand(
                    name="hotpotqa_mini_real3",
                    argv=[
                        sys.executable,
                        "scripts/benchmark_hotpotqa_mini.py",
                        "--input",
                        str(hotpotqa_input),
                        "--output",
                        str(output_root / "hotpotqa-mini-dev10-real3"),
                        "--sample-size",
                        "10",
                        "--seed",
                        "42",
                        "--real-llm-smoke-size",
                        "3",
                        "--config",
                        str(config_path),
                    ],
                    output_path=output_root / "hotpotqa-mini-dev10-real3",
                    log_path=logs / "hotpotqa-mini-dev10-real3.log",
                    acceptable_return_codes=(0, 1),
                ),
                BenchmarkCommand(
                    name="hotpotqa_isolated_real20",
                    argv=[
                        sys.executable,
                        "scripts/run_hotpotqa_isolated_benchmark.py",
                        "--input",
                        str(hotpotqa_input),
                        "--output",
                        str(output_root / "hotpotqa-isolated-dev20-real"),
                        "--sample-size",
                        "20",
                        "--seed",
                        "42",
                        "--answerer",
                        "llm",
                        "--config",
                        str(config_path),
                    ],
                    output_path=output_root / "hotpotqa-isolated-dev20-real",
                    log_path=logs / "hotpotqa-isolated-dev20-real.log",
                )
            ]
            if run_real_llm
            else []
        ),
    ]


def run_command(command: BenchmarkCommand) -> int:
    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    with command.log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command.argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def is_successful_exit(command: BenchmarkCommand, code: int) -> bool:
    return code in command.acceptable_return_codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenFusionKGQA benchmark suite.")
    parser.add_argument("--hotpotqa-input", required=True)
    parser.add_argument("--output-root", default="artifacts/benchmark-suite")
    parser.add_argument("--config", default="settings.local.real-llm.yaml")
    parser.add_argument("--real-llm", action="store_true")
    args = parser.parse_args(argv)

    commands = build_benchmark_commands(
        hotpotqa_input=Path(args.hotpotqa_input),
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        run_real_llm=args.real_llm,
    )
    failures = []
    for command in commands:
        code = run_command(command)
        if not is_successful_exit(command, code):
            failures.append((command.name, code, command.log_path))
    for name, code, log_path in failures:
        print(f"{name} failed with exit code {code}; see {log_path}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
