#!/usr/bin/env python3
"""Service-backed release candidate audit."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from graphrag_v2.llm.real_llm_config import resolve_real_llm_settings

SCHEMA_VERSION = "2026-06-07.p14.v1"
DEFAULT_OUTPUT_DIR = Path("artifacts/release-candidate-audit")
TAIL_CHARS = 6000
SENSITIVE_ENV_NAMES = (
    "NEO4J_PASSWORD",
    "ZHIPUAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KGQA_REAL_LLM_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_LLM_API_KEY",
    "KGQA_API_AUTH_TOKEN",
    "GRAPHRAG_API_KEY",
    "GRAPHRAG_EMBEDDING_API_KEY",
)

OFFLINE_COMMAND = "scripts/verify_release.sh"
NEO4J_TEST_COMMAND = (
    "python -m pytest graphrag_v2/tests/integration/test_neo4j_store.py "
    "graphrag_v2/tests/integration/test_community_pipeline.py -q"
)
DOCKER_COMMANDS = (
    "docker build -t openfusion-kgqa:rc-audit .",
    "docker run --rm openfusion-kgqa:rc-audit --help",
    "docker compose up -d neo4j",
    (
        "docker compose run --rm kgqa index examples/docs "
        "--output artifacts/compose-neo4j-demo "
        "--config settings.compose.neo4j.yaml "
        "--graph-store neo4j --community"
    ),
    "docker compose down",
)
REAL_LLM_COMMAND = (
    "KGQA_REAL_LLM_SMOKE=1 python -m pytest "
    "graphrag_v2/tests/integration/test_real_llm_smoke.py -q"
)
API_RUNTIME_COMMAND = "python -m pytest graphrag_v2/tests/unit/test_api_runtime.py -q"
API_RUNTIME_CONTRACTS = (
    "healthz,readyz,ask,metrics,bearer_auth,request_bounds,error_envelope,request_log_redaction"
)


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class GateResult:
    name: str
    status: str
    required: bool
    commands: list[str] = field(default_factory=list)
    reason: str = ""
    elapsed_seconds: float = 0.0
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_text(text: str, sensitive_values: Iterable[str] | None = None) -> str:
    redacted = text
    for value in sensitive_values or ():
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    for env_name in SENSITIVE_ENV_NAMES:
        redacted = re.sub(
            rf"({re.escape(env_name)}\s*=\s*)[^\s]+",
            rf"\1[REDACTED]",
            redacted,
        )
    return redacted


def sensitive_values_from_env(env: Mapping[str, str] | None = None) -> set[str]:
    source = os.environ if env is None else env
    return {source[name] for name in SENSITIVE_ENV_NAMES if source.get(name)}


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def command_to_text(command: Sequence[str], env_prefix: Mapping[str, str] | None = None) -> str:
    prefix = ""
    if env_prefix:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_prefix.items())
        prefix += " "
    return prefix + " ".join(shlex.quote(part) for part in command)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    display_env: Mapping[str, str] | None = None,
) -> CommandResult:
    start = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    sensitive = sensitive_values_from_env(merged_env)
    process = subprocess.run(
        list(command),
        cwd=cwd,
        env=merged_env,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    stdout_tail = redact_text(tail(process.stdout), sensitive)
    stderr_tail = redact_text(tail(process.stderr), sensitive)
    return CommandResult(
        command=command_to_text(command, display_env),
        returncode=process.returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        elapsed_seconds=round(elapsed, 6),
    )


def compute_overall_status(gates: Sequence[GateResult]) -> str:
    if any(gate.status == "failed" for gate in gates):
        return "failed"
    if any(gate.status == "blocked" for gate in gates):
        return "blocked"
    if any(gate.status == "skipped" and gate.required for gate in gates):
        return "blocked"
    return "passed"


def blocked_gate(
    name: str,
    *,
    required: bool,
    reason: str,
    commands: Sequence[str],
    metadata: Mapping[str, str] | None = None,
) -> GateResult:
    return GateResult(
        name=name,
        status="blocked",
        required=required,
        reason=reason,
        commands=list(commands),
        metadata=dict(metadata or {}),
    )


def skipped_gate(
    name: str,
    *,
    required: bool,
    reason: str,
    commands: Sequence[str],
    metadata: Mapping[str, str] | None = None,
) -> GateResult:
    return GateResult(
        name=name,
        status="skipped",
        required=required,
        reason=reason,
        commands=list(commands),
        metadata=dict(metadata or {}),
    )


def gate_from_command(
    name: str,
    *,
    required: bool,
    result: CommandResult,
    reason: str = "",
    metadata: Mapping[str, str] | None = None,
) -> GateResult:
    return GateResult(
        name=name,
        status="passed" if result.returncode == 0 else "failed",
        required=required,
        commands=[result.command],
        reason=reason,
        elapsed_seconds=result.elapsed_seconds,
        returncode=result.returncode,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        metadata=dict(metadata or {}),
    )


def run_offline_gate(root: Path, *, dry_run: bool, skip: bool = False) -> GateResult:
    if skip:
        return skipped_gate(
            "offline",
            required=True,
            reason="disabled by --skip-offline",
            commands=[OFFLINE_COMMAND],
        )
    if dry_run:
        return blocked_gate(
            "offline",
            required=True,
            reason="dry-run does not execute heavyweight gates",
            commands=[OFFLINE_COMMAND],
        )
    result = run_command(
        ["scripts/verify_release.sh"],
        cwd=root,
        env={"KGQA_REAL_LLM_SMOKE": "0"},
    )
    return gate_from_command("offline", required=True, result=result)


def run_api_runtime_gate(root: Path, *, dry_run: bool, skip: bool = False) -> GateResult:
    commands = [API_RUNTIME_COMMAND]
    metadata = {"contracts": API_RUNTIME_CONTRACTS}
    if skip:
        return skipped_gate(
            "api_runtime",
            required=True,
            reason="disabled by --skip-api-runtime",
            commands=commands,
            metadata=metadata,
        )
    if dry_run:
        return blocked_gate(
            "api_runtime",
            required=True,
            reason="dry-run does not execute audit gates",
            commands=commands,
            metadata=metadata,
        )
    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "graphrag_v2/tests/unit/test_api_runtime.py",
            "-q",
        ],
        cwd=root,
        env={"KGQA_REAL_LLM_SMOKE": "0"},
    )
    return gate_from_command(
        "api_runtime",
        required=True,
        result=result,
        metadata=metadata,
    )


def run_neo4j_gate(root: Path, *, dry_run: bool, skip: bool = False) -> GateResult:
    helper = root / "scripts" / "start_fresh_neo4j.sh"
    commands = ["scripts/start_fresh_neo4j.sh", NEO4J_TEST_COMMAND]
    if skip:
        return skipped_gate(
            "neo4j",
            required=True,
            reason="disabled by --skip-neo4j",
            commands=commands,
        )
    if dry_run:
        return blocked_gate(
            "neo4j",
            required=True,
            reason="dry-run does not execute heavyweight gates",
            commands=commands,
        )
    if not helper.exists():
        return blocked_gate(
            "neo4j",
            required=True,
            reason="scripts/start_fresh_neo4j.sh is not available",
            commands=commands,
        )

    start = time.monotonic()
    start_result = run_command(
        ["scripts/start_fresh_neo4j.sh"],
        cwd=root,
        env={
            "KGQA_NEO4J_HOME": "",
            "KGQA_NEO4J_INSTANCE_NAME": "",
            "KGQA_NEO4J_BOLT_PORT": "",
            "KGQA_NEO4J_HTTP_PORT": "",
        },
    )
    if start_result.returncode != 0:
        return GateResult(
            name="neo4j",
            status="failed",
            required=True,
            commands=[start_result.command],
            reason="fresh Neo4j startup failed",
            elapsed_seconds=start_result.elapsed_seconds,
            returncode=start_result.returncode,
            stdout_tail=start_result.stdout_tail,
            stderr_tail=start_result.stderr_tail,
        )

    env_file = _parse_neo4j_env_file(start_result.stdout_tail)
    if not env_file:
        return GateResult(
            name="neo4j",
            status="failed",
            required=True,
            commands=[start_result.command],
            reason="fresh Neo4j helper did not report env_file",
            elapsed_seconds=start_result.elapsed_seconds,
            returncode=start_result.returncode,
            stdout_tail=start_result.stdout_tail,
            stderr_tail=start_result.stderr_tail,
        )

    neo4j_env = _read_export_env(Path(env_file))
    test_result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "graphrag_v2/tests/integration/test_neo4j_store.py",
            "graphrag_v2/tests/integration/test_community_pipeline.py",
            "-q",
        ],
        cwd=root,
        env=neo4j_env,
    )
    _stop_neo4j(root, neo4j_env)
    elapsed = time.monotonic() - start
    return GateResult(
        name="neo4j",
        status="passed" if test_result.returncode == 0 else "failed",
        required=True,
        commands=[start_result.command, test_result.command],
        reason="" if test_result.returncode == 0 else "Neo4j integration tests failed",
        elapsed_seconds=round(elapsed, 6),
        returncode=test_result.returncode,
        stdout_tail=test_result.stdout_tail,
        stderr_tail=test_result.stderr_tail,
    )


def _parse_neo4j_env_file(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip().startswith("env_file:"):
            return line.split(":", 1)[1].strip()
    match = re.search(r'source\s+"([^"]*kgqa-neo4j\.env)"', output)
    if match:
        return match.group(1)
    return None


def _read_export_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export ") :].split("=", 1)
        env[key] = value.strip().strip('"')
    return env


def _stop_neo4j(root: Path, env: Mapping[str, str]) -> None:
    home = env.get("KGQA_NEO4J_HOME")
    if not home:
        return
    neo4j_home = Path(home)
    neo4j_binary = neo4j_home / "bin" / "neo4j"
    if not neo4j_binary.exists():
        return
    stop_env = dict(env)
    if "JAVA_HOME" in os.environ:
        stop_env["JAVA_HOME"] = os.environ["JAVA_HOME"]
    elif Path("/home/u2023312337/CoMaGRAG/external_tools/java17").exists():
        stop_env["JAVA_HOME"] = "/home/u2023312337/CoMaGRAG/external_tools/java17"
    stop_env["NEO4J_HOME"] = str(neo4j_home)
    stop_env["NEO4J_CONF"] = str(neo4j_home / "conf")
    run_command([str(neo4j_binary), "stop"], cwd=root, env=stop_env)


def run_docker_gate(
    root: Path,
    *,
    dry_run: bool,
    skip: bool = False,
    include: bool = False,
) -> GateResult:
    if skip:
        return skipped_gate(
            "docker",
            required=False,
            reason="disabled by --skip-docker",
            commands=DOCKER_COMMANDS,
        )
    if not include:
        return skipped_gate(
            "docker",
            required=False,
            reason="excluded from current release scope",
            commands=DOCKER_COMMANDS,
        )
    if dry_run:
        return blocked_gate(
            "docker",
            required=False,
            reason="dry-run does not execute heavyweight gates",
            commands=DOCKER_COMMANDS,
        )
    if shutil.which("docker") is None:
        return blocked_gate(
            "docker",
            required=False,
            reason="docker binary is not available",
            commands=DOCKER_COMMANDS,
        )
    info = run_command(["docker", "info"], cwd=root)
    if info.returncode != 0:
        return GateResult(
            name="docker",
            status="blocked",
            required=False,
            commands=["docker info", *DOCKER_COMMANDS],
            reason="docker daemon is not available",
            elapsed_seconds=info.elapsed_seconds,
            returncode=info.returncode,
            stdout_tail=info.stdout_tail,
            stderr_tail=info.stderr_tail,
        )
    compose = run_command(["docker", "compose", "version"], cwd=root)
    if compose.returncode != 0:
        return GateResult(
            name="docker",
            status="blocked",
            required=False,
            commands=["docker compose version", *DOCKER_COMMANDS],
            reason="docker compose is not available",
            elapsed_seconds=compose.elapsed_seconds,
            returncode=compose.returncode,
            stdout_tail=compose.stdout_tail,
            stderr_tail=compose.stderr_tail,
        )

    return _run_docker_commands(root)


def _run_docker_commands(root: Path) -> GateResult:
    start = time.monotonic()
    command_specs = (
        ["docker", "build", "-t", "openfusion-kgqa:rc-audit", "."],
        ["docker", "run", "--rm", "openfusion-kgqa:rc-audit", "--help"],
        ["docker", "compose", "up", "-d", "neo4j"],
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "kgqa",
            "index",
            "examples/docs",
            "--output",
            "artifacts/compose-neo4j-demo",
            "--config",
            "settings.compose.neo4j.yaml",
            "--graph-store",
            "neo4j",
            "--community",
        ],
    )
    commands: list[str] = []
    last_result: CommandResult | None = None
    status = "passed"
    reason = ""
    try:
        for command in command_specs:
            result = run_command(command, cwd=root)
            commands.append(result.command)
            last_result = result
            if result.returncode != 0:
                status = "failed"
                reason = f"command failed: {result.command}"
                break
    finally:
        down = run_command(["docker", "compose", "down"], cwd=root)
        commands.append(down.command)
    elapsed = time.monotonic() - start
    return GateResult(
        name="docker",
        status=status,
        required=False,
        commands=commands,
        reason=reason,
        elapsed_seconds=round(elapsed, 6),
        returncode=last_result.returncode if last_result else None,
        stdout_tail=last_result.stdout_tail if last_result else "",
        stderr_tail=last_result.stderr_tail if last_result else "",
    )


def run_real_llm_gate(root: Path, *, dry_run: bool, skip: bool = False) -> GateResult:
    commands = [REAL_LLM_COMMAND]
    settings = resolve_real_llm_settings()
    metadata = settings.metadata()
    if skip:
        return skipped_gate(
            "real_llm",
            required=False,
            reason="disabled by --skip-real-llm",
            commands=commands,
            metadata=metadata,
        )
    if dry_run:
        return blocked_gate(
            "real_llm",
            required=False,
            reason="dry-run does not execute heavyweight gates",
            commands=commands,
            metadata=metadata,
        )
    if os.getenv("KGQA_REAL_LLM_SMOKE") != "1":
        return blocked_gate(
            "real_llm",
            required=False,
            reason="KGQA_REAL_LLM_SMOKE=1 is not set",
            commands=commands,
            metadata=metadata,
        )
    blocker = settings.blocker_reason()
    if blocker:
        return blocked_gate(
            "real_llm",
            required=False,
            reason=blocker,
            commands=commands,
            metadata=metadata,
        )
    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "graphrag_v2/tests/integration/test_real_llm_smoke.py",
            "-q",
        ],
        cwd=root,
        env=settings.runtime_env(),
        display_env=settings.safe_display_env(),
    )
    return gate_from_command("real_llm", required=False, result=result, metadata=metadata)


def detect_environment(root: Path) -> dict[str, str]:
    settings = resolve_real_llm_settings()
    docker_state = "missing"
    if shutil.which("docker") is not None:
        docker_state = "available"
        info = run_command(["docker", "info"], cwd=root)
        if info.returncode != 0:
            docker_state = "daemon_unavailable"
    compose_state = "missing"
    if shutil.which("docker") is not None:
        compose = run_command(["docker", "compose", "version"], cwd=root)
        compose_state = "available" if compose.returncode == 0 else "missing"
    helper = root / "scripts" / "start_fresh_neo4j.sh"
    return {
        "docker": docker_state,
        "docker_compose": compose_state,
        "neo4j_helper": "available" if helper.exists() else "missing",
        "real_llm_opt_in": "set" if os.getenv("KGQA_REAL_LLM_SMOKE") == "1" else "unset",
        "real_llm_provider": settings.provider,
        "real_llm_model": settings.model,
        "real_llm_api_base": settings.api_base,
        "real_llm_config_path": settings.config_path,
        "real_llm_api_key_env": settings.api_key_env,
        "real_llm_api_key": "set" if settings.api_key_set else "unset",
    }


def build_report(
    gates: Sequence[GateResult],
    environment: Mapping[str, str],
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict:
    started = started_at or utc_now()
    finished = finished_at or utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_started_at": started,
        "audit_finished_at": finished,
        "overall_status": compute_overall_status(gates),
        "gates": [asdict(gate) for gate in gates],
        "environment": dict(environment),
        "report_paths": {
            "json": "report.json",
            "markdown": "report.md",
        },
    }


def write_reports(report: Mapping, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_markdown(report: Mapping) -> str:
    lines = [
        "# Release Candidate Audit",
        "",
        f"- Schema version: `{report['schema_version']}`",
        f"- Overall status: `{report['overall_status']}`",
        f"- Started: `{report['audit_started_at']}`",
        f"- Finished: `{report['audit_finished_at']}`",
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Required | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for gate in report["gates"]:
        required = "yes" if gate["required"] else "no"
        reason = gate.get("reason") or ""
        lines.append(
            f"| {gate['name']} | {gate['status']} | {required} | {reason} |"
        )
    lines.extend(["", "## Environment", ""])
    for key, value in report["environment"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gate Details", ""])
    for gate in report["gates"]:
        lines.extend(
            [
                f"### {gate['name']}",
                "",
                f"- Status: `{gate['status']}`",
                f"- Required: `{'yes' if gate['required'] else 'no'}`",
                f"- Return code: `{gate['returncode']}`",
                f"- Elapsed seconds: `{gate['elapsed_seconds']}`",
                "- Commands:",
            ]
        )
        for command in gate.get("commands", []):
            lines.append(f"  - `{command}`")
        if gate.get("metadata"):
            lines.extend(["", "- Metadata:"])
            for key, value in gate["metadata"].items():
                lines.append(f"  - `{key}`: `{value}`")
        if gate.get("stdout_tail"):
            lines.extend(["", "Stdout tail:", "", "```text", gate["stdout_tail"], "```"])
        if gate.get("stderr_tail"):
            lines.extend(["", "Stderr tail:", "", "```text", gate["stderr_tail"], "```"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_audit(args: argparse.Namespace) -> tuple[int, dict, dict[str, Path]]:
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    started = utc_now()
    gates = [
        run_offline_gate(root, dry_run=args.dry_run, skip=args.skip_offline),
        run_api_runtime_gate(
            root,
            dry_run=args.dry_run,
            skip=args.skip_api_runtime,
        ),
        run_neo4j_gate(root, dry_run=args.dry_run, skip=args.skip_neo4j),
        run_docker_gate(
            root,
            dry_run=args.dry_run,
            skip=args.skip_docker,
            include=args.include_docker,
        ),
        run_real_llm_gate(root, dry_run=args.dry_run, skip=args.skip_real_llm),
    ]
    report = build_report(
        gates,
        environment=detect_environment(root),
        started_at=started,
        finished_at=utc_now(),
    )
    paths = write_reports(report, output)
    exit_code = 1 if report["overall_status"] == "failed" else 0
    return exit_code, report, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run service-backed release candidate audit gates."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Audit output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a report without executing heavyweight gates.",
    )
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-api-runtime", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-real-llm", action="store_true")
    parser.add_argument(
        "--include-docker",
        action="store_true",
        help="Run Docker/compose gates. Docker is excluded from the current release scope by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, report, paths = run_audit(args)
    print(f"Release candidate audit status: {report['overall_status']}")
    print(f"JSON report: {paths['json']}")
    print(f"Markdown report: {paths['markdown']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
