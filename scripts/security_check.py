#!/usr/bin/env python3
"""Offline security and data-governance release checks."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_GITIGNORE_PATTERNS = (
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
)

REQUIRED_REDACTION_ENV_NAMES = (
    "NEO4J_PASSWORD",
    "ZHIPUAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KGQA_REAL_LLM_API_KEY",
    "GRAPHRAG_API_KEY",
    "GRAPHRAG_EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_LLM_API_KEY",
)
IGNORED_LOCAL_SETTINGS_FILES = (
    "settings.local.yaml",
    "settings.local.yml",
    "settings.local.json",
)
IGNORED_LOCAL_SETTINGS_PATTERNS = (
    "settings.local.*.yaml",
    "settings.local.*.yml",
    "settings.local.*.json",
)

GOVERNANCE_DOC_REQUIREMENTS = {
    "docs/operator_guide.md": (
        "artifact retention",
        "model output storage",
        "dependency vulnerability review",
        "license review",
    ),
    "DEPLOYMENT.md": (
        "generated artifacts may contain source text",
        "model outputs require retention review",
        "do not index sensitive data without governance review",
    ),
    "docs/artifacts.md": (
        "generated artifacts may contain source text",
        "artifact retention",
        "python scripts/security_check.py",
    ),
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "dist",
    "htmlcov",
    "kgqa_graphrag.egg-info",
    "logs",
    "output",
    "results",
}
PLACEHOLDER_MARKERS = (
    "dummy",
    "example",
    "kgqa-local",
    "local",
    "neo4j",
    "placeholder",
    "replace-me",
    "sample",
    "test",
    "your-",
)
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_-]*)"
    r"\s*[:=]\s*[\"']?([^\"'\s#]+)"
)
HIGH_RISK_SECRET_PREFIXES = (
    "akia",
    "asia",
    "ghp_",
    "github_pat_",
    "sk-",
    "sk_",
    "xox",
)


@dataclass
class CheckResult:
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def add(self, message: str) -> None:
        self.failures.append(message)

    def format(self) -> str:
        if self.ok:
            return "Security check passed."
        return "Security check failed:\n" + "\n".join(
            f"- {failure}" for failure in self.failures
        )


def run_checks(root: str | Path) -> CheckResult:
    root_path = Path(root)
    result = CheckResult()

    _check_gitignore(root_path, result)
    _check_redaction_allowlist(root_path, result)
    _check_governance_docs(root_path, result)
    _check_committed_text_for_credentials(root_path, result)

    return result


def _check_gitignore(root: Path, result: CheckResult) -> None:
    path = root / ".gitignore"
    if not path.exists():
        result.add(".gitignore is missing.")
        return
    lines = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in lines]
    if missing:
        result.add(f".gitignore is missing required patterns: {', '.join(missing)}")


def _check_redaction_allowlist(root: Path, result: CheckResult) -> None:
    path = root / "graphrag_v2" / "artifacts" / "run_observability.py"
    if not path.exists():
        result.add("redaction allowlist source is missing.")
        return
    actual_names = _load_sensitive_env_names(path, result)
    missing = [name for name in REQUIRED_REDACTION_ENV_NAMES if name not in actual_names]
    if missing:
        result.add(
            "redaction allowlist is missing env names: " + ", ".join(missing)
        )


def _check_governance_docs(root: Path, result: CheckResult) -> None:
    for relative_path, required_phrases in GOVERNANCE_DOC_REQUIREMENTS.items():
        path = root / relative_path
        if not path.exists():
            result.add(f"governance doc {relative_path} is missing.")
            continue
        text = path.read_text(encoding="utf-8").lower()
        missing = [phrase for phrase in required_phrases if phrase not in text]
        if missing:
            result.add(
                f"governance doc {relative_path} is missing: {', '.join(missing)}"
            )


def _check_committed_text_for_credentials(root: Path, result: CheckResult) -> None:
    for path in _iter_text_files(root):
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = CREDENTIAL_ASSIGNMENT_RE.search(line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            if not _is_sensitive_key(key):
                continue
            if _is_allowed_placeholder(value):
                continue
            if not _looks_like_hardcoded_credential(value):
                continue
            result.add(
                f"credential-like assignment in {relative}:{line_number}; "
                "use env vars or documented placeholders."
            )


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if _is_ignored_local_settings_file(path):
            continue
        if (
            path.suffix.lower() not in TEXT_SUFFIXES
            and path.name not in {".env", ".env.example", ".gitignore"}
        ):
            continue
        yield path


def _is_ignored_local_settings_file(path: Path) -> bool:
    name = path.name
    return name in IGNORED_LOCAL_SETTINGS_FILES or any(
        fnmatch.fnmatch(name, pattern) for pattern in IGNORED_LOCAL_SETTINGS_PATTERNS
    )


def _load_sensitive_env_names(path: Path, result: CheckResult) -> set[str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        result.add(f"redaction allowlist source cannot be parsed: {exc}")
        return set()

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "SENSITIVE_ENV_NAMES"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as exc:
            result.add(f"redaction allowlist is not a literal tuple/list: {exc}")
            return set()
        if not isinstance(value, (tuple, list, set)):
            result.add("redaction allowlist must be a literal tuple/list/set.")
            return set()
        return {str(item) for item in value}

    result.add("redaction allowlist SENSITIVE_ENV_NAMES is missing.")
    return set()


def _is_allowed_placeholder(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    exact_names = {"api_key", "token", "secret", "password"}
    return normalized in exact_names or any(
        normalized.endswith(suffix)
        for suffix in ("_api_key", "_token", "_secret", "_password")
    )


def _looks_like_hardcoded_credential(value: str) -> bool:
    normalized = value.strip().strip(",;").lower()
    if len(normalized) < 16:
        return False
    if normalized.startswith(("getattr(", "os.getenv(", "config.", "self.")):
        return False
    if normalized.startswith(("${{", "$", "%")):
        return False
    if "(" in normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in HIGH_RISK_SECRET_PREFIXES):
        return True
    if any(word in normalized for word in ("secret", "token", "password")):
        return True
    return bool(re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z0-9_/-]{32,}", value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline security and data-governance release checks."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        help="Repository root to check. Defaults to the current project root.",
    )
    args = parser.parse_args(argv)

    result = run_checks(Path(args.root))
    print(result.format())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
