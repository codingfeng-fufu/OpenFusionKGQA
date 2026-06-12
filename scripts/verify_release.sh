#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_INDEX="${KGQA_RELEASE_SMOKE_INDEX:-artifacts/release-smoke}"

run() {
  printf '\n+ %s\n' "$*"
  "$@"
}

if [ "${KGQA_REAL_LLM_SMOKE:-}" = "1" ]; then
  printf 'ERROR: KGQA_REAL_LLM_SMOKE=1 is for optional real API smoke, not release verification.\n' >&2
  exit 1
fi

unset NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD NEO4J_DATABASE

run python scripts/security_check.py

run python -m pip install -e ".[dev]"

run kgqa --help
run kgqa index --help
run kgqa ask --help
run kgqa inspect --help

rm -rf "$SMOKE_INDEX"
run kgqa index examples/docs --output "$SMOKE_INDEX"
run kgqa ask "GraphRAG 是什么？" --index "$SMOKE_INDEX"
run kgqa inspect graph --index "$SMOKE_INDEX"
run kgqa inspect run --index "$SMOKE_INDEX"

test -f "$SMOKE_INDEX/run_events.jsonl"
test -f "$SMOKE_INDEX/run_summary.json"

run scripts/evaluate_qa.py --index "$SMOKE_INDEX" --questions examples/eval/qa/questions.jsonl

run python -m pytest graphrag_v2/tests -q

printf '\nRelease verification passed.\n'
