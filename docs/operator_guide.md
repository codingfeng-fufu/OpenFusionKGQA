# Operator Guide

This guide is the handoff entry point for running OpenFusionKGQA locally. The
project is a `0.2.0-beta.1` prototype with local release-readiness gates; Docker
and compose are validation tools, not a hosted production deployment.

## Installation

Use Python 3.11+ from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Verify the default offline test suite:

```bash
python -m pytest graphrag_v2/tests -q
```

Current local baseline: `459 passed, 4 skipped`.

## Configuration

Create a starter settings file when you need non-default graph store or model
settings:

```bash
kgqa init --output settings.yaml
```

Keep local credentials in ignored local settings files. For real LLM smoke,
use `settings.local.real-llm.yaml` and point `KGQA_REAL_LLM_CONFIG` at that
file. Do not commit `.env*`, `settings.local.*`, generated artifacts, local
logs, or Docker volumes.

Common environment variables:

- `NEO4J_PASSWORD`: Neo4j password for local graph store validation.
- `KGQA_REAL_LLM_CONFIG`: ignored YAML config for real LLM smoke, usually `settings.local.real-llm.yaml`.
- `DEEPSEEK_API_KEY`: GitHub Actions secret used only to write the ignored DeepSeek smoke config.
- `KGQA_REAL_LLM_PROVIDER`: temporary provider override; defaults to the config provider or `deepseek`.
- `KGQA_REAL_LLM_MODEL`: temporary model override; defaults to the config model or `deepseek-v4-flash`.
- `KGQA_REAL_LLM_API_BASE`: temporary OpenAI-compatible base URL override.
- `KGQA_REAL_LLM_API_KEY`: temporary generic real LLM API key fallback.
- `ZHIPUAI_API_KEY`: legacy GLM/Zhipu temporary fallback when `llm_provider` is `glm`.
- `OPENAI_API_KEY`: OpenAI-compatible endpoints.
- `LOCAL_LLM_API_KEY`: local/vLLM-compatible endpoints when required.
- `KGQA_REAL_LLM_SMOKE=1`: explicit opt-in for real LLM smoke tests.

## Architecture And Artifact Lifecycle

```text
Documents -> chunks -> extraction -> fusion -> graph store -> QA
```

Default offline runs use JSON artifacts as the graph store. Service-backed
validation can write the same fused graph to Neo4j, where `index_id` isolates
one output directory from other graph data.

Artifact lifecycle:

1. `kgqa index` starts a new run, clears generated artifacts in the selected
   output path, and records run start metadata.
2. Document loading writes `document_scan.json` and `text_units.parquet`.
3. Extraction writes candidate entity, relationship, and triple artifacts.
4. Fusion writes canonical entities, relationships, rejected triples, and
   graph-store metadata.
5. Graph-store writing persists `graph.json` for JSON runs or scoped Neo4j
   nodes/relationships for Neo4j runs.
6. Run observability writes `run_events.jsonl`, `run_summary.json`, and links
   both paths from `index_metadata.json`.
7. `kgqa ask`, `kgqa inspect graph`, and `kgqa inspect run` consume these
   artifacts as stable operator-facing contracts.

## Offline Indexing

The default path needs no API key and no Neo4j:

```bash
kgqa index examples/docs --output artifacts/demo
kgqa inspect graph --index artifacts/demo
kgqa inspect run --index artifacts/demo
```

The run inspection command reads `index_metadata.json`, `run_events.jsonl`, and
`run_summary.json` to show run status, stage timings, failed-stage context, and
artifact paths.

Stage-specific debugging modes:

```bash
kgqa index examples/docs --output artifacts/docs --mode documents-only
kgqa index examples/docs --output artifacts/extraction --mode extraction-only
kgqa index examples/docs --output artifacts/fusion --mode fusion-only
```

## QA

Ask against an existing index:

```bash
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo --format json
```

Answers include graph evidence, community evidence, source citations, refusal
metadata, prompt version metadata, and query trace metadata. When no source text
evidence is available, QA refuses instead of producing an uncited answer.

## Neo4j Operations

Neo4j is the production graph store path for local validation. Start a fresh
local instance with:

```bash
scripts/start_fresh_neo4j.sh
source "$(ls -td ~/.local/share/kgqa-neo4j-openfusion-*/kgqa-neo4j.env | head -1)"
```

Run an indexed Neo4j path:

```bash
kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --graph-store neo4j \
  --strict-neo4j \
  --community

kgqa inspect graph --index artifacts/neo4j-demo --graph-store neo4j
```

Operational boundaries:

- Backup: use Neo4j-supported database backup/export tooling and record
  `kgqa inspect graph --graph-store neo4j` output with the backup.
- Restore: restore with Neo4j tooling, then inspect the target `index_id` and
  confirm `health_status: ready`, schema readiness, and counts.
- Rebuild: rerun `kgqa index ... --graph-store neo4j --strict-neo4j` against
  the same output directory. Staged replace preserves the previous successful
  index until promotion succeeds.
- Cleanup: there is no cleanup CLI yet. Delete old index scopes only after
  confirming the `index_id` is not referenced by active artifacts or consumers.

## QA Evaluation

Run the deterministic offline QA gate:

```bash
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl
```

Use thresholds for release gating:

```bash
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl \
  --min-route-accuracy 1.0 \
  --min-retrieval-hit-rate 1.0 \
  --min-citation-coverage 1.0 \
  --min-refusal-accuracy 1.0 \
  --min-citation-grounding-rate 1.0
```

For a service-backed QA gate, source the fresh Neo4j env file and pass the same
runtime config used for indexing:

```bash
source "$(ls -td ~/.local/share/kgqa-neo4j-openfusion-*/kgqa-neo4j.env | head -1)"

scripts/evaluate_qa.py \
  --index artifacts/llm-neo4j-smoke \
  --questions examples/eval/qa/questions.jsonl \
  --config settings.local.real-llm.yaml \
  --strict-neo4j \
  --answerer llm \
  --min-route-accuracy 1.0 \
  --min-retrieval-hit-rate 0.8 \
  --min-citation-grounding-rate 1.0
```

## Benchmark Commands

Run the low-cost offline checks first:

```bash
kgqa index examples/docs --output artifacts/demo
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl

scripts/benchmark_hotpotqa_mini.py \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --output artifacts/hotpotqa-mini \
  --sample-size 25 \
  --seed 42
```

Run a real LLM answerer smoke only after confirming `settings.local.real-llm.yaml`
contains a valid ignored API key:

```bash
python scripts/run_hotpotqa_isolated_benchmark.py \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --output artifacts/hotpotqa-isolated-dev20-real \
  --sample-size 20 \
  --seed 42 \
  --answerer llm \
  --config settings.local.real-llm.yaml
```

Run the 100-case beta gate after the 20-case smoke has zero runtime errors:

```bash
python scripts/run_hotpotqa_isolated_benchmark.py \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --output artifacts/hotpotqa-isolated-dev100-real \
  --sample-size 100 \
  --seed 42 \
  --answerer llm \
  --config settings.local.real-llm.yaml

python scripts/summarize_benchmark_reports.py \
  artifacts/hotpotqa-isolated-dev100-real/hotpotqa-isolated-report.json \
  --output artifacts/hotpotqa-isolated-dev100-real/benchmark-summary-gated.json \
  --min-em 0.60 \
  --min-f1 0.65 \
  --min-support-recall 0.95
```

Run a small real extraction smoke when API cost and runtime are acceptable:

```bash
python scripts/benchmark_support_real_extract.py \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --benchmark HotpotQA \
  --output artifacts/hotpotqa-support-real-extract-dev5 \
  --sample-size 5 \
  --seed 42 \
  --config settings.local.real-llm.yaml \
  --answerer llm
```

Interpretation guideline: gold-support HotpotQA isolated runs primarily test
retrieval, citation, and answer selection. Support real extraction runs also test
LLM extraction robustness and are expected to be noisier. DeepSeek uses standard
JSON response mode by default for extraction; if a provider rejects that mode,
set `model_supports_json: false` in the model config and treat parse failures as
an explicit benchmark metric.

## Release Verification

Run the local offline release gate:

```bash
scripts/verify_release.sh
```

The gate runs `python scripts/security_check.py`, installs the package, checks
CLI help, performs offline indexing and QA smoke, runs `kgqa inspect graph`,
runs `kgqa inspect run`, verifies run observability artifacts, runs QA eval, and
runs the full test suite.

## Release Candidate Audit

Run the service-backed release candidate audit when preparing an external
release review:

```bash
scripts/audit_release_candidate.py
```

The audit writes:

- `artifacts/release-candidate-audit/report.json`
- `artifacts/release-candidate-audit/report.md`

Gate statuses are explicit:

- `passed`: command ran and exited successfully.
- `failed`: command ran and exited unsuccessfully.
- `skipped`: an operator explicitly disabled the gate, or the gate is
  `excluded from current release scope`.
- `blocked`: a required local dependency, service, credential, or opt-in is
  missing.

The required `api_runtime` gate runs
`python -m pytest graphrag_v2/tests/unit/test_api_runtime.py -q` and verifies
the FastAPI `/healthz`, `/readyz`, `/metrics`, `/ask`, Bearer token,
request-length, stable error-envelope, request metrics, and request-log
redaction contracts with a local mock/json index. Docker/compose is
`excluded from current release scope` and is reported as `skipped` by default.
It can still be run manually with
`scripts/audit_release_candidate.py --include-docker`, but it is not a current
release candidate blocker.

Current local P18 audit scope from 2026-06-08: required `offline`,
`api_runtime`, and `neo4j` gates are part of the report; `docker` is `skipped`
because it is `excluded from current release scope`; `real_llm` is blocked until
`KGQA_REAL_LLM_SMOKE=1` and the ignored
`KGQA_REAL_LLM_CONFIG=settings.local.real-llm.yaml` file defines an API key. The
default model is `deepseek-v4-flash`; edit the config file's `model_provider`,
`model`, `api_base`, and `api_key` fields to run another OpenAI-compatible
provider. `DEEPSEEK_API_KEY` is used only by the optional GitHub Actions step
that writes this config file; `KGQA_REAL_LLM_API_KEY` remains a temporary
fallback. Legacy GLM is still available with `model_provider: zhipu` and
`llm_provider: glm`.

Current readiness boundary: this project is a production-path beta, not production-ready.

## Production Runtime API

P17 keeps the minimal service runtime around the existing QA path and adds
request hardening for local production-path validation:

```bash
export KGQA_API_INDEX_PATH=artifacts/demo
uvicorn graphrag_v2.api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /healthz`: process health.
- `GET /readyz`: index/config/graph-store readiness.
- `GET /metrics`: process-local request counters, status codes, latency sums,
  and API error-type counters for supervisor/monitoring integration.
- `POST /ask`: graph-grounded QA using the configured index.

Runtime env:

- `KGQA_API_INDEX_PATH`: required index/artifact path for readiness and `/ask`.
- `KGQA_API_CONFIG`: optional YAML/JSON config path.
- `KGQA_API_ANSWERER`: `mock` or `llm`; defaults to `mock`.
- `KGQA_API_STRICT_NEO4J`: set `1` to fail instead of falling back from Neo4j
  to local artifacts.
- `KGQA_API_AUTH_TOKEN`: optional bearer token for `/readyz` and `/ask`.
  Leave unset only for trusted local development.
- `KGQA_API_MAX_QUESTION_CHARS`: maximum `/ask` question length; defaults to
  `2048`.

When `KGQA_API_AUTH_TOKEN` is set, call protected endpoints with:

```bash
curl -H "Authorization: Bearer $KGQA_API_AUTH_TOKEN" \
  http://127.0.0.1:8000/readyz
```

Supervisor and monitoring contract:

- Use `/healthz` for process liveness.
- Use `/readyz` for dependency readiness before routing traffic.
- Scrape `/metrics` for 5xx/error-type counters and latency trends.
- Alert on sustained readiness failure, elevated 5xx, or latency growth.

Run the local API runtime live check against an already-started process:

```bash
scripts/check_api_runtime.py --base-url http://127.0.0.1:8000
```

When `KGQA_API_AUTH_TOKEN` is set, the script reads it from the environment or
accepts `--token`. It writes a stable JSON report and exits `0` only when
`/healthz`, `/readyz`, and `/metrics` pass. This is a local live-check contract
for supervisor/monitoring integration; it is not a hosted supervisor.

Run the local API runtime latency smoke against an already-started process:

```bash
scripts/benchmark_api_runtime.py \
  --base-url http://127.0.0.1:8000 \
  --path /healthz \
  --requests 50 \
  --concurrency 5 \
  --max-p95-ms 500 \
  --min-success-rate 0.99
```

Use `--path /readyz` or `--path /metrics` for other GET endpoints. Use
`--ask-question "GraphRAG 是什么？"` only when the target index is ready and the
operator accepts the extra QA workload. The benchmark writes stable JSON with
request count, success rate, p50/p95/p99 latency, error counts, and threshold
results. This is a local latency/capacity smoke, not a full production load test
or real LLM cost envelope.

API errors use a stable JSON envelope with `status`, `error_type`, `error`, and
`request_id`. Readiness failures on `/readyz` still return the readiness report
shape because operators consume its `checks` object directly.

The API emits structured JSON request logs with request id, path, status code,
and latency. Secrets are redacted through the same allowlist used by run
observability.

## Security And Data Governance

Run the security gate directly when reviewing a release candidate:

```bash
python scripts/security_check.py
```

Generated artifacts may contain source text, graph data, model outputs, failed
response snippets, QA reports, run logs, and review queues. Perform artifact retention review
and model output storage review before sharing artifacts or packaging release assets.

Before release, record dependency vulnerability review and license review for
runtime dependencies, development dependencies, and sample documents.

## Troubleshooting

- Missing input or config errors should return CLI exit code `2`; unexpected
  crashes return exit code `1`.
- Use `kgqa inspect run --index <path>` first for failed indexing runs. It shows
  failed stage, timing, event count, and run artifact paths.
- Use `index_metadata.json` when a run summary is missing; `kgqa inspect run`
  falls back to metadata-derived output.
- For Neo4j failures, rerun `kgqa inspect graph --graph-store neo4j` and verify
  `health_status`, `schema_ready`, missing schema lists, database, and counts.
- For real LLM paths, confirm `KGQA_REAL_LLM_CONFIG` points to an ignored
  config file with provider/model/API base/key values, and that
  `KGQA_REAL_LLM_SMOKE=1` is intentional.

## Known Limits

- The stable default extractor and answerer are mock implementations for
  deterministic local validation.
- Real LLM extraction and answering are explicit opt-in paths and require
  configured providers.
- Docker/compose files are excluded from current release scope. They remain
  optional local validation tools, not production deployment automation.
- Automatic source-text PII detection, online vulnerability scanning, legal
  license classification, hosted secret management, frontend UI, vector
  database integration, distributed indexing, and managed cloud deployment are
  not included.
