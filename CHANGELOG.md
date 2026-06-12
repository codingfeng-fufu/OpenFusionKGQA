# Changelog

All notable changes to this project are documented here.

## Unreleased

### Project Naming

- Renamed the public project identity to `OpenFusionKGQA`.
- Kept the `kgqa` CLI, `KGQA_*` environment variables, and `graphrag_v2`
  package path stable for compatibility.

### P21 API Runtime Load And Latency Profile

- Added `scripts/benchmark_api_runtime.py --base-url <url>` as a local API
  runtime latency/capacity smoke for an already-started FastAPI process.
- The benchmark supports `--requests`, `--concurrency`, `--max-p95-ms`,
  `--min-success-rate`, Bearer auth via `KGQA_API_AUTH_TOKEN` or `--token`, and
  optional `POST /ask` benchmarking through `--ask-question`.
- The report is stable JSON with request count, success rate, p50/p95/p99
  latency, error counts, and threshold results; token values are not written to
  the report.
- Documented that this local benchmark is not a full production load test or
  real LLM cost envelope.

### P20 API Runtime Operations Profile

- Added `scripts/check_api_runtime.py --base-url <url>` as a local API runtime
  live check for an already-started FastAPI process.
- The live check validates `/healthz`, `/readyz`, and `/metrics`, supports
  `KGQA_API_AUTH_TOKEN` or `--token`, writes stable JSON, and returns exit code
  `0` only when all checks pass.
- Documented the live-check command for supervisor/monitoring integration while
  keeping hosted supervisor, metrics backend, alerting, and incident response as
  remaining production work.

### P19 API Observability And Supervisor Contract

- Added `GET /metrics` to the FastAPI runtime with process-local request
  counters, status-code labels, latency sums, and API error-type counters.
- Protected `/metrics` with `KGQA_API_AUTH_TOKEN` when API auth is configured,
  while keeping `/healthz` open for process liveness checks.
- Extended structured request logging to feed the in-memory metrics collector.
- Extended the required `api_runtime` release-audit gate metadata to cover the
  metrics contract.
- Documented the local supervisor/monitoring contract: `/healthz` for liveness,
  `/readyz` for dependency readiness, and `/metrics` for status/error/latency
  monitoring.

### P18 API Runtime Release Gate

- Added a required `api_runtime` gate to `scripts/audit_release_candidate.py`.
- The gate runs `python -m pytest graphrag_v2/tests/unit/test_api_runtime.py -q`
  to verify `/healthz`, `/readyz`, `/ask`, Bearer auth, request-size limits,
  stable error envelopes, and request-log redaction.
- Added `KGQA_API_AUTH_TOKEN` to release-audit secret redaction.
- Updated release readiness documentation so API runtime hardening is part of
  the reproducible release evidence, while the project remains not
  production-ready.

### P17 API Production Hardening

- Added optional `KGQA_API_AUTH_TOKEN` Bearer token protection for `/readyz` and
  `/ask`; `/healthz` remains open for process health checks.
- Added `KGQA_API_MAX_QUESTION_CHARS` request-boundary configuration.
- Added stable API error envelopes with `status`, `error_type`, `error`, and
  `request_id`.
- Added API auth-token redaction in structured request logs and observability
  secret handling.

### P16 Production Runtime Foundation

- Added a minimal FastAPI runtime entry point:
  `uvicorn graphrag_v2.api.app:app`.
- Added `GET /healthz`, `GET /readyz`, and `POST /ask` around the existing
  graph-grounded QA path.
- Added API runtime env controls: `KGQA_API_INDEX_PATH`, `KGQA_API_CONFIG`,
  `KGQA_API_ANSWERER`, and `KGQA_API_STRICT_NEO4J`.
- Added structured JSON request logging with request id, status code, latency,
  and secret redaction.

### P15 Production Readiness Closeout

- Documented the current Go/No-Go decision in the public handoff docs:
  production-path beta, not production-ready.
- Recorded lightweight evidence for offline tests, security gate, DeepSeek
  `deepseek-v4-flash` real smoke, and release audit reporting.
- Documented remaining production blockers: hosted runtime, observability,
  load/cost testing, backup/restore drills, access control, and full release
  audit rerun.

### P14 Local Real LLM Config File

- Added `KGQA_REAL_LLM_CONFIG` so the opt-in real LLM smoke reads provider,
  model, API base, and API key from ignored YAML such as
  `settings.local.real-llm.yaml`.
- Kept DeepSeek `deepseek-v4-flash` as the default real LLM smoke path while
  allowing mainstream OpenAI-compatible providers through the same config file.
- Updated release candidate audit reports to record the config path and
  `set`/`unset` key state without writing the key value to `report.json` or
  `report.md`.
- Updated the optional GitHub Actions smoke job so `DEEPSEEK_API_KEY` is scoped
  only to the step that writes the ignored local config file.

### P13 Generic Real LLM Provider Gate

- Changed the opt-in real LLM gate from GLM/Zhipu-specific defaults to a
  generic OpenAI-compatible provider gate.
- Default real LLM smoke now uses DeepSeek `deepseek-v4-flash` with
  `DEEPSEEK_API_KEY`; `KGQA_REAL_LLM_API_KEY`, `KGQA_REAL_LLM_PROVIDER`,
  `KGQA_REAL_LLM_MODEL`, and `KGQA_REAL_LLM_API_BASE` support mainstream
  provider overrides.
- Release candidate audit reports now include redacted real LLM provider,
  model, API base, and key-env metadata in `report.json` and `report.md`.
- Legacy GLM/Zhipu remains available only through explicit provider selection.

### P11 Release Candidate Audit

- Added `scripts/audit_release_candidate.py` to run or record offline, Neo4j,
  Docker/compose, and real LLM release candidate gates.
- Added release candidate reports at
  `artifacts/release-candidate-audit/report.json` and
  `artifacts/release-candidate-audit/report.md`.
- Documented `blocked` gate semantics so missing real LLM credentials are
  audited as external release gaps instead of being treated as passed.
- Recorded the 2026-06-07 local P11 audit result: offline and Neo4j gates
  passed; real LLM remains blocked by local environment gaps.
- Updated release scope so Docker/compose is
  `excluded from current release scope` and reported as `skipped` by default;
  real LLM remains the only unresolved external release gate.

### P10 Documentation And Handoff

- Added `docs/operator_guide.md` as the clean developer/operator handoff guide
  for install, configuration, architecture/artifact lifecycle, indexing, QA,
  Neo4j, QA eval, release verification, security governance, troubleshooting,
  and known limits.
- Added a documentation handoff gate covering README baseline accuracy,
  operator guide coverage, examples, run observability artifacts, and the
  security gate.
- Updated examples and artifact docs for `kgqa inspect run`,
  `run_events.jsonl`, `run_summary.json`, community reports, local LLM
  extraction, and artifact retention guidance.

### P9 Security And Data Governance

- Added `scripts/security_check.py` as an offline release security gate for
  required ignore patterns, secret-redaction env coverage, credential-like text
  scans, and governance documentation checks.
- Extended operational secret redaction to cover `OPENAI_API_KEY` and
  `LOCAL_LLM_API_KEY`.
- Wired the security gate into `scripts/verify_release.sh`.
- Added release guidance for artifact retention, model output storage,
  dependency vulnerability review, and license review.

### P8 CI Release Deployment

- Hardened GitHub Actions so default validation stays offline while manual
  dispatch can opt into Neo4j, real LLM, and Docker checks.
- Extended `scripts/verify_release.sh` to inspect the latest run and verify
  run observability artifacts.
- Updated release/deployment docs for local Docker and compose validation
  boundaries.

### P7 Observability And Run Management

- Added deterministic local run observability artifacts:
  `run_events.jsonl` and `run_summary.json`.
- Added stable `run_id`, failed-stage context, stage timings, event counts, and
  run artifact links to `index_metadata.json`.
- Added `kgqa inspect run --index <path>`.
- Shared secret redaction across CLI stderr, failure metadata, event records,
  and run summaries, including `GRAPHRAG_EMBEDDING_API_KEY`.

### Added

- Added artifact contract constants and `docs/artifacts.md` for persisted artifact, metadata, graph JSON, and QA JSON contracts.
- Added `document_scan.json` ingestion manifest with included/ignored/rejected file decisions and scan-count metadata.
- Added input guardrails for unsupported-file policy, maximum file size, and maximum document count.
- Added LLM extraction prompt version metadata, failed chunk ids, and optional response cache with hit/miss counters.
- Added relation schema registry, manual graph-fusion overrides, rejection reasons, fusion provenance, and fusion parameter metadata.
- Added `scripts/export_review_queue.py` for local accepted/rejected triple review JSONL export.
- Added Neo4j health/schema reporting with schema version, expected/missing schema names, health status, and richer `kgqa inspect graph` output.
- Added strict Neo4j indexing preflight via `kgqa index --graph-store neo4j --strict-neo4j`.
- Added staged Neo4j replacement writes so successful canonical indexes are preserved until staging promotion succeeds.
- Added graph-store health/schema/write-strategy metadata fields to `index_metadata.json`.
- Added Neo4j operator guidance for backup, restore, rebuild, and old index cleanup.
- Added P6 QA quality gate metrics: entity/relationship recall and MRR, citation grounding, threshold checks, answer prompt version metadata, and query trace metadata.

### Changed

- Stabilized parquet artifact writing so empty artifacts keep contract columns.
- Updated local verification baseline to `459 passed, 4 skipped`.

## 0.2.0-beta.1 - 2026-06-05

### Added

- Added local Docker CLI image support for beta validation.
- Added docker-compose Neo4j + `kgqa` local smoke configuration.
- Added `DEPLOYMENT.md` with local Python, Docker CLI, docker-compose Neo4j, and opt-in real LLM smoke instructions.
- Added `settings.compose.neo4j.yaml` for compose-based Neo4j verification.
- Added an optional GitHub Actions Neo4j service job gated behind `workflow_dispatch`.

### Changed

- Updated package version to `0.2.0-beta.1`.
- Updated README, examples, production plan, and release checklist for beta validation.

### Known Limits

- Docker/compose is for local validation only, not a production deployment.
- Structured logging, metrics, production deployment, automated Neo4j cleanup/migration commands, frontend, vector database, distributed indexing, model training, and Azure-specific provider support are not included.

## 0.1.0 - 2026-06-04

### Added

- Packaged `openfusion-kgqa` with the `kgqa` console script.
- Added `kgqa init`, `kgqa index`, `kgqa ask`, and `kgqa inspect`.
- Added the default `kgqa index` full path for offline JSON demos.
- Added document loading and chunk provenance for `.txt`, `.md`, and `.pdf` files.
- Added mock entity, relationship, and triple extraction.
- Added graph fusion with entity resolution, relation alignment, triple scoring, and rejected triple artifacts.
- Added JSON and Neo4j graph stores.
- Added GraphRAG-style community detection and community report artifacts.
- Added graph-grounded QA with graph evidence, community evidence, and citations.
- Added `kgqa ask --format text|json` with a structured QA output contract.
- Added QA refusal metadata so no-source-evidence questions do not produce uncited answers.
- Added explicit `kgqa ask --answerer mock|llm` selection with strict LLM error handling.
- Added QA evaluation tooling with `scripts/evaluate_qa.py` and `examples/eval/qa/questions.jsonl`.
- Added examples and CI-ready tests.
- Added a strict `--extractor llm` path with GLM/Zhipu, OpenAI-compatible, vLLM/local provider registry, JSON extraction prompts, response parsing, repair retry, partial salvage, and CLI coverage.
- Added extraction config for LLM provider selection, request rate limiting, concurrent extraction, prompt/token/cost budgets, salvage behavior, and LLM extraction metadata stats for P1 hardening.
- Added deterministic `index_id` metadata and Neo4j read/write isolation per output index.
- Added scoped Neo4j replacement semantics so re-indexing one output path does not delete other graph data.
- Added Neo4j graph inspection details, including database, `index_id`, entity counts, relationship counts, and text unit counts.
- Added automatic Neo4j KGQA schema constraint/index setup plus `schema_ready` inspection output.
- Added Neo4j transaction timeout, transient retry, and operation-scoped error context.
- Added `metadata_schema_version`, run status/timing metadata, failure metadata, and graph store write stats in `index_metadata.json`.
- Added centralized CLI error handling with exit code `2` for expected user/service errors, exit code `1` for unexpected errors, and basic secret redaction.
- Added `kgqa ask --config` for non-default graph store configuration.
- Added LLM token, latency, error, and optional config-driven cost metadata.
- Added a gated real GLM + Neo4j smoke test controlled by `KGQA_REAL_LLM_SMOKE=1` and `ZHIPUAI_API_KEY`.
- Added extraction evaluation tooling with `scripts/evaluate_extraction.py`, entity/relationship recall metrics, and `examples/eval/expected_graph.json`.
- Added release verification with `scripts/verify_release.sh`.

### Known Limits

- The stable offline extractor is `mock`; `--extractor llm` is available as a hardened real LLM path with strict real-client checks, opt-in real smoke, and a minimal extraction eval.
- `--community` requires `--graph-store neo4j`.
- Real LLM smoke is opt-in and is not part of default CI or release verification.
- The LLM provider registry does not include Azure-specific provider support; large-scale production QA evaluation, frontend, vector database, model training, and distributed indexing are not included.
