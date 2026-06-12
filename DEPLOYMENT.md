# Deployment And Local Verification

OpenFusionKGQA is a `0.2.0-beta.1` prototype. The commands below are for local
development and beta validation, not a production deployment recipe.

## Local Python

```bash
python -m pip install -e ".[dev]"
kgqa index examples/docs --output artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa inspect graph --index artifacts/demo
scripts/verify_release.sh
```

## CI And Release Gates

Default GitHub Actions validation runs offline tests only. Manual
`workflow_dispatch` exposes optional `run_neo4j`, `run_real_llm`, and
`run_docker` checks for release candidates.

`scripts/verify_release.sh` is the local offline release gate. It covers package
installation, CLI help, offline indexing, QA, graph inspection, run inspection,
QA evaluation, run observability artifacts, and the full test suite.

Docker and compose commands remain local validation tools, not a hosted
production deployment.

## Security And Data Governance

Generated artifacts may contain source text, extracted graph data, QA answers,
model outputs, failed-response snippets, run summaries, and local review queues.
Do not index sensitive data without governance review.

Model outputs require retention review before sharing or packaging release
artifacts. Keep `artifacts/`, local logs, Docker volumes, `.env*`, and
`settings.local.*` files out of git unless a file is deliberately sanitized and
documented as sample data.

Before a release candidate, run:

```bash
python scripts/security_check.py
```

The release checklist also requires dependency vulnerability review and license
review for runtime dependencies, development dependencies, and sample documents.

## Docker CLI

Build the local CLI image:

```bash
docker build -t openfusion-kgqa:0.2.0-beta.1 .
```

Run the offline JSON demo from the container:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  openfusion-kgqa:0.2.0-beta.1 \
  index examples/docs --output artifacts/docker-demo
```

Inspect and ask:

```bash
docker run --rm -v "$PWD:/workspace" openfusion-kgqa:0.2.0-beta.1 \
  inspect graph --index artifacts/docker-demo

docker run --rm -v "$PWD:/workspace" openfusion-kgqa:0.2.0-beta.1 \
  ask "GraphRAG 是什么？" --index artifacts/docker-demo
```

## Docker Compose With Neo4j

Start Neo4j:

```bash
docker compose up -d neo4j
```

Run an end-to-end Neo4j + community smoke:

```bash
docker compose run --rm kgqa \
  index examples/docs \
  --output artifacts/compose-neo4j-demo \
  --config settings.compose.neo4j.yaml \
  --graph-store neo4j \
  --community

docker compose run --rm kgqa \
  inspect graph \
  --index artifacts/compose-neo4j-demo \
  --config settings.compose.neo4j.yaml \
  --graph-store neo4j
```

For production-path validation, run the same index with strict Neo4j enabled:

```bash
docker compose run --rm kgqa \
  index examples/docs \
  --output artifacts/compose-neo4j-strict \
  --config settings.compose.neo4j.yaml \
  --graph-store neo4j \
  --strict-neo4j
```

`kgqa inspect graph --graph-store neo4j` should report `health_status: ready`,
the Neo4j `database`, canonical `index_id`, graph counts, `schema_version`,
`schema_ready: True`, KGQA constraints/indexes, empty missing-schema lists, and
metadata path. Failed strict runs return a non-zero exit and write failed run
metadata instead of silently falling back to JSON.

Stop local services:

```bash
docker compose down
```

Use `docker compose down -v` only when you also want to delete the local Neo4j
data volume.

## Neo4j Operations Notes

Neo4j data is scoped by the stable `index_id` derived from the output directory.
With the default `staged_replace_on_write: true`, rebuilds for the same output
directory write to a staging index and promote only after the full graph write
succeeds. A failed staged write is cleaned up best effort and should leave the
previous canonical index intact.

Operational boundaries:

- Backup: use Neo4j-supported database backup/export tooling for the selected
  database. Record `kgqa inspect graph --graph-store neo4j` output with the
  backup, especially `index_id`, `schema_version`, counts, and missing schema
  lists.
- Restore: restore the database with Neo4j tooling, then inspect the target index
  and verify `health_status: ready`, counts, and metadata path.
- Rebuild: rerun `kgqa index ... --graph-store neo4j --strict-neo4j` against the
  same output directory. Staged replace preserves the previous successful index
  until promotion succeeds.
- Cleanup: there is no CLI cleanup command yet. Delete old index scopes only
  after confirming the `index_id` is no longer referenced by active artifacts or
  consumers.

## Real LLM Smoke

Real model calls are opt-in and are not part of default CI or release
verification. The default provider/model is DeepSeek `deepseek-v4-flash`:

```bash
export KGQA_REAL_LLM_SMOKE=1
export KGQA_REAL_LLM_CONFIG=settings.local.real-llm.yaml
cp settings.llm.neo4j.example.yaml settings.local.real-llm.yaml
# Fill models.default_chat_model.api_key in settings.local.real-llm.yaml.
python -m pytest graphrag_v2/tests/integration/test_real_llm_smoke.py -q
```

Use `settings.local.real-llm.yaml` to set another OpenAI-compatible
`model_provider`, `model`, `api_base`, and `api_key`. `DEEPSEEK_API_KEY` is
used only by the optional GitHub Actions step that writes this ignored config
file; `KGQA_REAL_LLM_PROVIDER`, `KGQA_REAL_LLM_MODEL`,
`KGQA_REAL_LLM_API_BASE`, and `KGQA_REAL_LLM_API_KEY` remain available as
temporary overrides. Legacy GLM remains available with `model_provider: zhipu`
or `llm_provider: glm`. When using `scripts/start_fresh_neo4j.sh`, source the
generated `kgqa-neo4j.env`; `NEO4J_URI`, `NEO4J_USERNAME`, and
`NEO4J_DATABASE` override YAML graph-store values so the selected free port is
used consistently.
and `llm_provider: glm`.

Do not commit API keys, `settings.local.*`, generated artifacts, Docker volumes,
or local logs.
