# OpenFusionKGQA

OpenFusionKGQA is a beta prototype for open-corpus knowledge graph fusion and graph-grounded question answering.

它不是普通向量 RAG，也不是完整复现 Microsoft GraphRAG。项目目标是把本地开放文本转成可追溯的知识图谱，再基于图结构、原文证据和可选社区报告生成带 citation 的回答。

```text
Documents
  -> Text Units
  -> Entity / Relationship / Triple Extraction
  -> Graph Fusion
  -> JSON or Neo4j Graph Store
  -> GraphRAG-style Local / Global QA
  -> Answers with Citations
```

当前版本：`0.2.0-beta.1`

## Highlights

- 文档读取：支持 `.txt`、`.md`、`.pdf`，并保留 chunk provenance。
- 结构化抽取：支持稳定离线 `mock` extractor，以及 opt-in 真实 LLM JSON extraction。
- Graph fusion：对候选实体、关系、三元组做归一、关系对齐、证据检查、评分和 rejected-triple 记录。
- 图存储：默认 JSON artifacts；可选 Neo4j 作为 production-path graph store。
- GraphRAG-style QA：支持 local/global 路由、graph/text evidence retrieval、community evidence 和 citation。
- Benchmark：包含 offline QA eval、HotpotQA adapters、failure taxonomy、release verification。
- CLI：提供 `kgqa init`、`kgqa index`、`kgqa ask`、`kgqa inspect`。
- API skeleton：提供 FastAPI `healthz`、`readyz`、`metrics`、`ask` 本地服务入口。

## Current Status

This repository is a runnable beta, not production-ready.
The default opt-in real LLM smoke path uses DeepSeek `deepseek-v4-flash`.

Latest local release gate:

```text
scripts/verify_release.sh
Release verification passed
Full pytest: 459 passed, 4 skipped
```

Representative QA results:

```text
Offline QA eval: 7/7 passed
HotpotQA isolated real20: 16/20 passed, EM 0.80, token F1 0.80, support recall 1.0
HotpotQA isolated real100: 78/100 passed, EM 0.60, token F1 0.7083, support recall 1.0
```

Interpretation:

- Evidence retrieval and citation grounding are relatively stable.
- Final answer selection still has room for improvement.
- Real LLM extraction works, but structured-output parsing and robustness remain beta gaps.

## Quickstart

Requirements:

- Python 3.11+
- No API key required for the default offline demo
- No Neo4j required for the default offline demo

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the offline JSON demo:

```bash
kgqa index examples/docs --output artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa inspect graph --index artifacts/demo
kgqa inspect run --index artifacts/demo
```

Run the basic QA evaluation:

```bash
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl
```

Run the offline release gate:

```bash
scripts/verify_release.sh
```

`scripts/verify_release.sh` intentionally disables real LLM smoke and clears Neo4j environment variables. It verifies the default offline path: security check, package install, CLI help, indexing, QA, graph inspection, run inspection, QA eval, run observability artifacts, and the test suite.

## CLI Usage

Create a default settings file:

```bash
kgqa init --output settings.yaml
```

Index documents with the default mock extractor and JSON graph store:

```bash
kgqa index examples/docs --output artifacts/demo
```

Use stage-specific indexing for debugging:

```bash
kgqa index examples/docs --output artifacts/docs --mode documents-only
kgqa index examples/docs --output artifacts/extraction --mode extraction-only
kgqa index examples/docs --output artifacts/fusion --mode fusion-only
```

Ask questions:

```bash
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa ask "Neo4j 在项目里起什么作用？" --index artifacts/demo
kgqa ask "这批文档主要讲了哪些主题？" --index artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo --format json
```

Inspect generated artifacts:

```bash
kgqa inspect graph --index artifacts/demo
kgqa inspect entities --index artifacts/demo
kgqa inspect relationships --index artifacts/demo
kgqa inspect rejected --index artifacts/demo
kgqa inspect run --index artifacts/demo
```

`kgqa ask` defaults to `--answerer mock` for deterministic offline behavior. Real LLM answer generation requires `--answerer llm` and an explicit local config.

## Generated Artifacts

An index directory contains the persisted contracts used by QA, inspection, and evaluation:

```text
document_scan.json
text_units.parquet
candidate_entities.parquet
candidate_relationships.parquet
candidate_triples.parquet
entities.parquet
relationships.parquet
rejected_triples.parquet
graph.json
index_metadata.json
run_events.jsonl
run_summary.json
```

When community reporting is enabled, it also writes:

```text
communities.parquet
community_reports.parquet
```

See [docs/artifacts.md](docs/artifacts.md) for artifact contracts and response metadata.

## Neo4j Path

The default JSON graph store is best for offline demos and CI. Neo4j is the production-path graph store used for local service-backed validation.

Start a local Neo4j instance by your preferred method, set credentials, then run:

```bash
export NEO4J_PASSWORD="your-password"

kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --graph-store neo4j \
  --strict-neo4j \
  --community

kgqa inspect graph \
  --index artifacts/neo4j-demo \
  --graph-store neo4j

kgqa ask "Neo4j 在项目里起什么作用？" \
  --index artifacts/neo4j-demo
```

`--community` currently requires `--graph-store neo4j`. Community detection is implemented as a GraphRAG-style extension using a Neo4j graph projection, NetworkX Louvain community detection, and rule-based community reports.

For Docker and docker-compose validation, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Real LLM Extraction and Answering

Real model calls are opt-in. The repository does not require API keys for the default demo or release verification.

Create a local config from the example and keep it uncommitted:

```bash
cp settings.llm.neo4j.example.yaml settings.local.real-llm.yaml
# Fill models.default_chat_model.api_key in settings.local.real-llm.yaml.
```

Run an LLM-backed extraction path:

```bash
kgqa index examples/docs \
  --output artifacts/llm-demo \
  --config settings.local.real-llm.yaml \
  --extractor llm
```

Ask with a real LLM answerer:

```bash
kgqa ask "GraphRAG 是什么？" \
  --index artifacts/llm-demo \
  --config settings.local.real-llm.yaml \
  --answerer llm
```

The real LLM path is strict: missing credentials, unavailable clients, or invalid endpoints fail explicitly instead of silently falling back to mock behavior.

## API Skeleton

After building an index:

```bash
export KGQA_API_INDEX_PATH=artifacts/demo
uvicorn graphrag_v2.api.app:app --host 127.0.0.1 --port 8000
```

Available local endpoints:

```text
GET  /healthz
GET  /readyz
GET  /metrics
POST /ask
```

Optional environment variables include `KGQA_API_CONFIG`, `KGQA_API_ANSWERER`, `KGQA_API_STRICT_NEO4J`, `KGQA_API_AUTH_TOKEN`, and `KGQA_API_MAX_QUESTION_CHARS`.

This API is a runtime skeleton for validation and integration work. `/metrics` is intended for local monitoring/supervisor checks. It is not a hosted production deployment.

## Benchmarking

Offline QA eval:

```bash
kgqa index examples/docs --output artifacts/demo
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl
```

HotpotQA mini adapter:

```bash
scripts/benchmark_hotpotqa_mini.py \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --output artifacts/hotpotqa-mini \
  --sample-size 25 \
  --seed 42
```

If the local HotpotQA file is not available, the adapter can explicitly download the dev distractor set:

```bash
scripts/benchmark_hotpotqa_mini.py \
  --download \
  --output artifacts/hotpotqa-mini \
  --sample-size 25 \
  --seed 42
```

Generated benchmark outputs stay under `artifacts/` and should not be committed.

## Repository Layout

```text
OpenFusionKGQA/
├── graphrag_v2/              # Main package
│   ├── api/                  # FastAPI runtime skeleton
│   ├── cli/                  # kgqa CLI
│   ├── community/            # GraphRAG-style community detection and reports
│   ├── document/             # Document loading and chunking
│   ├── extraction/           # Mock and LLM extraction paths
│   ├── graph_fusion/         # Entity/relation/triple fusion
│   ├── graph_store/          # JSON and Neo4j stores
│   ├── qa/                   # Query routing, retrieval, answering
│   └── tests/                # Unit and integration tests
├── examples/                 # Minimal public demo data
├── scripts/                  # Evaluation, release, and local utility scripts
├── docs/                     # Public operator and artifact documentation
├── DEPLOYMENT.md             # Local Docker/compose validation notes
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Public Documentation

- [examples/README.md](examples/README.md): Minimal offline demo.
- [docs/operator_guide.md](docs/operator_guide.md): Local operator handoff.
- [docs/artifacts.md](docs/artifacts.md): Artifact and response contracts.
- [DEPLOYMENT.md](DEPLOYMENT.md): Local Python, Docker CLI, and docker-compose validation.
- [CHANGELOG.md](CHANGELOG.md): Version history.

## Known Limits

- This is a beta prototype, not a production-ready hosted service.
- The default offline path uses deterministic mock extraction and mock answer generation.
- Real LLM extraction is available but still sensitive to structured-output quality and parse robustness.
- Final answer selection remains weaker than evidence retrieval on larger multi-hop samples.
- Community reports are currently rule-based/mock reports, not full LLM-generated hierarchical GraphRAG reports.
- `--community` currently requires Neo4j.
- Docker and compose files are local validation tools, not a production deployment recipe.
- The project does not include a frontend, vector database, distributed indexing system, model training pipeline, or automated Neo4j cleanup/migration command.

## Security and Data Notes

Generated artifacts may contain source text, extracted graph data, model outputs, failed-response snippets, QA reports, run logs, and local review queues. Keep these out of git unless deliberately sanitized as sample data.

Do not commit:

```text
.env*
settings.local.*
artifacts/
graphrag_v2/artifacts/
local logs
API keys
Neo4j data volumes
benchmark outputs
```

Run the security check before release candidates:

```bash
python scripts/security_check.py
```
