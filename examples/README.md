# Examples

这个目录提供 OpenFusionKGQA 的最小可运行 demo。默认路径不需要 API key、不需要 Neo4j。

## Offline JSON Demo

从仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
kgqa index examples/docs --output artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa inspect graph --index artifacts/demo
kgqa inspect run --index artifacts/demo
```

输入文档：

```text
examples/docs/graphrag.md
examples/docs/neo4j.txt
```

生成目录：

```text
artifacts/demo/
├── document_scan.json
├── text_units.parquet
├── candidate_entities.parquet
├── candidate_relationships.parquet
├── candidate_triples.parquet
├── entities.parquet
├── relationships.parquet
├── rejected_triples.parquet
├── graph.json
├── run_events.jsonl
├── run_summary.json
└── index_metadata.json
```

## Ask More Questions

```bash
kgqa ask "Neo4j 在项目里起什么作用？" --index artifacts/demo
kgqa ask "知识图谱和 RAG 是怎么结合的？" --index artifacts/demo
```

默认 demo 使用 `MockAnswerer`，重点是验证图谱证据、原文引用和 CLI 输出结构。

## QA Evaluation

基础 QA eval 使用 `examples/eval/qa/questions.jsonl`，覆盖 local、relationship、global route、no-answer 和 citation-sensitive case：

```bash
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl
```

默认输出 JSON，指标包含 retrieval hit rate、citation coverage、refusal accuracy、route accuracy 和 latency。需要 Markdown report 时：

```bash
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl \
  --format markdown \
  --output artifacts/qa-eval.md
```

## Release Smoke

发布前可以从仓库根目录运行完整离线 smoke：

```bash
scripts/verify_release.sh
```

该脚本会使用 `artifacts/release-smoke`，不需要 API key，不需要 Neo4j，并会清空 Neo4j 连接环境变量。它也会运行基础 QA eval。
它会先运行 `scripts/security_check.py`，并检查 `kgqa inspect run`、
`run_events.jsonl` 和 `run_summary.json`。

## Docker Demo

本地 CLI 容器验证：

```bash
docker build -t openfusion-kgqa:0.2.0-beta.1 .
docker run --rm -v "$PWD:/workspace" openfusion-kgqa:0.2.0-beta.1 \
  index examples/docs --output artifacts/docker-demo
```

本地 Neo4j compose smoke：

```bash
docker compose up -d neo4j
docker compose run --rm kgqa \
  index examples/docs \
  --output artifacts/compose-neo4j-demo \
  --config settings.compose.neo4j.yaml \
  --graph-store neo4j \
  --community
```

这些命令只用于本地验证，不是生产部署方案。

## Neo4j Demo

如果本机已有 Neo4j：

```bash
export NEO4J_PASSWORD="your-password"
kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --graph-store neo4j \
  --community
kgqa inspect graph --index artifacts/neo4j-demo --graph-store neo4j
kgqa ask "这批文档主要讲了哪些主题？" --index artifacts/neo4j-demo
kgqa inspect communities --index artifacts/neo4j-demo --graph-store neo4j
kgqa inspect community-reports --index artifacts/neo4j-demo --graph-store neo4j
```

默认连接：

- `bolt://localhost:7687`
- user: `neo4j`
- password env: `NEO4J_PASSWORD`
- database: `neo4j`

Neo4j 写入会自动创建 KGQA schema constraints 和 indexes；`kgqa inspect graph`
会显示 `schema_ready`，用于确认当前连接是否具备项目需要的 schema。

如需自定义配置：

```bash
kgqa init --output settings.yaml
kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --config settings.yaml \
  --graph-store neo4j \
  --community
```

## Local/OpenAI-Compatible LLM Extraction

Local or OpenAI-compatible extraction is explicit opt-in. Keep endpoint and key
values in ignored local settings or environment variables:

```bash
kgqa index examples/docs \
  --output artifacts/local-llm-demo \
  --config settings.local.yaml \
  --extractor llm

scripts/evaluate_extraction.py \
  --index artifacts/local-llm-demo \
  --expected examples/eval/expected_graph.json
```

## Real LLM + Neo4j Smoke

真实 LLM smoke 需要显式开关，默认测试不会调用外部 API。默认 provider/model
是 DeepSeek `deepseek-v4-flash`：

```bash
source "$(ls -td ~/.local/share/kgqa-neo4j-openfusion-*/kgqa-neo4j.env | head -1)"
export KGQA_REAL_LLM_SMOKE=1
export KGQA_REAL_LLM_CONFIG=settings.local.real-llm.yaml

cp settings.llm.neo4j.example.yaml settings.local.real-llm.yaml
# 在 settings.local.real-llm.yaml 中填写 models.default_chat_model.api_key。
python -m pytest graphrag_v2/tests/integration/test_real_llm_smoke.py -q

kgqa index examples/docs \
  --output artifacts/llm-neo4j-smoke \
  --config "$KGQA_REAL_LLM_CONFIG" \
  --extractor llm \
  --graph-store neo4j \
  --strict-neo4j \
  --community

scripts/evaluate_extraction.py \
  --index artifacts/llm-neo4j-smoke \
  --expected examples/eval/expected_graph.json

scripts/evaluate_qa.py \
  --index artifacts/llm-neo4j-smoke \
  --questions examples/eval/qa/questions.jsonl \
  --config "$KGQA_REAL_LLM_CONFIG" \
  --strict-neo4j \
  --answerer llm \
  --min-route-accuracy 1.0 \
  --min-retrieval-hit-rate 0.8 \
  --min-citation-grounding-rate 1.0
```

要切换其他 OpenAI-compatible provider，修改
`settings.local.real-llm.yaml` 中的 `model_provider`、`model`、`api_base`
和 `api_key`；`KGQA_REAL_LLM_PROVIDER`、`KGQA_REAL_LLM_MODEL`、
`KGQA_REAL_LLM_API_BASE` 和 `KGQA_REAL_LLM_API_KEY` 仍可作为临时覆盖。
