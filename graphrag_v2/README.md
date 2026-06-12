# graphrag_v2

`graphrag_v2` 是 OpenFusionKGQA 的历史内部包名。公开项目名是
OpenFusionKGQA，主实现链路把文档处理、知识抽取、图谱融合、图存储、
社区报告和图谱问答放在同一条可测试链路里。

## Architecture

```text
Documents
  -> document loader / chunker
  -> extraction explorer
  -> graph fusion supervisor
  -> graph store
  -> community detection / reports
  -> graph-grounded QA
```

主要模块：

```text
graphrag_v2/
├── cli/           # kgqa 命令行入口
├── config/        # Pydantic 配置模型和加载器
├── document/      # 文档读取和 chunk 切分
├── extraction/    # mock / LLM 抽取接口和校验
├── graph_fusion/  # 实体归并、关系对齐、三元组评分
├── graph_store/   # JSON 和 Neo4j 图存储
├── community/     # 社区检测、报告生成、Neo4j 写入
├── artifacts/     # parquet / JSON artifact 写入
├── qa/            # graph-grounded QA 主路径
├── query/         # 早期 GraphRAG query 原型
└── tests/         # 单元和集成测试
```

## Main Entry Points

Python API：

```python
from graphrag_v2.config import GraphRagConfig
from graphrag_v2.indexing import index_fusion_only
from graphrag_v2.qa import GraphGroundedQA

# Indexing API is async because real extractors may call remote models.
metadata = await index_fusion_only(
    input_path="examples/docs",
    output_path="artifacts/demo",
    config=GraphRagConfig(),
    extractor_name="mock",
    graph_store_provider="json",
)

qa = GraphGroundedQA.from_index("artifacts/demo")
result = qa.ask("GraphRAG 是什么？")
print(result.answer)
```

CLI：

```bash
kgqa index examples/docs --output artifacts/demo
kgqa ask "GraphRAG 是什么？" --index artifacts/demo
kgqa inspect graph --index artifacts/demo
```

## Indexing Modes

`kgqa index` 默认运行完整离线闭环：

```bash
kgqa index examples/docs --output artifacts/demo
```

也可以按阶段调试：

```bash
kgqa index examples/docs --output artifacts/docs --mode documents-only
kgqa index examples/docs --output artifacts/extraction --mode extraction-only
kgqa index examples/docs --output artifacts/fusion --mode fusion-only
```

当前 CLI 支持 `mock` 和 `llm` extractor。`mock` 是稳定离线 demo 路径；`llm` 是严格真实 LLM 路径，当前 provider registry 默认使用 DeepSeek `deepseek-v4-flash`，并兼容 GLM/Zhipu、OpenAI-compatible、vLLM/local endpoint，已具备 JSON prompt、解析、repair retry、partial salvage、token/latency/cost metadata、请求节流、并发调度、预算保护、可选真实 LLM smoke 和最小 extraction eval。缺少可用真实客户端、API key 或 endpoint 时，`--extractor llm` 会明确失败，不会静默回退到 mock。

## QA Behavior

`graphrag_v2.qa` 的主入口是 `GraphGroundedQA`：

- `QueryRouter` 判断 local / global 路由。
- `EntityLinker` 将问题链接到图谱实体。
- `GraphRetriever` 检索实体邻域和关系证据。
- `CommunityRetriever` 检索社区报告证据。
- `EvidenceRetriever` 追溯原文 chunk。
- `MockAnswerer` 生成可测试回答。
- `LLMAnswerer` 提供显式真实模型回答入口。

CLI 输出固定为：

```text
Answer:
Graph Evidence:
Community Evidence:
Citations:
```

`kgqa ask` 支持 `--format text|json` 和 `--answerer mock|llm`。JSON 输出包含 `answer`、`route`、
`graph_evidence`、`community_evidence`、`citations`、`refused`、
`refusal_reason` 和 `metadata`。如果没有可追溯 source chunk evidence，QA
会返回 refusal，而不是生成无引用答案。默认 answerer 是 `mock`；显式选择
`llm` 时需要真实 LLM 配置，缺少可用模型客户端、API key 或 endpoint 会返回错误。

QA JSON 保持稳定的顶层 contract。调试和审计细节放在 `metadata` 下，包括
`answer_prompt_version` 和 `query_trace`；trace 包含 route、linked entities、
retrieved relationships、retrieved communities 和 retrieved text chunks。trace
不会包含 raw prompts、credentials、hidden reasoning 或完整 LLM response。

## QA Evaluation

基础离线 QA eval：

```bash
kgqa index examples/docs --output artifacts/demo
scripts/evaluate_qa.py \
  --index artifacts/demo \
  --questions examples/eval/qa/questions.jsonl
```

默认输出 JSON report，也支持 `--format markdown --output artifacts/qa-eval.md`。
当前指标包含 retrieval hit rate、entity/relationship recall、entity/relationship
MRR、citation coverage、citation grounding rate、refusal accuracy、route accuracy
和 latency。

P6 QA gate 可以通过阈值参数让 release verification 明确失败：

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

## Metadata And Errors

`index_metadata.json` 使用 `metadata_schema_version: 1`，并记录 `run_status`、
run start/finish time、elapsed seconds、indexing mode、阶段计数、LLM/extraction
metadata 和 graph store 写入统计。失败时会尽量记录 `run_error_type` 和脱敏后的
`run_error_message`。

- `run_events.jsonl` and `run_summary.json` record each indexing run with a
  stable `run_id`, stage timings, counters, provider/model context, failed-stage
  details, and redacted operational error messages.

CLI 约定：用户输入、配置、缺失文件和外部依赖错误返回 exit code `2`；未预期异常返回
exit code `1`。stderr 不应包含 API key 或 Neo4j password。

## Graph Stores

JSON graph store 是默认离线路径，适合 demo、测试和 CI：

```bash
kgqa index examples/docs --output artifacts/demo --graph-store json
```

Neo4j 是生产路径。社区检测和报告写入当前要求 Neo4j：

```bash
export NEO4J_PASSWORD="your-password"
kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --graph-store neo4j \
  --community

kgqa inspect graph --index artifacts/neo4j-demo --graph-store neo4j
kgqa ask "Neo4j 在项目里起什么作用？" --index artifacts/neo4j-demo
```

Neo4j 数据按输出目录生成的 `index_id` 隔离。重跑同一个输出目录会替换该
index scope，不会删除其他索引或外部 Neo4j 数据。非默认 Neo4j 连接参数通过
`kgqa init --output settings.yaml` 生成配置后传入 `--config settings.yaml`。
生产索引建议同时传入 `--strict-neo4j`，这样会在写入前检查 Neo4j 连接和 schema
路径，Neo4j 不可用时直接失败并写入失败 metadata，不会生成看起来成功的生产索引。
默认 `staged_replace_on_write: true` 时，同一 `index_id` 的重建会先写入 staging
index，全部写入成功后才 promote；失败时保留上一版成功索引。Neo4j 写入会自动确保
KGQA constraints 和 indexes 存在；`kgqa inspect graph` 会输出 `health_status`、
`schema_version`、`schema_ready`、expected/missing schema names、database、
counts、`metadata_path`、`write_strategy` 和 `staging_index_id`。这些字段也会写入
`index_metadata.json` 的 `graph_store_*` metadata。

Neo4j 运维以 database 备份和 `index_id` 隔离为边界：备份/恢复使用 Neo4j 官方工具，
恢复后用 `kgqa inspect graph --graph-store neo4j` 核对 schema 和 counts；重建同一
输出目录使用 `kgqa index ... --graph-store neo4j --strict-neo4j`；清理旧 index id
目前需要在确认不再使用后通过 Neo4j 维护查询按 `index_id` 删除。

## Development

安装：

```bash
python -m pip install -e ".[dev]"
```

运行测试：

```bash
python -m pytest graphrag_v2/tests -q
```

发布前离线验证：

```bash
scripts/verify_release.sh
```

该脚本会清空 Neo4j 连接环境变量，不会启用真实 LLM smoke。
它会运行 `python scripts/security_check.py`、offline JSON demo、基础 QA eval、
`kgqa inspect run` 和完整测试。

本地 Docker CLI 验证：

```bash
docker build -t openfusion-kgqa:0.2.0-beta.1 .
docker run --rm -v "$PWD:/workspace" openfusion-kgqa:0.2.0-beta.1 --help
```

docker-compose Neo4j 本地 smoke 见仓库根目录 [DEPLOYMENT.md](../DEPLOYMENT.md)。

只跑 KGQA 集成测试：

```bash
python -m pytest graphrag_v2/tests/integration/test_kgqa_pipeline.py -q
```

需要本地 Neo4j 的生产路径验证：

```bash
scripts/start_fresh_neo4j.sh
source "$(ls -td ~/.local/share/kgqa-neo4j-openfusion-*/kgqa-neo4j.env | head -1)"

python -m pytest graphrag_v2/tests/integration/test_neo4j_store.py \
  graphrag_v2/tests/integration/test_community_pipeline.py -q
```

端到端 Neo4j demo：

```bash
cp settings.neo4j.example.yaml settings.local.yaml
kgqa index examples/docs \
  --output artifacts/neo4j-demo \
  --config settings.local.yaml \
  --graph-store neo4j \
  --community
kgqa inspect graph --index artifacts/neo4j-demo --config settings.local.yaml --graph-store neo4j
kgqa ask "Neo4j 在项目里起什么作用？" --index artifacts/neo4j-demo --config settings.local.yaml
```

真实 LLM + Neo4j smoke 需要显式开关。默认 provider/model 是
DeepSeek `deepseek-v4-flash`：

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
  --community
scripts/evaluate_extraction.py --index artifacts/llm-neo4j-smoke --expected examples/eval/expected_graph.json
```

要切换其他 OpenAI-compatible provider，修改
`settings.local.real-llm.yaml` 中的 `model_provider`、`model`、`api_base`
和 `api_key`。`KGQA_REAL_LLM_PROVIDER`、`KGQA_REAL_LLM_MODEL`、
`KGQA_REAL_LLM_API_BASE` 和 `KGQA_REAL_LLM_API_KEY` 仍可作为临时覆盖；
Legacy GLM 仍可用 `model_provider: zhipu` / `llm_provider: glm` 显式启用。

基础 QA eval：

```bash
scripts/evaluate_qa.py --index artifacts/demo --questions examples/eval/qa/questions.jsonl
```

当前测试基线（默认离线测试，未启用真实 LLM smoke）：`459 passed, 4 skipped`。

完整交接指南见 [docs/operator_guide.md](../docs/operator_guide.md)。安全/治理 gate 可单独运行：

```bash
python scripts/security_check.py
```

## Compatibility Notes

- `query/` 中仍保留早期 GraphRAG query 原型，但新产品入口是 `qa/` 和 `kgqa ask`。
- 默认 demo 不依赖 API key。`--extractor llm` 和 `--answerer llm` 都要求真实 LLM 配置；真实 LLM smoke 仅在显式环境变量和可用凭据/endpoint 存在时运行。
