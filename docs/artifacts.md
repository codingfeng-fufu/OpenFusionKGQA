# Artifact and Response Contracts

This document defines the persisted interfaces for local indexing and QA.
The source of truth for exact keys and column names is
`graphrag_v2.artifacts.contracts`.

## Schema Version

`index_metadata.json` uses `metadata_schema_version: 1`.

Backward-compatible changes may add optional metadata keys. Removing keys,
renaming keys, or changing parquet column names requires a schema version bump.

## Generated Artifacts

Index runs clear generated artifacts at run start and then rewrite the artifacts
for the selected mode:

- `document_scan.json`
- `text_units.parquet`
- `candidate_entities.parquet`
- `candidate_relationships.parquet`
- `candidate_triples.parquet`
- `entities.parquet`
- `relationships.parquet`
- `rejected_triples.parquet`
- `graph.json`
- `run_events.jsonl`
- `run_summary.json`
- `communities.parquet`
- `community_reports.parquet`

Empty parquet artifacts must still be written with their contract columns.

## Document Scan Manifest

`document_scan.json` records every scanned file decision.

Top-level keys:

- `input_path`
- `unsupported_file_policy`
- `records`
- `num_files`
- `num_included_files`
- `num_ignored_files`
- `num_rejected_files`
- `num_empty_documents`

Each record contains:

- `source_path`
- `title`
- `extension`
- `size_bytes`
- `status`
- `reason`

`status` is one of `included`, `ignored`, or `rejected`.
Rejected files fail indexing after the manifest and document-stage metadata are
written.

## Metadata Groups

Successful runs include run metadata (`run_status`, `run_mode`,
`run_started_at`, `run_finished_at`, `run_elapsed_seconds`) plus stage metadata.

Document-stage metadata includes input file counts, document counts, text unit
counts, chunk size, and chunk overlap.

LLM extraction metadata includes provider/model identity, token/latency/cost
stats when available, prompt version, parse/repair/salvage counts, failed chunk
ids, budget status, concurrency/rate limits, and cache hit/miss counters.

Fusion metadata includes `fusion_parameters_version`,
`fusion_relation_schema_mode`, `fusion_relation_schema_version`, `fusion_scoring_version`,
`fusion_scoring_weights`, accepted triple counts, and manual override counts.
Relationship and rejected-triple row metadata may include source triple ids,
source relationship ids, relation alignment details, and rejection reasons.

Graph-store metadata includes provider and write outcome fields for every
fusion/full run:

- `graph_store_provider`
- `graph_store_written`
- `graph_store_index_id`
- `graph_store_num_text_units`
- `graph_store_num_entities`
- `graph_store_num_relationships`
- `graph_store_health_status`

Neo4j runs may also include database, schema, and rebuild metadata:

- `graph_store_database`
- `graph_store_schema_ready`
- `graph_store_schema_constraint_count`
- `graph_store_schema_index_count`
- `graph_store_schema_version`
- `graph_store_missing_schema_constraints`
- `graph_store_missing_schema_indexes`
- `graph_store_write_strategy`
- `graph_store_staging_index_id`

`kgqa inspect graph --graph-store neo4j` is the operator-facing view of the same
health/schema state. A successful production Neo4j index should report
`health_status: ready`, empty missing-schema lists, and graph counts matching
the persisted artifacts.

## Run Observability

Each indexing run writes:

- `run_events.jsonl`
- `run_summary.json`

`run_events.jsonl` records run and stage lifecycle events, elapsed seconds,
counts, provider/model context, and redacted errors. `run_summary.json` is the
operator-facing summary used by `kgqa inspect run --index <path>`.

The same paths are linked from `index_metadata.json` through `run_events_path`
and `run_summary_path`.

## Human Review Queue

`scripts/export_review_queue.py` exports accepted relationships and rejected
triples from local artifacts to JSONL without requiring Neo4j:

```bash
scripts/export_review_queue.py --index artifacts/demo --output artifacts/review.jsonl
```

Each JSONL record includes `type`, `status`, entity/relation labels,
confidence, source triple ids, evidence chunk ids, and rejection reasons when
available.

## Security And Retention

Generated artifacts may contain source text, extracted graph data, model
outputs, failed-response snippets, QA reports, run logs, and local review
queues. Treat artifact retention as a release decision: keep generated outputs
out of git unless the data is deliberately sanitized and documented as sample
data.

Run `python scripts/security_check.py` before release packaging. The security
gate verifies ignore patterns, redaction env coverage, obvious credential-like
text, and data-governance documentation.

## QA JSON

`kgqa ask --format json` returns a stable object with:

- question and selected route
- answer and refusal fields
- citations and used graph/community ids
- graph, community, and text evidence
- source provider and provider metadata
- `refused` boolean derived from `refusal_reason`
