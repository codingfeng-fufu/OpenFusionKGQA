"""Persisted artifact and response contracts."""

from __future__ import annotations


METADATA_SCHEMA_VERSION = 1

GENERATED_ARTIFACTS = (
    "run_events.jsonl",
    "run_summary.json",
    "document_scan.json",
    "text_units.parquet",
    "candidate_entities.parquet",
    "candidate_relationships.parquet",
    "candidate_triples.parquet",
    "entities.parquet",
    "relationships.parquet",
    "rejected_triples.parquet",
    "graph.json",
    "communities.parquet",
    "community_reports.parquet",
)

DOCUMENT_SCAN_KEYS = (
    "input_path",
    "unsupported_file_policy",
    "records",
    "num_files",
    "num_included_files",
    "num_ignored_files",
    "num_rejected_files",
    "num_empty_documents",
)

DOCUMENT_SCAN_RECORD_KEYS = (
    "source_path",
    "title",
    "extension",
    "size_bytes",
    "status",
    "reason",
)

TEXT_UNITS_COLUMNS = (
    "chunk_id",
    "doc_id",
    "source_path",
    "chunk_index",
    "text",
    "n_tokens",
    "metadata",
)

CANDIDATE_ENTITIES_COLUMNS = (
    "id",
    "name",
    "type",
    "description",
    "confidence",
    "evidence_chunk_ids",
    "metadata",
)

CANDIDATE_RELATIONSHIPS_COLUMNS = (
    "id",
    "source",
    "target",
    "relation",
    "description",
    "confidence",
    "evidence_chunk_ids",
    "metadata",
)

CANDIDATE_TRIPLES_COLUMNS = (
    "id",
    "source_name",
    "target_name",
    "relation_mention",
    "canonical_relation",
    "description",
    "extraction_confidence",
    "relation_alignment_score",
    "evidence_support_score",
    "graph_consistency_score",
    "triple_score",
    "status",
    "evidence_chunk_ids",
    "metadata",
)

ENTITIES_COLUMNS = (
    "id",
    "name",
    "canonical_name",
    "type",
    "description",
    "aliases",
    "evidence_chunk_ids",
    "confidence",
    "metadata",
)

RELATIONSHIPS_COLUMNS = (
    "id",
    "source_entity_id",
    "target_entity_id",
    "source_name",
    "target_name",
    "relation",
    "original_relations",
    "description",
    "confidence",
    "evidence_chunk_ids",
    "extraction_count",
    "metadata",
)

COMMUNITIES_COLUMNS = (
    "id",
    "level",
    "title",
    "summary",
    "entity_ids",
    "relationship_ids",
    "text_unit_ids",
    "size",
    "rank",
    "metadata",
)

COMMUNITY_REPORTS_COLUMNS = (
    "id",
    "community_id",
    "title",
    "summary",
    "full_content",
    "findings",
    "key_entities",
    "key_relationships",
    "evidence_chunk_ids",
    "rank",
    "metadata",
)

GRAPH_KEYS = ("created_at", "nodes", "edges", "statistics")
GRAPH_STATISTICS_KEYS = ("num_nodes", "num_edges", "num_rejected_triples")

SUCCESS_METADATA_KEYS = (
    "metadata_schema_version",
    "created_at",
    "index_id",
    "run_id",
    "input_path",
    "output_path",
    "mode",
    "run_mode",
    "run_status",
    "run_started_at",
    "run_finished_at",
    "run_elapsed_seconds",
    "run_failed_stage",
    "run_stage_timings",
    "run_event_count",
    "run_summary_path",
    "run_events_path",
)

DOCUMENT_METADATA_KEYS = (
    "num_documents",
    "num_text_units",
    "num_input_files",
    "num_included_files",
    "num_ignored_files",
    "num_rejected_files",
    "num_empty_documents",
    "chunk_size",
    "chunk_overlap",
)

EXTRACTION_METADATA_KEYS = (
    "extractor",
    "num_candidate_entities",
    "num_candidate_relationships",
    "num_candidate_triples",
)

FUSION_METADATA_KEYS = (
    "num_entities",
    "num_relationships",
    "num_rejected_triples",
    "fusion_min_confidence",
    "fusion_parameters_version",
    "fusion_relation_schema_mode",
    "fusion_relation_schema_version",
    "fusion_scoring_version",
    "fusion_scoring_weights",
    "fusion_num_accepted_triples",
    "fusion_entity_override_count",
    "fusion_relation_override_count",
)

JSON_GRAPH_STORE_METADATA_KEYS = (
    "graph_store_provider",
    "graph_store_written",
    "graph_store_index_id",
    "graph_store_num_text_units",
    "graph_store_num_entities",
    "graph_store_num_relationships",
    "graph_store_health_status",
)

NEO4J_GRAPH_STORE_METADATA_KEYS = (
    "graph_store_provider",
    "graph_store_written",
    "graph_store_database",
    "graph_store_index_id",
    "graph_store_num_text_units",
    "graph_store_num_entities",
    "graph_store_num_relationships",
    "graph_store_health_status",
    "graph_store_schema_ready",
    "graph_store_schema_constraint_count",
    "graph_store_schema_index_count",
    "graph_store_schema_version",
    "graph_store_missing_schema_constraints",
    "graph_store_missing_schema_indexes",
    "graph_store_write_strategy",
    "graph_store_staging_index_id",
)

COMMUNITY_METADATA_KEYS = (
    "num_communities",
    "num_community_reports",
    "community_algorithm",
    "community_reporter",
)

LLM_EXTRACTION_METADATA_KEYS = (
    "llm_provider",
    "llm_model_id",
    "llm_model_name",
    "llm_mock_mode",
    "llm_total_calls",
    "llm_total_errors",
    "llm_total_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_latency_seconds",
    "llm_max_latency_seconds",
    "llm_average_latency_seconds",
    "llm_estimated_cost",
    "extraction_prompt_version",
    "extraction_parse_failures",
    "extraction_repair_attempts",
    "extraction_max_gleanings",
    "extraction_gleaning_attempts",
    "extraction_gleaning_failures",
    "extraction_gleaned_entities",
    "extraction_gleaned_relationships",
    "extraction_salvaged_entities",
    "extraction_salvaged_relationships",
    "extraction_dropped_entities",
    "extraction_dropped_relationships",
    "extraction_failed_chunks",
    "extraction_failed_chunk_ids",
    "extraction_budget_exceeded",
    "extraction_requests_per_minute",
    "extraction_concurrent_requests",
    "extraction_max_prompt_tokens_per_chunk",
    "extraction_max_total_tokens",
    "extraction_max_estimated_cost",
    "extraction_salvage_on_parse_failure",
    "extraction_cache_enabled",
    "extraction_cache_hits",
    "extraction_cache_misses",
    "extraction_elapsed_seconds",
)

QA_RESULT_KEYS = (
    "question",
    "route",
    "answer",
    "citations",
    "refusal_reason",
    "used_entities",
    "used_relationships",
    "used_communities",
    "used_community_reports",
    "confidence",
    "graph_evidence",
    "community_evidence",
    "text_evidence",
    "source_provider",
    "metadata",
    "refused",
)

GRAPH_EVIDENCE_KEYS = (
    "linked_entities",
    "relationships",
    "text_chunk_ids",
    "retrieval_metadata",
)
LINKED_ENTITY_KEYS = (
    "id",
    "name",
    "canonical_name",
    "type",
    "description",
    "score",
    "aliases",
    "evidence_chunk_ids",
)
RELATIONSHIP_EVIDENCE_KEYS = (
    "id",
    "source_entity_id",
    "target_entity_id",
    "source_name",
    "target_name",
    "relation",
    "description",
    "confidence",
    "extraction_count",
    "evidence_chunk_ids",
    "score",
    "hop",
)
TEXT_EVIDENCE_KEYS = (
    "chunk_id",
    "doc_id",
    "source_path",
    "chunk_index",
    "text",
    "score",
)
