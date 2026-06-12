"""Indexing entry points for CLI commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from graphrag_v2.artifacts import (
    fail_run_metadata,
    finish_run_metadata,
    start_run_metadata,
    write_community_artifacts,
    write_document_artifacts,
    write_extraction_artifacts,
    write_fusion_artifacts,
)
from graphrag_v2.artifacts.run_observability import RunObserver
from graphrag_v2.community import run_community_pipeline
from graphrag_v2.config import GraphRagConfig
from graphrag_v2.document import chunk_documents, scan_documents
from graphrag_v2.document.models import DocumentScanResult, TextUnit
from graphrag_v2.extraction import (
    BaseExtractor,
    LLMExtractor,
    MockExtractor,
    validate_extraction_result,
)
from graphrag_v2.extraction.cache import ExtractionCache
from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from graphrag_v2.graph_fusion import fuse_graph
from graphrag_v2.graph_fusion.fusion import DEFAULT_MIN_CONFIDENCE
from graphrag_v2.graph_store import GraphStoreError, GraphStoreStats, create_graph_store
from graphrag_v2.llm import create_chat_provider


def index_documents_only(
    input_path: str | Path,
    output_path: str | Path,
    config: GraphRagConfig,
    mode: str = "documents-only",
) -> dict:
    """Run the document processing and artifact writing stage."""
    metadata = start_run_metadata(output_path, mode=mode, input_path=input_path)
    observer = RunObserver.from_metadata(output_path, metadata)
    observer.run_start(mode=mode, input_path=input_path)
    observer.stage_start("document")
    try:
        _load_and_write_document_stage(
            input_path=input_path,
            output_path=output_path,
            config=config,
            mode=mode,
        )
        metadata = _read_current_metadata(output_path)
        observer.stage_end("document", counts=_document_counts(metadata))
        observer.run_end()
        return finish_run_metadata(
            output_path,
            stage_timings=observer.stage_timings,
        )
    except Exception as exc:
        if observer.failed_stage is None:
            observer.stage_failed("document", exc)
        observer.run_failed(exc)
        fail_run_metadata(
            output_path,
            exc=exc,
            failed_stage=observer.failed_stage,
            stage_timings=observer.stage_timings,
        )
        raise


async def index_extraction_only(
    input_path: str | Path,
    output_path: str | Path,
    config: GraphRagConfig,
    extractor_name: str = "mock",
    mode: str = "extraction-only",
) -> dict:
    """Run document processing plus candidate extraction."""
    metadata = start_run_metadata(output_path, mode=mode, input_path=input_path)
    observer = RunObserver.from_metadata(output_path, metadata)
    observer.run_start(mode=mode, input_path=input_path)
    current_stage = "document"
    extractor = None
    try:
        observer.stage_start("document")
        text_units = _load_and_write_document_stage(
            input_path=input_path,
            output_path=output_path,
            config=config,
            mode=mode,
        )
        metadata = _read_current_metadata(output_path)
        observer.stage_end("document", counts=_document_counts(metadata))

        current_stage = "extraction"
        extractor = _create_extractor(extractor_name, config)
        provider = _extractor_provider(extractor_name, extractor)
        model = _extractor_model(extractor)
        observer.stage_start("extraction", provider=provider, model=model)
        candidate_entities, candidate_relationships, candidate_triples = (
            await _extract_candidates(
                text_units=text_units,
                extractor=extractor,
                fail_on_invalid_chunk=config.extraction.fail_on_invalid_chunk,
            )
        )
        deduped_entities = _dedupe_entities(candidate_entities)
        deduped_relationships = _dedupe_relationships(candidate_relationships)
        deduped_triples = _dedupe_triples(candidate_triples)

        write_extraction_artifacts(
            output_path=output_path,
            candidate_entities=deduped_entities,
            candidate_relationships=deduped_relationships,
            candidate_triples=deduped_triples,
            extractor=extractor_name,
            extraction_metadata=_extractor_metadata(extractor),
        )
        observer.stage_end(
            "extraction",
            counts=_extraction_counts(
                deduped_entities,
                deduped_relationships,
                deduped_triples,
            ),
            provider=provider,
            model=model,
        )
        observer.run_end()
        return finish_run_metadata(
            output_path,
            stage_timings=observer.stage_timings,
        )
    except Exception as exc:
        provider = (
            _extractor_provider(extractor_name, extractor)
            if current_stage == "extraction"
            else None
        )
        model = _extractor_model(extractor) if current_stage == "extraction" else None
        if observer.failed_stage is None:
            observer.stage_failed(
                current_stage,
                exc,
                provider=provider,
                model=model,
            )
        observer.run_failed(exc)
        fail_run_metadata(
            output_path,
            exc=exc,
            failed_stage=observer.failed_stage,
            stage_timings=observer.stage_timings,
        )
        raise


async def index_fusion_only(
    input_path: str | Path,
    output_path: str | Path,
    config: GraphRagConfig,
    extractor_name: str = "mock",
    mode: str = "fusion-only",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    graph_store_provider: str = "json",
    enable_community: bool = False,
    strict_neo4j: bool = False,
) -> dict:
    """Run document processing, extraction, and graph fusion."""
    metadata = start_run_metadata(output_path, mode=mode, input_path=input_path)
    observer = RunObserver.from_metadata(output_path, metadata)
    observer.run_start(mode=mode, input_path=input_path)
    current_stage = "document"
    extractor = None
    if enable_community and graph_store_provider != "neo4j":
        current_stage = "community"
        exc = GraphStoreError("--community requires --graph-store neo4j.")
        observer.stage_start("community", provider=graph_store_provider)
        observer.stage_failed("community", exc, provider=graph_store_provider)
        observer.run_failed(exc)
        fail_run_metadata(
            output_path,
            exc=exc,
            failed_stage=observer.failed_stage,
            stage_timings=observer.stage_timings,
        )
        raise exc

    try:
        if strict_neo4j:
            current_stage = "graph_store"
            observer.stage_start("graph_store", provider=graph_store_provider)
            if graph_store_provider != "neo4j":
                raise GraphStoreError(
                    "Strict Neo4j indexing requires graph_store_provider='neo4j'."
                )
            graph_store = create_graph_store(
                provider="neo4j",
                index_path=output_path,
                config=config.graph_store,
            )
            preflight = getattr(graph_store, "preflight", None)
            if callable(preflight):
                preflight()
            observer.stage_end(
                "graph_store",
                counts={"preflight": True},
                provider=graph_store_provider,
            )
            current_stage = "document"

        observer.stage_start("document")
        text_units = _load_and_write_document_stage(
            input_path=input_path,
            output_path=output_path,
            config=config,
            mode=mode,
        )
        metadata = _read_current_metadata(output_path)
        observer.stage_end("document", counts=_document_counts(metadata))

        current_stage = "extraction"
        extractor = _create_extractor(extractor_name, config)
        provider = _extractor_provider(extractor_name, extractor)
        model = _extractor_model(extractor)
        observer.stage_start("extraction", provider=provider, model=model)
        candidate_entities, candidate_relationships, candidate_triples = (
            await _extract_candidates(
                text_units=text_units,
                extractor=extractor,
                fail_on_invalid_chunk=config.extraction.fail_on_invalid_chunk,
            )
        )
        deduped_entities = _dedupe_entities(candidate_entities)
        deduped_relationships = _dedupe_relationships(candidate_relationships)
        deduped_triples = _dedupe_triples(candidate_triples)
        write_extraction_artifacts(
            output_path=output_path,
            candidate_entities=deduped_entities,
            candidate_relationships=deduped_relationships,
            candidate_triples=deduped_triples,
            extractor=extractor_name,
            extraction_metadata=_extractor_metadata(extractor),
        )
        observer.stage_end(
            "extraction",
            counts=_extraction_counts(
                deduped_entities,
                deduped_relationships,
                deduped_triples,
            ),
            provider=provider,
            model=model,
        )
        current_stage = "fusion"
        observer.stage_start("fusion")
        fusion_result = fuse_graph(
            candidate_entities=deduped_entities,
            candidate_relationships=deduped_relationships,
            candidate_triples=deduped_triples,
            min_confidence=min_confidence,
            relation_schema_mode=config.fusion.relation_schema_mode,
        )
        observer.stage_end("fusion", counts=_fusion_counts(fusion_result))
        graph_store_written = False
        graph_store_error = None
        graph_store_stats = None
        current_stage = "graph_store"
        observer.stage_start("graph_store", provider=graph_store_provider)
        try:
            graph_store = create_graph_store(
                provider=graph_store_provider,
                index_path=output_path,
                config=config.graph_store,
            )
            graph_store_stats = graph_store.write_graph(fusion_result)
            graph_store_written = True
        except GraphStoreError as exc:
            graph_store_error = str(exc)
            if graph_store_provider != "json":
                write_fusion_artifacts(
                    output_path=output_path,
                    fusion_result=fusion_result,
                    min_confidence=min_confidence,
                    graph_store_provider=graph_store_provider,
                    graph_store_written=False,
                    graph_store_error=graph_store_error,
                    graph_store_metadata=_graph_store_metadata(graph_store_stats),
                )
                observer.stage_failed(
                    "graph_store",
                    exc,
                    provider=graph_store_provider,
                )
                raise

        write_fusion_artifacts(
            output_path=output_path,
            fusion_result=fusion_result,
            min_confidence=min_confidence,
            graph_store_provider=graph_store_provider,
            graph_store_written=graph_store_written,
            graph_store_error=graph_store_error,
            graph_store_metadata=_graph_store_metadata(graph_store_stats),
        )
        graph_store_counts = _graph_store_counts(graph_store_stats)
        if graph_store_error is not None:
            graph_store_counts.update(
                {
                    "written": False,
                    "error_type": "GraphStoreError",
                }
            )
        observer.stage_end(
            "graph_store",
            counts=graph_store_counts,
            provider=graph_store_provider,
        )
        if enable_community:
            current_stage = "community"
            observer.stage_start("community", provider=graph_store_provider)
            community_result = run_community_pipeline(
                output_path=output_path,
                config=config,
            )
            write_community_artifacts(
                output_path=output_path,
                communities=community_result.communities,
                reports=community_result.reports,
                algorithm=config.community.algorithm,
                reporter=config.community.reporter,
            )
            observer.stage_end(
                "community",
                counts=_community_counts(community_result),
                provider=graph_store_provider,
            )
        observer.run_end()
        return finish_run_metadata(
            output_path,
            stage_timings=observer.stage_timings,
        )
    except Exception as exc:
        provider = None
        model = None
        if current_stage == "extraction":
            provider = _extractor_provider(extractor_name, extractor)
            model = _extractor_model(extractor)
        elif current_stage == "graph_store":
            provider = graph_store_provider
        elif current_stage == "community":
            provider = graph_store_provider
        if observer.failed_stage is None:
            observer.stage_failed(
                current_stage,
                exc,
                provider=provider,
                model=model,
            )
        observer.run_failed(exc)
        fail_run_metadata(
            output_path,
            exc=exc,
            failed_stage=observer.failed_stage,
            stage_timings=observer.stage_timings,
        )
        raise


def _load_and_write_document_stage(
    input_path: str | Path,
    output_path: str | Path,
    config: GraphRagConfig,
    mode: str,
) -> list[TextUnit]:
    document_scan = scan_documents(
        input_path=input_path,
        encoding=config.input.encoding,
        unsupported_file_policy=config.input.unsupported_file_policy,
        max_file_size_bytes=config.input.max_file_size_bytes,
        max_document_count=config.input.max_document_count,
    )
    documents = document_scan.documents
    text_units = chunk_documents(
        documents=documents,
        chunk_size=config.chunks.size,
        chunk_overlap=config.chunks.overlap,
        encoding_model=config.chunks.encoding_model,
    )
    write_document_artifacts(
        output_path=output_path,
        documents=documents,
        text_units=text_units,
        input_path=input_path,
        chunk_size=config.chunks.size,
        chunk_overlap=config.chunks.overlap,
        mode=mode,
        document_scan=document_scan,
    )
    _raise_if_rejected_document_scan(document_scan)
    return text_units


def _raise_if_rejected_document_scan(document_scan: DocumentScanResult) -> None:
    if document_scan.num_rejected_files == 0:
        return
    rejected = [
        f"{Path(record.source_path).name}:{record.reason or 'unknown'}"
        for record in document_scan.records
        if record.status == "rejected"
    ]
    preview = ", ".join(rejected[:5])
    if len(rejected) > 5:
        preview = f"{preview}, ..."
    raise ValueError(
        f"Rejected {document_scan.num_rejected_files} input file(s): {preview}. "
        "See document_scan.json for the full input manifest."
    )


def _document_counts(metadata: dict) -> dict:
    return {
        "num_documents": metadata.get("num_documents"),
        "num_input_files": metadata.get("num_input_files"),
        "num_included_files": metadata.get("num_included_files"),
        "num_rejected_files": metadata.get("num_rejected_files"),
        "num_empty_documents": metadata.get("num_empty_documents"),
        "num_text_units": metadata.get("num_text_units"),
    }


def _extraction_counts(
    candidate_entities: list[ExtractedEntity],
    candidate_relationships: list[ExtractedRelationship],
    candidate_triples: list[CandidateTriple],
) -> dict:
    return {
        "num_candidate_entities": len(candidate_entities),
        "num_candidate_relationships": len(candidate_relationships),
        "num_candidate_triples": len(candidate_triples),
    }


def _fusion_counts(fusion_result) -> dict:
    return {
        "num_entities": len(fusion_result.entities),
        "num_relationships": len(fusion_result.relationships),
        "num_rejected_triples": len(fusion_result.rejected_triples),
    }


def _graph_store_counts(graph_store_stats: GraphStoreStats | None) -> dict:
    if graph_store_stats is None:
        return {}
    return {
        "num_text_units": graph_store_stats.num_text_units,
        "num_entities": graph_store_stats.num_entities,
        "num_relationships": graph_store_stats.num_relationships,
        "health_status": graph_store_stats.health_status,
    }


def _community_counts(community_result) -> dict:
    return {
        "num_communities": len(community_result.communities),
        "num_community_reports": len(community_result.reports),
    }


def _extractor_provider(extractor_name: str, extractor=None) -> str:
    if extractor_name == "mock":
        return "mock"
    provider_name = getattr(extractor, "provider_name", None)
    if provider_name:
        return str(provider_name)
    return extractor_name


def _extractor_model(extractor=None) -> str | None:
    model_name = getattr(extractor, "model_name", None)
    if model_name is None:
        return None
    return str(model_name)


def _read_current_metadata(output_path: str | Path) -> dict:
    metadata_path = Path(output_path) / "index_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    by_id = {entity.id: entity for entity in entities}
    return list(by_id.values())


def _graph_store_metadata(stats: GraphStoreStats | None) -> dict:
    if stats is None:
        return {}
    metadata = {
        "graph_store_provider": stats.provider,
        "graph_store_database": stats.database,
        "graph_store_num_text_units": stats.num_text_units,
        "graph_store_num_entities": stats.num_entities,
        "graph_store_num_relationships": stats.num_relationships,
        "graph_store_health_status": stats.health_status,
        "graph_store_schema_ready": stats.schema_ready,
        "graph_store_schema_constraint_count": (
            len(stats.schema_constraints)
            if stats.schema_constraints is not None
            else None
        ),
        "graph_store_schema_index_count": (
            len(stats.schema_indexes)
            if stats.schema_indexes is not None
            else None
        ),
        "graph_store_schema_version": stats.schema_version,
        "graph_store_missing_schema_constraints": stats.missing_schema_constraints,
        "graph_store_missing_schema_indexes": stats.missing_schema_indexes,
        "graph_store_write_strategy": stats.write_strategy,
        "graph_store_staging_index_id": stats.staging_index_id,
    }
    if stats.index_id is not None:
        metadata["graph_store_index_id"] = stats.index_id
    return {key: value for key, value in metadata.items() if value is not None}


def _dedupe_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    by_id = {relationship.id: relationship for relationship in relationships}
    return list(by_id.values())


def _dedupe_triples(triples: list[CandidateTriple]) -> list[CandidateTriple]:
    by_id = {triple.id: triple for triple in triples}
    return list(by_id.values())


async def _extract_candidates(
    text_units: list[TextUnit],
    extractor: BaseExtractor,
    fail_on_invalid_chunk: bool,
) -> tuple[list[ExtractedEntity], list[ExtractedRelationship], list[CandidateTriple]]:
    candidate_entities: list[ExtractedEntity] = []
    candidate_relationships: list[ExtractedRelationship] = []
    candidate_triples: list[CandidateTriple] = []
    concurrent_requests = max(1, int(getattr(extractor, "concurrent_requests", 1) or 1))
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def extract_one(text_unit: TextUnit) -> ExtractionResult | None:
        async with semaphore:
            return await _extract_candidate_unit(
                text_unit=text_unit,
                extractor=extractor,
                fail_on_invalid_chunk=fail_on_invalid_chunk,
            )

    results = await asyncio.gather(*(extract_one(text_unit) for text_unit in text_units))

    for result in results:
        if result is None:
            continue
        candidate_entities.extend(result.entities)
        candidate_relationships.extend(result.relationships)
        candidate_triples.extend(result.triples)

    return candidate_entities, candidate_relationships, candidate_triples


async def _extract_candidate_unit(
    text_unit: TextUnit,
    extractor: BaseExtractor,
    fail_on_invalid_chunk: bool,
) -> ExtractionResult | None:
    try:
        result = await extractor.extract(text_unit)
    except ValueError:
        if fail_on_invalid_chunk:
            raise
        return None

    errors = validate_extraction_result(result)
    if errors:
        _mark_failed_chunk(extractor, text_unit.chunk_id)
        if fail_on_invalid_chunk:
            raise ValueError(
                f"Invalid extraction result for {text_unit.chunk_id}: {errors}"
            )
        return None
    return result


def _create_extractor(extractor_name: str, config: GraphRagConfig) -> BaseExtractor:
    if extractor_name == "mock":
        return MockExtractor()
    if extractor_name == "llm":
        model_id = config.extraction.llm_model_id
        model_config = config.get_language_model_config(model_id)
        llm_client = create_chat_provider(
            provider=config.extraction.llm_provider,
            model_config=model_config,
            require_real=True,
        )
        return LLMExtractor(
            llm_client=llm_client,
            max_retries=config.extraction.max_retries,
            default_confidence=config.extraction.default_confidence,
            model_id=model_id,
            model_name=model_config.model,
            provider_name=config.extraction.llm_provider,
            requests_per_minute=config.extraction.requests_per_minute,
            concurrent_requests=config.extraction.concurrent_requests,
            max_prompt_tokens_per_chunk=config.extraction.max_prompt_tokens_per_chunk,
            max_total_tokens=config.extraction.max_total_tokens,
            max_estimated_cost=config.extraction.max_estimated_cost,
            salvage_on_parse_failure=config.extraction.salvage_on_parse_failure,
            max_gleanings=config.extraction.max_gleanings,
            cache=_create_extraction_cache(config),
        )
    raise ValueError(f"Unsupported extractor: {extractor_name}")


def _extractor_metadata(extractor: BaseExtractor) -> dict:
    get_metadata = getattr(extractor, "get_metadata", None)
    if callable(get_metadata):
        return dict(get_metadata())
    return {}


def _create_extraction_cache(config: GraphRagConfig) -> ExtractionCache | None:
    if not config.extraction.cache_enabled:
        return None
    cache_dir = (
        Path(config.extraction.cache_dir)
        if config.extraction.cache_dir is not None
        else Path(config.cache.base_dir) / "extraction"
    )
    if not cache_dir.is_absolute():
        cache_dir = Path(config.root_dir) / cache_dir
    return ExtractionCache(cache_dir=cache_dir)


def _mark_failed_chunk(
    extractor: BaseExtractor,
    chunk_id: str | None = None,
) -> None:
    stats = getattr(extractor, "stats", None)
    if stats is not None and hasattr(stats, "failed_chunks"):
        failed_chunk_ids = getattr(stats, "failed_chunk_ids", None)
        if isinstance(failed_chunk_ids, list) and chunk_id in failed_chunk_ids:
            return
        stats.failed_chunks += 1
        if isinstance(failed_chunk_ids, list) and chunk_id:
            failed_chunk_ids.append(chunk_id)
