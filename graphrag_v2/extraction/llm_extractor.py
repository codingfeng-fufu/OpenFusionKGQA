"""LLM-backed candidate knowledge extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction.base import BaseExtractor
from graphrag_v2.extraction.cache import ExtractionCache
from graphrag_v2.extraction.models import (
    CandidateTriple,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from graphrag_v2.extraction.prompts import (
    ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)

EXTRACTION_PROMPT_VERSION = "2026-06-08.v3"

EXTRACTION_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "type", "description", "confidence"],
            },
        },
        "relationships": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "source",
                    "target",
                    "relation",
                    "description",
                    "confidence",
                ],
            },
        },
    },
    "required": ["entities", "relationships"],
}


class LLMExtractionError(ValueError):
    """Raised when an LLM extraction response cannot be normalized."""


@dataclass
class LLMExtractionStats:
    """Runtime statistics for LLM extraction."""

    llm_calls: int = 0
    parse_failures: int = 0
    repair_attempts: int = 0
    gleaning_attempts: int = 0
    gleaning_failures: int = 0
    gleaned_entities: int = 0
    gleaned_relationships: int = 0
    salvaged_entities: int = 0
    salvaged_relationships: int = 0
    dropped_entities: int = 0
    dropped_relationships: int = 0
    failed_chunks: int = 0
    failed_chunk_ids: list[str] = field(default_factory=list)
    budget_exceeded: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    elapsed_seconds: float = 0.0


class LLMExtractor(BaseExtractor):
    """LLM extractor with JSON parsing, retry, and deterministic IDs."""

    def __init__(
        self,
        llm_client: Any | None = None,
        max_retries: int = 2,
        default_confidence: float = 0.7,
        model_id: str = "default_chat_model",
        model_name: str | None = None,
        provider_name: str | None = None,
        requests_per_minute: int | None = None,
        concurrent_requests: int = 1,
        max_prompt_tokens_per_chunk: int | None = None,
        max_total_tokens: int | None = None,
        max_estimated_cost: float | None = None,
        salvage_on_parse_failure: bool = True,
        max_gleanings: int = 1,
        cache: ExtractionCache | None = None,
    ):
        if llm_client is None:
            raise LLMExtractionError(
                "LLMExtractor requires an explicit LLM provider client."
            )
        if requests_per_minute is not None and requests_per_minute < 1:
            raise LLMExtractionError("requests_per_minute must be at least 1")
        self.llm_client = llm_client
        self.max_retries = max(1, max_retries)
        self.default_confidence = default_confidence
        self.model_id = model_id
        self.model_name = (
            model_name
            or getattr(self.llm_client, "model_name", None)
            or getattr(self.llm_client, "model", "")
        )
        self.provider_name = provider_name or getattr(
            self.llm_client,
            "provider_name",
            "unknown",
        )
        self.requests_per_minute = requests_per_minute
        self.concurrent_requests = max(1, concurrent_requests)
        self.max_prompt_tokens_per_chunk = max_prompt_tokens_per_chunk
        self.max_total_tokens = max_total_tokens
        self.max_estimated_cost = max_estimated_cost
        self.salvage_on_parse_failure = salvage_on_parse_failure
        self.max_gleanings = max(0, max_gleanings)
        self.cache = cache
        self._request_window_seconds = 60.0
        self._request_timestamps: deque[float] = deque()
        self._rate_limit_lock = asyncio.Lock()
        self.stats = LLMExtractionStats()

    async def extract(self, text_unit: TextUnit) -> ExtractionResult:
        """Extract candidate entities and relationships from one text unit."""
        started_at = time.perf_counter()
        messages = self._build_messages(text_unit)
        self._check_prompt_budget(text_unit, messages)
        cache_key = self._cache_key(text_unit, messages)
        cached_payload = self._get_cached_payload(cache_key, text_unit)
        if cached_payload is not None:
            self.stats.elapsed_seconds += time.perf_counter() - started_at
            return self._payload_to_result(cached_payload, text_unit)

        last_error: Exception | None = None
        last_response = ""

        for attempt in range(self.max_retries):
            self._check_runtime_budget(text_unit.chunk_id)
            await self._wait_for_rate_limit()
            self.stats.llm_calls += 1
            try:
                response = await asyncio.to_thread(
                    self._chat_completion,
                    messages,
                )
            except Exception as exc:
                self._record_failed_chunk(text_unit.chunk_id)
                self.stats.elapsed_seconds += time.perf_counter() - started_at
                raise LLMExtractionError(
                    "LLM extraction failed "
                    f"(chunk_id={text_unit.chunk_id}, stage=llm_call): {exc}"
                ) from exc
            last_response = str(response)
            try:
                payload = _extract_json_object(last_response)
                payload = await self._run_gleaning(text_unit, payload)
                result = self._payload_to_result(payload, text_unit)
                self._set_cached_payload(cache_key, payload)
                self.stats.elapsed_seconds += time.perf_counter() - started_at
                return result
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
                self.stats.parse_failures += 1
                if attempt < self.max_retries - 1:
                    self.stats.repair_attempts += 1
                if last_response.strip():
                    messages = self._build_repair_messages(
                        text_unit=text_unit,
                        previous_response=last_response,
                        error=str(exc),
                    )
                else:
                    messages = self._build_messages(text_unit)

        if self.salvage_on_parse_failure:
            result = self._salvage_result(last_response, text_unit)
            if result.entities or result.relationships:
                self.stats.elapsed_seconds += time.perf_counter() - started_at
                return result

        self._record_failed_chunk(text_unit.chunk_id)
        self.stats.elapsed_seconds += time.perf_counter() - started_at
        raise LLMExtractionError(
            "LLM extraction failed "
            f"(chunk_id={text_unit.chunk_id}, stage=parse_or_repair): {last_error}"
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return extraction metadata safe to persist in index metadata."""
        client_stats = _client_stats(self.llm_client)
        return {
            "llm_provider": self.provider_name,
            "llm_model_id": self.model_id,
            "llm_model_name": self.model_name,
            "llm_mock_mode": bool(getattr(self.llm_client, "mock_mode", False)),
            "llm_total_calls": self.stats.llm_calls,
            "llm_total_errors": int(getattr(self.llm_client, "total_errors", 0)),
            "llm_total_tokens": int(client_stats.get("total_tokens", 0) or 0),
            "llm_prompt_tokens": int(client_stats.get("prompt_tokens", 0) or 0),
            "llm_completion_tokens": int(
                client_stats.get("completion_tokens", 0) or 0
            ),
            "llm_total_latency_seconds": float(
                client_stats.get("total_latency_seconds", 0.0) or 0.0
            ),
            "llm_max_latency_seconds": float(
                client_stats.get("max_latency_seconds", 0.0) or 0.0
            ),
            "llm_average_latency_seconds": float(
                client_stats.get("average_latency_seconds", 0.0) or 0.0
            ),
            "llm_estimated_cost": client_stats.get("estimated_cost"),
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "extraction_parse_failures": self.stats.parse_failures,
            "extraction_repair_attempts": self.stats.repair_attempts,
            "extraction_max_gleanings": self.max_gleanings,
            "extraction_gleaning_attempts": self.stats.gleaning_attempts,
            "extraction_gleaning_failures": self.stats.gleaning_failures,
            "extraction_gleaned_entities": self.stats.gleaned_entities,
            "extraction_gleaned_relationships": self.stats.gleaned_relationships,
            "extraction_salvaged_entities": self.stats.salvaged_entities,
            "extraction_salvaged_relationships": self.stats.salvaged_relationships,
            "extraction_dropped_entities": self.stats.dropped_entities,
            "extraction_dropped_relationships": self.stats.dropped_relationships,
            "extraction_failed_chunks": self.stats.failed_chunks,
            "extraction_failed_chunk_ids": list(self.stats.failed_chunk_ids),
            "extraction_budget_exceeded": self.stats.budget_exceeded,
            "extraction_requests_per_minute": self.requests_per_minute,
            "extraction_concurrent_requests": self.concurrent_requests,
            "extraction_max_prompt_tokens_per_chunk": self.max_prompt_tokens_per_chunk,
            "extraction_max_total_tokens": self.max_total_tokens,
            "extraction_max_estimated_cost": self.max_estimated_cost,
            "extraction_salvage_on_parse_failure": self.salvage_on_parse_failure,
            "extraction_cache_enabled": self.cache is not None,
            "extraction_cache_hits": self.stats.cache_hits,
            "extraction_cache_misses": self.stats.cache_misses,
            "extraction_elapsed_seconds": round(self.stats.elapsed_seconds, 6),
        }

    def _build_messages(self, text_unit: TextUnit) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_extraction_prompt(
                    chunk_id=text_unit.chunk_id,
                    source_path=text_unit.source_path,
                    text=text_unit.text,
                ),
            },
        ]

    def _build_repair_messages(
        self,
        text_unit: TextUnit,
        previous_response: str,
        error: str,
    ) -> list[dict[str, str]]:
        repair_prompt = (
            "Your previous extraction response was invalid.\n"
            f"Error: {error}\n"
            "Rewrite the extraction as valid JSON using the required schema. "
            "Return only JSON.\n\n"
            f"Previous response:\n{previous_response}\n\n"
            + build_extraction_prompt(
                chunk_id=text_unit.chunk_id,
                source_path=text_unit.source_path,
                text=text_unit.text,
            )
        )
        return [
            {
                "role": "system",
                "content": ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT,
            },
            {"role": "user", "content": repair_prompt},
        ]

    def _build_gleaning_messages(
        self,
        text_unit: TextUnit,
        payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        existing = json.dumps(
            {
                "entities": payload.get("entities", []),
                "relationships": payload.get("relationships", []),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = (
            "Review the text unit for missing knowledge graph candidates.\n"
            "Return only compact valid JSON with the same entities/relationships "
            "shape.\n"
            "Return ONLY newly missed entities and relationships. Do not repeat "
            "items already extracted. Every new relationship must reference an "
            "entity in either the existing entities or your new entities. Keep "
            "descriptions under 12 words. Return empty lists if nothing is missing.\n\n"
            f"Existing extraction JSON:\n{existing}\n\n"
            + build_extraction_prompt(
                chunk_id=text_unit.chunk_id,
                source_path=text_unit.source_path,
                text=text_unit.text,
            )
        )
        return [
            {
                "role": "system",
                "content": ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]

    def _build_gleaning_repair_messages(
        self,
        text_unit: TextUnit,
        payload: dict[str, Any],
        previous_response: str,
        error: str,
    ) -> list[dict[str, str]]:
        repair_prompt = (
            "Your previous gleaning response was invalid.\n"
            f"Error: {error}\n"
            "Rewrite only the missed extraction as compact valid JSON. "
            "Return only JSON.\n\n"
            f"Previous response:\n{previous_response}\n\n"
            + self._build_gleaning_messages(text_unit, payload)[1]["content"]
        )
        return [
            {
                "role": "system",
                "content": ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT,
            },
            {"role": "user", "content": repair_prompt},
        ]

    def _chat_completion(self, messages: list[dict[str, str]]) -> str | Any:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
        }
        if bool(getattr(self.llm_client, "supports_guided_json", False)):
            kwargs["guided_json"] = EXTRACTION_GUIDED_JSON_SCHEMA
        elif bool(getattr(self.llm_client, "supports_response_format_json", False)):
            kwargs["response_format_json"] = True
        return self.llm_client.chat_completion(**kwargs)

    async def _run_gleaning(
        self,
        text_unit: TextUnit,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.max_gleanings <= 0:
            return payload

        current_payload = payload
        for _ in range(self.max_gleanings):
            self.stats.gleaning_attempts += 1
            messages = self._build_gleaning_messages(text_unit, current_payload)
            self._check_prompt_budget(text_unit, messages)
            gleaning_payload = await self._extract_gleaning_payload(
                text_unit=text_unit,
                current_payload=current_payload,
                messages=messages,
            )
            if gleaning_payload is None:
                break
            current_payload, added_entities, added_relationships = _merge_payloads(
                current_payload,
                gleaning_payload,
            )
            self.stats.gleaned_entities += added_entities
            self.stats.gleaned_relationships += added_relationships
            if added_entities == 0 and added_relationships == 0:
                break
        return current_payload

    async def _extract_gleaning_payload(
        self,
        *,
        text_unit: TextUnit,
        current_payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        last_response = ""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._check_runtime_budget(text_unit.chunk_id)
            await self._wait_for_rate_limit()
            self.stats.llm_calls += 1
            try:
                response = await asyncio.to_thread(
                    self._chat_completion,
                    messages,
                )
            except Exception:
                self.stats.gleaning_failures += 1
                return None
            last_response = str(response)
            try:
                return _extract_json_object(last_response)
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
                self.stats.parse_failures += 1
                if attempt < self.max_retries - 1:
                    self.stats.repair_attempts += 1
                if last_response.strip():
                    messages = self._build_gleaning_repair_messages(
                        text_unit=text_unit,
                        payload=current_payload,
                        previous_response=last_response,
                        error=str(exc),
                    )
                else:
                    messages = self._build_gleaning_messages(
                        text_unit,
                        current_payload,
                    )

        self.stats.gleaning_failures += 1
        return None

    def _payload_to_result(
        self,
        payload: dict[str, Any],
        text_unit: TextUnit,
    ) -> ExtractionResult:
        entities = self._extract_entities(payload, text_unit)
        entity_names = {entity.name for entity in entities}
        relationships = self._extract_relationships(payload, text_unit, entity_names)
        triples = [
            CandidateTriple(
                id=_stable_id("candidate_triple", relationship.id),
                source_name=relationship.source,
                target_name=relationship.target,
                relation_mention=relationship.relation,
                canonical_relation=None,
                description=relationship.description,
                extraction_confidence=relationship.confidence,
                relation_alignment_score=None,
                evidence_support_score=None,
                graph_consistency_score=None,
                triple_score=None,
                status="candidate",
                evidence_chunk_ids=relationship.evidence_chunk_ids,
                metadata={
                    "relationship_id": relationship.id,
                    "extractor": "llm",
                    "chunk_id": text_unit.chunk_id,
                },
            )
            for relationship in relationships
        ]
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            triples=triples,
        )

    def _salvage_result(self, response: str, text_unit: TextUnit) -> ExtractionResult:
        payload = _salvage_json_payload(response)
        if payload is None:
            return ExtractionResult()
        before_dropped_entities = self.stats.dropped_entities
        before_dropped_relationships = self.stats.dropped_relationships
        try:
            result = self._payload_to_result(payload, text_unit)
        except LLMExtractionError:
            return ExtractionResult()
        self.stats.salvaged_entities += len(result.entities)
        self.stats.salvaged_relationships += len(result.relationships)
        self.stats.dropped_entities = max(
            self.stats.dropped_entities,
            before_dropped_entities + _count_dropped_entities(payload, result),
        )
        self.stats.dropped_relationships = max(
            self.stats.dropped_relationships,
            before_dropped_relationships + _count_dropped_relationships(payload, result),
        )
        return result

    def _extract_entities(
        self,
        payload: dict[str, Any],
        text_unit: TextUnit,
    ) -> list[ExtractedEntity]:
        raw_entities = payload.get("entities", [])
        if not isinstance(raw_entities, list):
            raise LLMExtractionError("entities must be a list")

        entities_by_name: dict[str, ExtractedEntity] = {}
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                self.stats.dropped_entities += 1
                continue
            name = _clean_text(raw_entity.get("name"))
            if not name:
                self.stats.dropped_entities += 1
                continue
            entity_type = _clean_text(raw_entity.get("type")) or "Entity"
            description = _clean_text(raw_entity.get("description")) or name
            confidence = _confidence(
                raw_entity.get("confidence"),
                default=self.default_confidence,
            )
            entities_by_name[name] = ExtractedEntity(
                id=_stable_id("candidate_entity", _canonical_key(name)),
                name=name,
                type=entity_type,
                description=description,
                confidence=confidence,
                evidence_chunk_ids=[text_unit.chunk_id],
                metadata={
                    "extractor": "llm",
                    "chunk_id": text_unit.chunk_id,
                    "source_path": text_unit.source_path,
                },
            )
        return list(entities_by_name.values())

    def _extract_relationships(
        self,
        payload: dict[str, Any],
        text_unit: TextUnit,
        entity_names: set[str],
    ) -> list[ExtractedRelationship]:
        raw_relationships = payload.get("relationships", [])
        if not isinstance(raw_relationships, list):
            raise LLMExtractionError("relationships must be a list")

        relationships: list[ExtractedRelationship] = []
        for index, raw_relationship in enumerate(raw_relationships):
            if not isinstance(raw_relationship, dict):
                self.stats.dropped_relationships += 1
                continue
            source = _clean_text(raw_relationship.get("source"))
            target = _clean_text(raw_relationship.get("target"))
            if source not in entity_names or target not in entity_names:
                self.stats.dropped_relationships += 1
                continue
            relation = _normalize_relation(raw_relationship.get("relation"))
            if not relation:
                self.stats.dropped_relationships += 1
                continue
            description = (
                _clean_text(raw_relationship.get("description"))
                or f"{source} {relation} {target}"
            )
            confidence = _confidence(
                raw_relationship.get("confidence"),
                default=self.default_confidence,
            )
            relationship_id = _stable_id(
                "candidate_relationship",
                text_unit.chunk_id,
                _canonical_key(source),
                relation,
                _canonical_key(target),
            )
            relationships.append(
                ExtractedRelationship(
                    id=relationship_id,
                    source=source,
                    target=target,
                    relation=relation,
                    description=description,
                    confidence=confidence,
                    evidence_chunk_ids=[text_unit.chunk_id],
                    metadata={
                        "extractor": "llm",
                        "chunk_id": text_unit.chunk_id,
                        "relationship_index": index,
                    },
                )
            )
        return relationships

    def _check_prompt_budget(
        self,
        text_unit: TextUnit,
        messages: list[dict[str, str]],
    ) -> None:
        if self.max_prompt_tokens_per_chunk is None:
            return
        prompt_tokens = getattr(text_unit, "n_tokens", 0) or _estimate_tokens(
            "\n".join(message["content"] for message in messages)
        )
        if int(prompt_tokens) > self.max_prompt_tokens_per_chunk:
            self.stats.budget_exceeded = True
            self._record_failed_chunk(text_unit.chunk_id)
            raise LLMExtractionError(
                "LLM extraction prompt budget exceeded "
                f"(chunk_id={text_unit.chunk_id}, "
                f"prompt_tokens={prompt_tokens}, "
                f"max_prompt_tokens_per_chunk={self.max_prompt_tokens_per_chunk})"
            )

    def _check_runtime_budget(self, chunk_id: str | None = None) -> None:
        client_stats = _client_stats(self.llm_client)
        total_tokens = int(client_stats.get("total_tokens", 0) or 0)
        if (
            self.max_total_tokens is not None
            and total_tokens >= self.max_total_tokens
        ):
            self.stats.budget_exceeded = True
            self._record_failed_chunk(chunk_id)
            raise LLMExtractionError(
                "LLM extraction token budget exceeded "
                f"(total_tokens={total_tokens}, max_total_tokens={self.max_total_tokens})"
            )
        estimated_cost = client_stats.get("estimated_cost")
        if (
            self.max_estimated_cost is not None
            and estimated_cost is not None
            and float(estimated_cost) >= self.max_estimated_cost
        ):
            self.stats.budget_exceeded = True
            self._record_failed_chunk(chunk_id)
            raise LLMExtractionError(
                "LLM extraction cost budget exceeded "
                f"(estimated_cost={estimated_cost}, "
                f"max_estimated_cost={self.max_estimated_cost})"
            )

    async def _wait_for_rate_limit(self) -> None:
        if self.requests_per_minute is None:
            return

        async with self._rate_limit_lock:
            while True:
                now = time.monotonic()
                while (
                    self._request_timestamps
                    and now
                    - self._request_timestamps[0]
                    >= self._request_window_seconds
                ):
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < self.requests_per_minute:
                    self._request_timestamps.append(now)
                    return

                sleep_seconds = max(
                    0.0,
                    self._request_window_seconds
                    - (now - self._request_timestamps[0]),
                )
                await asyncio.sleep(sleep_seconds)

    def _cache_key(
        self,
        text_unit: TextUnit,
        messages: list[dict[str, str]],
    ) -> str:
        fingerprint = {
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "default_confidence": self.default_confidence,
            "max_retries": self.max_retries,
            "max_gleanings": self.max_gleanings,
            "salvage_on_parse_failure": self.salvage_on_parse_failure,
            "supports_guided_json": bool(
                getattr(self.llm_client, "supports_guided_json", False)
            ),
            "chunk_id": text_unit.chunk_id,
            "source_path": text_unit.source_path,
            "text_sha256": hashlib.sha256(
                text_unit.text.encode("utf-8")
            ).hexdigest(),
            "messages_sha256": hashlib.sha256(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }
        serialized = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_cached_payload(
        self,
        cache_key: str,
        text_unit: TextUnit,
    ) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        cached = self.cache.get(cache_key)
        if not _looks_like_extraction_payload(cached):
            self.stats.cache_misses += 1
            return None
        self.stats.cache_hits += 1
        return cached

    def _set_cached_payload(self, cache_key: str, payload: dict[str, Any]) -> None:
        if self.cache is None:
            return
        self.cache.set(cache_key, payload)

    def _record_failed_chunk(self, chunk_id: str | None) -> None:
        self.stats.failed_chunks += 1
        if chunk_id and chunk_id not in self.stats.failed_chunk_ids:
            self.stats.failed_chunk_ids.append(chunk_id)


def _extract_json_object(response: str) -> dict[str, Any]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise LLMExtractionError("response JSON must be an object")
    return payload


def _salvage_json_payload(response: str) -> dict[str, Any] | None:
    candidates = _extract_array_candidates(response, "entities")
    relationships = _extract_array_candidates(response, "relationships")
    if not candidates and not relationships:
        return None
    return {
        "entities": candidates,
        "relationships": relationships,
    }


def _extract_array_candidates(response: str, key: str) -> list[Any]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', response)
    if match is None:
        return []
    start = match.end() - 1
    end = _find_matching_bracket(response, start)
    if end is None:
        end = response.rfind("]")
    if end == -1 or end <= start:
        return []
    try:
        payload = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _find_matching_bracket(value: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _count_dropped_entities(
    payload: dict[str, Any],
    result: ExtractionResult,
) -> int:
    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        return 0
    return max(0, len(raw_entities) - len(result.entities))


def _count_dropped_relationships(
    payload: dict[str, Any],
    result: ExtractionResult,
) -> int:
    raw_relationships = payload.get("relationships", [])
    if not isinstance(raw_relationships, list):
        return 0
    return max(0, len(raw_relationships) - len(result.relationships))


def _merge_payloads(
    base_payload: dict[str, Any],
    additional_payload: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    base_entities = _payload_list(base_payload, "entities")
    base_relationships = _payload_list(base_payload, "relationships")
    additional_entities = _payload_list(additional_payload, "entities")
    additional_relationships = _payload_list(additional_payload, "relationships")

    merged_entities = list(base_entities)
    entity_keys = {
        _canonical_key(_clean_text(entity.get("name")))
        for entity in merged_entities
        if isinstance(entity, dict) and _clean_text(entity.get("name"))
    }
    added_entities = 0
    for entity in additional_entities:
        if not isinstance(entity, dict):
            continue
        name = _clean_text(entity.get("name"))
        key = _canonical_key(name)
        if not key or key in entity_keys:
            continue
        merged_entities.append(entity)
        entity_keys.add(key)
        added_entities += 1

    merged_relationships = list(base_relationships)
    relationship_keys = {
        _relationship_key(relationship)
        for relationship in merged_relationships
        if isinstance(relationship, dict)
    }
    relationship_keys.discard(None)
    added_relationships = 0
    for relationship in additional_relationships:
        if not isinstance(relationship, dict):
            continue
        key = _relationship_key(relationship)
        if key is None or key in relationship_keys:
            continue
        source_key, _, target_key = key
        if source_key not in entity_keys or target_key not in entity_keys:
            continue
        merged_relationships.append(relationship)
        relationship_keys.add(key)
        added_relationships += 1

    return (
        {
            **base_payload,
            "entities": merged_entities,
            "relationships": merged_relationships,
        },
        added_entities,
        added_relationships,
    )


def _payload_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def _relationship_key(relationship: dict[str, Any]) -> tuple[str, str, str] | None:
    source = _canonical_key(_clean_text(relationship.get("source")))
    target = _canonical_key(_clean_text(relationship.get("target")))
    relation = _normalize_relation(relationship.get("relation"))
    if not source or not target or not relation:
        return None
    return source, relation, target


def _client_stats(llm_client: Any) -> dict[str, Any]:
    get_stats = getattr(llm_client, "get_stats", None)
    if callable(get_stats):
        return dict(get_stats())
    return {}


def _looks_like_extraction_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("entities"), list)
        and isinstance(value.get("relationships"), list)
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_relation(value: Any) -> str:
    relation = _clean_text(value).lower()
    relation = re.sub(r"[^a-z0-9_\-\s]+", "", relation)
    relation = re.sub(r"[\s\-]+", "_", relation).strip("_")
    return relation


def _confidence(value: Any, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    if confidence > 1.0 and confidence <= 10.0:
        confidence = confidence / 10.0
    return max(0.0, min(1.0, confidence))


def _canonical_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _stable_id(prefix: str, *parts: str) -> str:
    content = ":".join(parts)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4)
