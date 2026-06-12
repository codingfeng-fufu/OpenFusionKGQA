"""Tests for LLM-backed extraction."""

from __future__ import annotations

import json

import pytest

from graphrag_v2.artifacts import LLM_EXTRACTION_METADATA_KEYS
from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction.cache import ExtractionCache
from graphrag_v2.extraction import LLMExtractionError, LLMExtractor
from graphrag_v2.extraction.llm_extractor import EXTRACTION_PROMPT_VERSION
from graphrag_v2.extraction.prompts import build_extraction_prompt
from graphrag_v2.extraction.validators import validate_extraction_result


class StubLLMClient:
    def __init__(self, responses: list[str], stats: dict | None = None):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []
        self.call_kwargs: list[dict] = []
        self.total_errors = 0
        self.stats = stats or {}
        self.provider_name = "stub"
        self.model_name = "stub-model"
        self.mock_mode = False
        self.supports_guided_json = False
        self.supports_response_format_json = False

    def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs,
            }
        )
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    def get_stats(self):
        return self.stats


def test_extraction_prompt_requests_type_and_storage_relationships():
    prompt = build_extraction_prompt(
        chunk_id="chunk_1",
        source_path="doc.txt",
        text="Neo4j is a graph database used to store entities and relationships.",
    )

    assert 'relation "is_a"' in prompt
    assert 'relation "stores"' in prompt
    assert "under 12 words" in prompt
    assert "Preserve open predicates" in prompt


@pytest.mark.asyncio
async def test_llm_extractor_parses_valid_json():
    client = StubLLMClient([_valid_response()])
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert [entity.name for entity in result.entities] == [
        "GraphRAG",
        "Knowledge Graph",
    ]
    assert result.relationships[0].relation == "uses"
    assert result.triples[0].source_name == "GraphRAG"
    assert result.entities[0].metadata["extractor"] == "llm"
    assert validate_extraction_result(result) == []
    assert extractor.get_metadata()["llm_total_calls"] == 1
    assert extractor.get_metadata()["llm_provider"] == "stub"
    assert extractor.get_metadata()["llm_mock_mode"] is False
    assert extractor.get_metadata()["extraction_parse_failures"] == 0
    assert extractor.get_metadata()["llm_total_tokens"] == 0
    assert extractor.get_metadata()["llm_estimated_cost"] is None
    assert extractor.get_metadata()["extraction_prompt_version"] == (
        EXTRACTION_PROMPT_VERSION
    )
    _assert_metadata_keys(extractor.get_metadata(), LLM_EXTRACTION_METADATA_KEYS)


@pytest.mark.asyncio
async def test_llm_extractor_runs_one_gleaning_round_by_default():
    client = StubLLMClient([_partial_response(), _gleaning_response()])
    extractor = LLMExtractor(llm_client=client)

    result = await extractor.extract(_text_unit())

    assert len(client.calls) == 2
    assert "missing knowledge graph candidates" in client.calls[1][1]["content"]
    assert [entity.name for entity in result.entities] == [
        "GraphRAG",
        "Knowledge Graph",
    ]
    assert result.relationships[0].relation == "uses"
    metadata = extractor.get_metadata()
    assert metadata["extraction_max_gleanings"] == 1
    assert metadata["extraction_gleaning_attempts"] == 1
    assert metadata["extraction_gleaning_failures"] == 0
    assert metadata["extraction_gleaned_entities"] == 1
    assert metadata["extraction_gleaned_relationships"] == 1
    assert metadata["llm_total_calls"] == 2


@pytest.mark.asyncio
async def test_llm_extractor_passes_guided_json_to_supported_clients():
    client = StubLLMClient([_valid_response()])
    client.supports_guided_json = True
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    await extractor.extract(_text_unit())

    guided_json = client.call_kwargs[0]["guided_json"]
    assert guided_json["required"] == ["entities", "relationships"]
    assert "entities" in guided_json["properties"]
    assert "relationships" in guided_json["properties"]


@pytest.mark.asyncio
async def test_llm_extractor_passes_response_format_json_to_supported_clients():
    client = StubLLMClient([_valid_response()])
    client.supports_response_format_json = True
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    await extractor.extract(_text_unit())

    assert client.call_kwargs[0]["response_format_json"] is True
    assert "guided_json" not in client.call_kwargs[0]


@pytest.mark.asyncio
async def test_llm_extractor_metadata_includes_client_stats():
    client = StubLLMClient(
        [_valid_response()],
        stats={
            "total_tokens": 15,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_latency_seconds": 0.25,
            "max_latency_seconds": 0.25,
            "average_latency_seconds": 0.25,
            "estimated_cost": 0.002,
        },
    )
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    await extractor.extract(_text_unit())

    metadata = extractor.get_metadata()
    assert metadata["llm_total_tokens"] == 15
    assert metadata["llm_prompt_tokens"] == 10
    assert metadata["llm_completion_tokens"] == 5
    assert metadata["llm_total_latency_seconds"] == 0.25
    assert metadata["llm_estimated_cost"] == 0.002


@pytest.mark.asyncio
async def test_llm_extractor_accepts_fenced_json():
    client = StubLLMClient([f"```json\n{_valid_response()}\n```"])
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert len(result.entities) == 2
    assert len(result.relationships) == 1


@pytest.mark.asyncio
async def test_llm_extractor_repairs_invalid_response():
    client = StubLLMClient(["not json", _valid_response()])
    extractor = LLMExtractor(llm_client=client, max_retries=2, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert len(client.calls) == 2
    assert "previous extraction response was invalid" in client.calls[1][1]["content"]
    assert result.relationships[0].source == "GraphRAG"
    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 2
    assert metadata["extraction_parse_failures"] == 1
    assert metadata["extraction_repair_attempts"] == 1


@pytest.mark.asyncio
async def test_llm_extractor_retries_original_prompt_after_empty_response():
    client = StubLLMClient(["", _valid_response()])
    extractor = LLMExtractor(llm_client=client, max_retries=2, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert len(client.calls) == 2
    assert "previous extraction response was invalid" not in client.calls[1][1]["content"]
    assert result.relationships[0].source == "GraphRAG"
    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 2
    assert metadata["extraction_parse_failures"] == 1


@pytest.mark.asyncio
async def test_llm_extractor_records_failed_chunk_on_provider_exception():
    class FailingClient(StubLLMClient):
        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False, **kwargs):
            self.calls.append(messages)
            self.call_kwargs.append({"temperature": temperature, **kwargs})
            raise TimeoutError("provider timed out")

    client = FailingClient([])
    extractor = LLMExtractor(llm_client=client, max_retries=2, max_gleanings=0)

    with pytest.raises(LLMExtractionError, match="stage=llm_call"):
        await extractor.extract(_text_unit())

    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 1
    assert metadata["extraction_failed_chunks"] == 1
    assert metadata["extraction_failed_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_llm_extractor_keeps_initial_result_when_gleaning_repair_fails():
    client = StubLLMClient([_partial_response(), "not json", "still not json"])
    extractor = LLMExtractor(llm_client=client, max_retries=2)

    result = await extractor.extract(_text_unit())

    assert [entity.name for entity in result.entities] == ["GraphRAG"]
    assert result.relationships == []
    metadata = extractor.get_metadata()
    assert metadata["extraction_gleaning_attempts"] == 1
    assert metadata["extraction_gleaning_failures"] == 1
    assert metadata["extraction_parse_failures"] == 2
    assert metadata["extraction_failed_chunks"] == 0


@pytest.mark.asyncio
async def test_llm_extractor_raises_after_retry_exhaustion():
    client = StubLLMClient(["not json", "still not json"])
    extractor = LLMExtractor(
        llm_client=client,
        max_retries=2,
        salvage_on_parse_failure=False,
        max_gleanings=0,
    )

    with pytest.raises(LLMExtractionError, match="chunk_id=chunk_1"):
        await extractor.extract(_text_unit())
    metadata = extractor.get_metadata()
    assert metadata["extraction_failed_chunks"] == 1
    assert metadata["extraction_failed_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_llm_extractor_salvages_partial_payload_after_repair_failure():
    partial_response = """
    The response is almost valid:
    {
      "entities": [
        {"name": "GraphRAG", "type": "Technology", "description": "GraphRAG", "confidence": 0.9},
        {"type": "Broken"}
      ],
      "relationships": [
        {"source": "GraphRAG", "target": "Missing", "relation": "mentions"},
        {"source": "GraphRAG", "target": "GraphRAG", "relation": "related-to"}
      ],
    """
    client = StubLLMClient(["not json", partial_response])
    extractor = LLMExtractor(llm_client=client, max_retries=2, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert [entity.name for entity in result.entities] == ["GraphRAG"]
    assert len(result.relationships) == 1
    assert result.relationships[0].relation == "related_to"
    metadata = extractor.get_metadata()
    assert metadata["extraction_parse_failures"] == 2
    assert metadata["extraction_salvaged_entities"] == 1
    assert metadata["extraction_salvaged_relationships"] == 1
    assert metadata["extraction_dropped_entities"] == 1
    assert metadata["extraction_dropped_relationships"] == 1
    assert metadata["extraction_failed_chunks"] == 0


@pytest.mark.asyncio
async def test_llm_extractor_filters_relationships_with_missing_entities():
    payload = json.loads(_valid_response())
    payload["relationships"].append(
        {
            "source": "GraphRAG",
            "target": "Missing",
            "relation": "mentions",
            "description": "Invalid endpoint.",
            "confidence": 0.8,
        }
    )
    client = StubLLMClient([json.dumps(payload)])
    extractor = LLMExtractor(llm_client=client, max_gleanings=0)

    result = await extractor.extract(_text_unit())

    assert len(result.relationships) == 1
    assert result.relationships[0].target == "Knowledge Graph"
    assert validate_extraction_result(result) == []
    assert extractor.get_metadata()["extraction_dropped_relationships"] == 1


@pytest.mark.asyncio
async def test_llm_extractor_rejects_prompt_over_budget():
    client = StubLLMClient([_valid_response()])
    extractor = LLMExtractor(
        llm_client=client,
        max_prompt_tokens_per_chunk=2,
        max_gleanings=0,
    )

    with pytest.raises(LLMExtractionError, match="prompt budget exceeded"):
        await extractor.extract(_text_unit())

    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 0
    assert metadata["extraction_budget_exceeded"] is True
    assert metadata["extraction_failed_chunks"] == 1
    assert metadata["extraction_failed_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_llm_extractor_uses_cache_for_repeated_chunk():
    client = StubLLMClient([_valid_response()])
    cache = ExtractionCache()
    extractor = LLMExtractor(llm_client=client, cache=cache, max_gleanings=0)

    first = await extractor.extract(_text_unit())
    second = await extractor.extract(_text_unit())

    assert len(client.calls) == 1
    assert [entity.name for entity in first.entities] == [
        entity.name for entity in second.entities
    ]
    metadata = extractor.get_metadata()
    assert metadata["extraction_cache_enabled"] is True
    assert metadata["extraction_cache_hits"] == 1
    assert metadata["extraction_cache_misses"] == 1
    assert metadata["llm_total_calls"] == 1


@pytest.mark.asyncio
async def test_llm_extractor_persists_cache_to_directory(temp_dir):
    cache_dir = temp_dir / "cache"
    first_client = StubLLMClient([_valid_response()])
    first_extractor = LLMExtractor(
        llm_client=first_client,
        cache=ExtractionCache(cache_dir=cache_dir),
        max_gleanings=0,
    )

    await first_extractor.extract(_text_unit())

    second_client = StubLLMClient([json.dumps({"entities": [], "relationships": []})])
    second_extractor = LLMExtractor(
        llm_client=second_client,
        cache=ExtractionCache(cache_dir=cache_dir),
        max_gleanings=0,
    )
    result = await second_extractor.extract(_text_unit())

    assert len(second_client.calls) == 0
    assert [entity.name for entity in result.entities] == [
        "GraphRAG",
        "Knowledge Graph",
    ]
    assert list(cache_dir.glob("*.json"))
    assert second_extractor.get_metadata()["extraction_cache_hits"] == 1


def test_llm_extractor_rejects_invalid_rate_limit():
    client = StubLLMClient([_valid_response()])

    with pytest.raises(LLMExtractionError, match="requests_per_minute"):
        LLMExtractor(llm_client=client, requests_per_minute=0, max_gleanings=0)


@pytest.mark.asyncio
async def test_llm_extractor_rate_limits_requests(monkeypatch):
    client = StubLLMClient([_valid_response(), _valid_response()])
    extractor = LLMExtractor(
        llm_client=client,
        requests_per_minute=1,
        max_gleanings=0,
    )
    extractor._request_window_seconds = 0.01
    now = 100.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(
        "graphrag_v2.extraction.llm_extractor.asyncio.sleep",
        fake_sleep,
    )
    monkeypatch.setattr(
        "graphrag_v2.extraction.llm_extractor.time.monotonic",
        fake_monotonic,
    )

    await extractor.extract(_text_unit())
    await extractor.extract(_text_unit())

    assert len(client.calls) == 2
    assert sleeps
    assert sleeps[0] > 0
    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 2
    assert metadata["extraction_requests_per_minute"] == 1
    assert metadata["extraction_budget_exceeded"] is False


@pytest.mark.asyncio
async def test_llm_extractor_rejects_token_budget():
    client = StubLLMClient(
        [_valid_response()],
        stats={"total_tokens": 10},
    )
    extractor = LLMExtractor(
        llm_client=client,
        max_total_tokens=10,
        max_gleanings=0,
    )

    with pytest.raises(LLMExtractionError, match="token budget exceeded"):
        await extractor.extract(_text_unit())

    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 0
    assert metadata["extraction_budget_exceeded"] is True
    assert metadata["extraction_failed_chunks"] == 1
    assert metadata["extraction_failed_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_llm_extractor_rejects_cost_budget():
    client = StubLLMClient(
        [_valid_response()],
        stats={"estimated_cost": 0.25},
    )
    extractor = LLMExtractor(
        llm_client=client,
        max_estimated_cost=0.25,
        max_gleanings=0,
    )

    with pytest.raises(LLMExtractionError, match="cost budget exceeded"):
        await extractor.extract(_text_unit())

    metadata = extractor.get_metadata()
    assert metadata["llm_total_calls"] == 0
    assert metadata["extraction_budget_exceeded"] is True
    assert metadata["extraction_failed_chunks"] == 1
    assert metadata["extraction_failed_chunk_ids"] == ["chunk_1"]


def _text_unit() -> TextUnit:
    return TextUnit(
        chunk_id="chunk_1",
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        chunk_index=0,
        text="GraphRAG uses Knowledge Graph evidence.",
        n_tokens=8,
    )


def _valid_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": "GraphRAG",
                    "type": "Technology",
                    "description": "GraphRAG combines graph evidence and RAG.",
                    "confidence": 0.9,
                },
                {
                    "name": "Knowledge Graph",
                    "type": "Technology",
                    "description": "Knowledge Graph stores structured evidence.",
                    "confidence": 0.86,
                },
            ],
            "relationships": [
                {
                    "source": "GraphRAG",
                    "target": "Knowledge Graph",
                    "relation": "uses",
                    "description": "GraphRAG uses knowledge graph evidence.",
                    "confidence": 0.84,
                }
            ],
        }
    )


def _partial_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": "GraphRAG",
                    "type": "Technology",
                    "description": "GraphRAG combines graph evidence and RAG.",
                    "confidence": 0.9,
                }
            ],
            "relationships": [],
        }
    )


def _gleaning_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": "Knowledge Graph",
                    "type": "Technology",
                    "description": "Knowledge Graph stores structured evidence.",
                    "confidence": 0.86,
                }
            ],
            "relationships": [
                {
                    "source": "GraphRAG",
                    "target": "Knowledge Graph",
                    "relation": "uses",
                    "description": "GraphRAG uses knowledge graph evidence.",
                    "confidence": 0.84,
                }
            ],
        }
    )


def _assert_metadata_keys(metadata: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in metadata]
    assert missing == []
