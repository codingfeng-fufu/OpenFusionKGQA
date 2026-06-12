"""Extraction configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID


class ExtractionConfig(BaseModel):
    """Configuration for candidate knowledge extraction."""

    extractor_provider: str = Field(
        default="mock",
        description="Default extractor provider.",
    )
    llm_model_id: str = Field(
        default=DEFAULT_CHAT_MODEL_ID,
        description="Model ID used by LLM extraction.",
    )
    llm_provider: str = Field(
        default="deepseek",
        description="LLM provider used by LLM extraction.",
    )
    max_retries: int = Field(
        default=2,
        description="Maximum JSON extraction/repair attempts per text unit.",
    )
    max_gleanings: int = Field(
        default=1,
        description="Default supplemental LLM gleaning rounds per text unit.",
    )
    fail_on_invalid_chunk: bool = Field(
        default=True,
        description="Fail indexing when a text unit cannot be extracted.",
    )
    default_confidence: float = Field(
        default=0.7,
        description="Default confidence for LLM fields without confidence values.",
    )
    requests_per_minute: int | None = Field(
        default=None,
        description="Optional request-per-minute rate limit for LLM extraction.",
    )
    concurrent_requests: int = Field(
        default=1,
        description="Maximum concurrent LLM extraction requests.",
    )
    max_prompt_tokens_per_chunk: int | None = Field(
        default=None,
        description="Optional approximate prompt token limit per text unit.",
    )
    max_total_tokens: int | None = Field(
        default=None,
        description="Optional total token budget for one extraction run.",
    )
    max_estimated_cost: float | None = Field(
        default=None,
        description="Optional estimated cost budget for one extraction run.",
    )
    salvage_on_parse_failure: bool = Field(
        default=True,
        description="Try to salvage valid records when JSON repair fails.",
    )
    cache_enabled: bool = Field(
        default=False,
        description="Enable LLM extraction response caching.",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Optional directory for persistent LLM extraction cache files.",
    )

    @model_validator(mode="after")
    def _validate_model(self):
        provider = self.extractor_provider.strip().lower()
        if provider not in {"mock", "llm"}:
            raise ValueError("extractor_provider must be one of: mock, llm")
        self.extractor_provider = provider
        if not self.llm_model_id.strip():
            raise ValueError("llm_model_id cannot be empty")
        self.llm_model_id = self.llm_model_id.strip()
        llm_provider = self.llm_provider.strip().lower()
        if llm_provider not in {
            "glm",
            "deepseek",
            "openai-compatible",
            "openai",
            "vllm",
            "local",
        }:
            raise ValueError(
                "llm_provider must be one of: glm, deepseek, "
                "openai-compatible, openai, vllm, local"
            )
        if llm_provider == "openai":
            llm_provider = "openai-compatible"
        self.llm_provider = llm_provider
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.max_gleanings < 0:
            raise ValueError("max_gleanings must be greater than or equal to 0")
        if not 0.0 <= self.default_confidence <= 1.0:
            raise ValueError("default_confidence must be between 0 and 1")
        if self.requests_per_minute is not None and self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        if self.concurrent_requests < 1:
            raise ValueError("concurrent_requests must be at least 1")
        if self.max_prompt_tokens_per_chunk is not None and self.max_prompt_tokens_per_chunk < 1:
            raise ValueError("max_prompt_tokens_per_chunk must be at least 1")
        if self.max_total_tokens is not None and self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be at least 1")
        if self.max_estimated_cost is not None and self.max_estimated_cost < 0:
            raise ValueError("max_estimated_cost must be greater than or equal to 0")
        if self.cache_dir is not None:
            cache_dir = self.cache_dir.strip()
            if not cache_dir:
                raise ValueError("cache_dir cannot be empty")
            self.cache_dir = cache_dir
        return self
