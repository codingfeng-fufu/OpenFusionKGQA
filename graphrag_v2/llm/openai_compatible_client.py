"""OpenAI-compatible chat client for local model servers."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleClient:
    """Minimal chat-completions client for vLLM and similar local servers."""

    provider_name = "openai-compatible"
    mock_mode = False
    def __init__(
        self,
        api_base: str | None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_timeout: float = 60.0,
        default_max_tokens: int | None = None,
        supports_guided_json: bool = False,
        supports_response_format_json: bool = False,
        prompt_token_cost_per_1k: float | None = None,
        completion_token_cost_per_1k: float | None = None,
        api_base_env_names: tuple[str, ...] | None = None,
        api_key_env_names: tuple[str, ...] | None = None,
    ):
        api_base_env_names = api_base_env_names or (
            "KGQA_REAL_LLM_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "LOCAL_LLM_API_BASE",
        )
        api_key_env_names = api_key_env_names or (
            "KGQA_REAL_LLM_API_KEY",
            "OPENAI_API_KEY",
            "LOCAL_LLM_API_KEY",
        )
        self.api_base = (
            api_base
            or _first_env(api_base_env_names)
        )
        self.api_key = (
            api_key
            or _first_env(api_key_env_names)
        )
        self.model = model
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.default_max_tokens = default_max_tokens
        self.supports_guided_json = supports_guided_json
        self.supports_response_format_json = supports_response_format_json
        self.prompt_token_cost_per_1k = prompt_token_cost_per_1k
        self.completion_token_cost_per_1k = completion_token_cost_per_1k

        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0
        self.total_errors = 0
        self.total_latency_seconds = 0.0
        self.max_latency_seconds = 0.0

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        guided_json: dict[str, Any] | None = None,
        response_format_json: bool = False,
    ) -> str | Any:
        """Call an OpenAI-compatible /chat/completions endpoint."""
        if stream:
            raise ValueError("OpenAICompatibleClient does not support stream=True")
        if not self.api_base:
            raise ValueError(
                "OpenAI-compatible provider requires api_base, OPENAI_BASE_URL, "
                "OPENAI_API_BASE, or LOCAL_LLM_API_BASE."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        token_limit = max_tokens if max_tokens is not None else self.default_max_tokens
        if token_limit is not None:
            payload["max_tokens"] = token_limit
        if guided_json is not None:
            payload["guided_json"] = guided_json
        elif response_format_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            started_at = time.perf_counter()
            try:
                response = self._post_json(payload)
                latency_seconds = time.perf_counter() - started_at
                self.total_calls += 1
                self._record_latency(latency_seconds)
                self._record_usage(response.get("usage"))
                return _extract_message_content(response)
            except Exception:
                self._record_latency(time.perf_counter() - started_at)
                self.total_errors += 1
                if attempt >= self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1))
        return ""

    def get_stats(self) -> dict[str, Any]:
        """Return runtime statistics in the same shape as GLMClient."""
        average_latency = (
            self.total_latency_seconds / self.total_calls
            if self.total_calls > 0
            else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_errors": self.total_errors,
            "total_latency_seconds": round(self.total_latency_seconds, 6),
            "max_latency_seconds": round(self.max_latency_seconds, 6),
            "average_latency_seconds": round(average_latency, 6),
            "estimated_cost": self._estimated_cost(),
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout,
            ) as response:
                body = _read_response_body(response, timeout=self.request_timeout)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                "OpenAI-compatible chat completion failed "
                f"(status={exc.code}, body={error_body[:500]})"
            ) from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI-compatible response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("OpenAI-compatible response JSON must be an object")
        return decoded

    def _record_usage(self, usage: Any | None) -> None:
        if usage is None:
            return
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def _record_latency(self, latency_seconds: float) -> None:
        self.total_latency_seconds += latency_seconds
        self.max_latency_seconds = max(self.max_latency_seconds, latency_seconds)

    def _estimated_cost(self) -> float | None:
        if (
            self.prompt_token_cost_per_1k is None
            or self.completion_token_cost_per_1k is None
        ):
            return None
        cost = (
            (self.prompt_tokens / 1000.0) * self.prompt_token_cost_per_1k
            + (self.completion_tokens / 1000.0) * self.completion_token_cost_per_1k
        )
        return round(cost, 8)


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI-compatible choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI-compatible choice did not include a message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible message content must be a string")
    return content


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _read_response_body(response: Any, *, timeout: float) -> str:
    """Read response body with a wall-clock timeout for chunked responses."""
    _set_response_socket_timeout(response, timeout)
    if (
        timeout <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "setitimer")
    ):
        return response.read().decode("utf-8")

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(
            "OpenAI-compatible response body read timed out "
            f"after {timeout} seconds"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return response.read().decode("utf-8")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _set_response_socket_timeout(response: Any, timeout: float) -> None:
    if timeout <= 0:
        return
    candidates = [
        ("fp", "raw", "_sock"),
        ("fp", "_sock"),
        ("raw", "_sock"),
        ("_sock",),
    ]
    for path in candidates:
        target = response
        for attr in path:
            target = getattr(target, attr, None)
            if target is None:
                break
        settimeout = getattr(target, "settimeout", None)
        if callable(settimeout):
            settimeout(timeout)
            return


def _usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
