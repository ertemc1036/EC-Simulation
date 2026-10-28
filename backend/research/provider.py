"""OpenAI-compatible chat transport for research runs.

Tests install a provider with ``set_research_provider``. The HTTP function is the
only network path and is not used by offline demo fixtures.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from backend.config import settings

http_attempts = 0


class CredentialsMissing(Exception):
    """Key or mode gate blocked the call before any socket."""


class ProviderCallError(Exception):
    """The transport was attempted and failed. The message never includes a key."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    usage_estimated: bool = False


class ResearchProvider(Protocol):
    name: str

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        model: str,
    ) -> Completion: ...


_installed: ResearchProvider | None = None


def set_research_provider(provider: ResearchProvider | None) -> None:
    global _installed
    _installed = provider


def get_research_provider() -> ResearchProvider:
    if _installed is not None:
        return _installed
    return OpenAICompatibleProvider()


def credential_block_reason() -> str | None:
    """None when a call is allowed to proceed to a provider object.

    An installed test provider counts as ready so suites never need a real key.
    """
    if _installed is not None:
        return None
    if not settings.openai_api_key.strip():
        return "OPENAI_API_KEY is missing. No provider call was made."
    if not settings.allow_llm_mode:
        return "ALLOW_LLM_MODE is false. No provider call was made."
    return None


def reset_http_attempts() -> None:
    global http_attempts
    http_attempts = 0


def post_chat_completions(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST a chat completion. Increments ``http_attempts`` before the socket."""
    global http_attempts
    http_attempts += 1
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderCallError("Provider HTTP request failed.") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderCallError("Provider HTTP response was not JSON.") from exc
    if not isinstance(data, dict):
        raise ProviderCallError("Provider HTTP response was not an object.")
    return data


def _message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ProviderCallError("Provider response had no choices.")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise ProviderCallError("Provider response message was empty.")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        text = "\n".join(parts).strip()
        if text:
            return text
    raise ProviderCallError("Provider response content was empty.")


class OpenAICompatibleProvider:
    """Chat completions against the configured OpenAI-compatible base URL."""

    name = "openai_compatible_http"

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        model: str,
    ) -> Completion:
        if not settings.openai_api_key.strip() or not settings.allow_llm_mode:
            raise CredentialsMissing("Research provider credentials are not available.")
        url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        data = post_chat_completions(
            url=url,
            api_key=settings.openai_api_key,
            payload={
                "model": model or settings.openai_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=float(settings.request_timeout_seconds),
        )
        text = _message_text(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        estimated = "prompt_tokens" not in usage
        if estimated:
            prompt_tokens = _estimate_messages(messages)
            completion_tokens = max(1, (len(text) + 3) // 4)
        else:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=str(data.get("model") or model or settings.openai_model),
            usage_estimated=estimated,
        )


def _estimate_messages(messages: list[dict[str, str]]) -> int:
    chars = sum(len(message.get("content") or "") for message in messages)
    return max(1, (chars + 3) // 4)
