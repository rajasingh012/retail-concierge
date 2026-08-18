"""Chat client factory — provider-agnostic LLM access.

Uses Microsoft Agent Framework's `OpenAIChatCompletionClient`, which natively
speaks the OpenAI Chat Completions wire protocol. All providers (vLLM, DeepSeek)
expose this endpoint, so swapping backends is just a `base_url` change.

Provider presets:
    vllm     → AMD Developer Cloud MI300X, default base http://localhost:8000/v1
    deepseek → cloud API, https://api.deepseek.com/v1

PROVIDERS[provider] = (base_url, env_key, extras_dict). The agent builder
merges `extras_dict` into `Agent(default_options=...)` blindly; the agent
never names a provider, so adding a new provider = one tuple.

`extra_body` is the OpenAI SDK's typed escape hatch for fields outside the
Chat Completions spec (DeepSeek `thinking`, etc.). MAF 1.14.0+ forwards it
through to `chat.completions.create(**kwargs)` unchanged — no app-side
routing needed.

Adding a provider = one entry in PROVIDERS below.
"""
from __future__ import annotations

import os
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient


# Each entry: (default_base_url, env_var_for_api_key, per-request
# extras_dict). The agent builder merges extras_dict into `default_options`.
PROVIDERS: dict[str, tuple[str, str | None, dict[str, Any]]] = {
    "vllm": ("http://localhost:8000/v1", None, {}),
    "deepseek": (
        "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY",
        {
            "extra_body": {
                # Disable thinking for contract turns. DeepSeek defaults to
                # thinking enabled with effort "high". When thinking is on
                # and the agent performs tool calls, ``reasoning_content``
                # must be forwarded in every subsequent turn or the API
                # returns 400. Disabling thinking avoids that MAF-
                # compatibility issue entirely.
                "thinking": {"type": "disabled"},
                "max_completion_tokens": 8192,
            },
        },
    ),
}


def provider_extras(provider: str) -> dict[str, Any]:
    """Return ``PROVIDERS[provider][2]`` for merging into ``default_options``."""
    try:
        _, _, extras = PROVIDERS[provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available: {sorted(PROVIDERS)}"
        ) from e
    return dict(extras)


def build_chat_client(provider: str, model: str, **overrides: Any) -> OpenAIChatCompletionClient:
    """Build an `OpenAIChatCompletionClient` from a provider name and model.

    Args:
        provider: Key in PROVIDERS (e.g. "vllm", "deepseek").
        model: Model name (provider-specific, e.g. "google/gemma-3-27b-it"
               for vLLM, "deepseek-chat" for DeepSeek).
        **overrides: Forwarded to `OpenAIChatCompletionClient.__init__`. Recognized
            keys: base_url, api_key.
    """
    try:
        default_base_url, env_key, _extras = PROVIDERS[provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available: {sorted(PROVIDERS)}"
        ) from e

    base_url = overrides.pop("base_url", os.getenv("RETAIL_BASE_URL") or default_base_url)

    api_key = overrides.pop("api_key", None)
    if not api_key and env_key:
        api_key = os.environ.get(env_key)
    api_key = api_key or "EMPTY"  # vLLM doesn't require auth; OpenAI SDK still wants a non-empty string

    return OpenAIChatCompletionClient(model=model, api_key=api_key, base_url=base_url, **overrides)