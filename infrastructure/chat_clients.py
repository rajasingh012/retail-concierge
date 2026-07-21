"""Chat client factory — provider-agnostic LLM access.

Uses Microsoft Agent Framework's `OpenAIChatCompletionClient`, which natively
speaks the OpenAI Chat Completions wire protocol. Both vLLM (on AMD Dev Cloud
MI300X) and DeepSeek expose this endpoint, so swapping backends is just a
`base_url` change.

Provider presets:
    vllm     → AMD Developer Cloud MI300X, default base http://localhost:8000/v1
    deepseek → local-dev fallback, https://api.deepseek.com/v1

Adding a provider = one entry in PROVIDERS below.
"""
from __future__ import annotations

import os
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient


# ---------- provider registry ----------
# Each entry: (default_base_url, env_var_for_api_key)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "vllm":     ("http://localhost:8000/v1", None),         # no auth on local vLLM
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


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
        default_base_url, env_key = PROVIDERS[provider]
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
