"""Chat client factory — provider-agnostic LLM access.

Uses Microsoft Agent Framework's `OpenAIChatCompletionClient`, which natively
speaks the OpenAI Chat Completions wire protocol. All providers (vLLM, DeepSeek)
expose this endpoint, so swapping backends is just a `base_url` change.

Provider presets:
    vllm     → AMD Developer Cloud MI300X, default base http://localhost:8000/v1
    deepseek → cloud API, https://api.deepseek.com/v1

Provider-specific request fields that aren't part of the OpenAI Chat
Completions typed signature (e.g. DeepSeek's ``thinking``) are passed via
the OpenAI SDK's documented ``extra_body`` kwarg. MAF forwards
``default_options`` through to ``chat.completions.create(**kwargs)``,
so wrapping the fields in ``extra_body`` is the MAF-recommended way —
no SDK patching required (verified against MAF 1.14.0).

Adding a provider = one entry in PROVIDERS below.
"""
from __future__ import annotations

import os
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient


# ---------- provider registry ----------
# Each entry: (default_base_url, env_var_for_api_key, per-request default_options).
# Merge these into the Agent's `default_options` to attach provider-specific
# fields. Fields the OpenAI SDK doesn't know (``thinking``, custom sampling
# params, etc.) must live inside ``extra_body`` — that's the SDK's typed
# escape hatch for unrecognized JSON body fields.
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
    """Return the per-provider extras dict for merging into ``default_options``.

    Read by callers (e.g. ``use_cases.shopping_agent``) so the agent's
    ``default_options`` can include provider-specific fields without
    hard-coding provider names. Wrap any non-OpenAI-standard fields in
    ``extra_body`` so the OpenAI SDK passes them through to the request body.
    """
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