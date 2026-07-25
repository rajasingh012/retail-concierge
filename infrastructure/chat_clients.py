"""Chat client factory — provider-agnostic LLM access.

Uses Microsoft Agent Framework's `OpenAIChatCompletionClient`, which natively
speaks the OpenAI Chat Completions wire protocol. All providers (vLLM, DeepSeek,
MiniMax) expose this endpoint, so swapping backends is just a `base_url` change.

Provider presets:
    vllm     → AMD Developer Cloud MI300X, default base http://localhost:8000/v1
    deepseek → cloud API, https://api.deepseek.com/v1
    minimax  → cloud API, https://api.minimax.io/v1

Adding a provider = one entry in PROVIDERS below.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient
from openai.resources.chat.completions import AsyncCompletions

_OPENAI_COMPLETIONS_CREATE_KWARGS: frozenset[str] = frozenset(
    inspect.signature(AsyncCompletions.create).parameters
)


# ---------- provider registry ----------
# Each entry: (default_base_url, env_var_for_api_key, per-request extras).
# `extras` are merged into every chat-completion request via the OpenAI SDK's
# `extra_body` — a documented escape hatch for provider-specific fields that
# don't exist on the OpenAI type stubs (e.g. ``reasoning_split`` for MiniMax,
# custom sampling params on local vLLM builds). Unknown fields on the server
# side are silently ignored, so a MiniMax-only extra on vLLM is a no-op.
PROVIDERS: dict[str, tuple[str, str | None, dict[str, Any]]] = {
    "vllm":     ("http://localhost:8000/v1", None, {}),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", {
        # Disable thinking for contract turns. DeepSeek defaults to thinking
        # enabled with effort "high". When thinking is enabled and the agent
        # performs tool calls, `reasoning_content` must be forwarded in every
        # subsequent turn or the API returns 400. Disabling thinking avoids
        # this MAF-compatibility issue entirely.
        "thinking": {"type": "disabled"},
        "max_completion_tokens": 8192,
    }),
    # MiniMax-M3 puts `<think>...</think>` blocks inside `content` by default.
    # `reasoning_split=True` routes them into a separate `reasoning_details`
    # field, leaving the agent's JSON contract unwrapped in `content`.
    # `max_completion_tokens` is the docs-recommended length cap; default is
    # conservative and the agent's reasoning plus JSON contract needs more.
    "minimax":  (
        "https://api.minimax.io/v1",
        "MINIMAX_API_KEY",
        {"reasoning_split": True, "max_completion_tokens": 8192},
    ),
}


def _install_extra_body_routing() -> None:
    """Route unknown ``default_options`` keys through the OpenAI SDK's
    ``extra_body`` instead of passing them as typed kwargs.

    MAF forwards every key in ``default_options`` straight to OpenAI SDK's
    ``chat.completions.create(**kwargs)``. Provider-specific fields
    (``reasoning_split``, ``thinking``) raise ``TypeError`` on the OpenAI SDK
    because they aren't part of its typed signature. The OpenAI SDK exposes
    ``extra_body`` for this case; we collect unrecognised keys and let the
    SDK merge them into the request JSON. The SDK itself decides what counts
    as a known kwarg — derived once from its current signature.
    """
    original = OpenAIChatCompletionClient._prepare_options

    def wrapped(self, messages, options):
        prepared = original(self, messages, options)
        existing_extra = prepared.pop("extra_body", None) or {}
        for key in list(prepared):
            if key not in _OPENAI_COMPLETIONS_CREATE_KWARGS and key != "messages":
                existing_extra[key] = prepared.pop(key)
        if existing_extra:
            prepared["extra_body"] = existing_extra
        return prepared

    OpenAIChatCompletionClient._prepare_options = wrapped


_install_extra_body_routing()


def provider_extras(provider: str) -> dict[str, Any]:
    """Return the per-provider extras dict for embedding into request bodies.

    Read by callers (e.g. ``use_cases.shopping_agent``) so the agent's
    ``default_options`` can include provider-specific fields without
    hard-coding MiniMax names. Unknown extras on a different provider are
    forwarded and silently ignored by the server.
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