"""LLM client abstractions.

Provider-agnostic surface that all chat clients must implement.
Agents depend on `ChatModelClient` (a Protocol), not on any
specific SDK class — so swapping vLLM / DeepSeek / etc.
is a one-line change in the composition root.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    """Minimal message envelope used across providers."""
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ChatTurn:
    """One exchange: system instruction + conversation history + new user input."""
    system: str
    history: List[ChatMessage]
    user_input: str


@runtime_checkable
class ChatModelClient(Protocol):
    """The interface every concrete LLM client must satisfy.

    C# analogy:
        public interface IChatModelClient {
            string Complete(ChatTurn turn);
        }
    Pythonic twist: structural typing — any class with a matching
    `complete(turn) -> str` method is implicitly an implementation.
    No `implements` keyword, no DI container.
    """

    model: str

    def complete(self, turn: ChatTurn) -> str: ...


# ---------- concrete implementations ----------

class OpenAICompatProvider:
    """Any provider that speaks the OpenAI Chat Completions wire protocol.

    Covers DeepSeek, OpenRouter, Together, Groq, etc.
    Provider-specific extensions (e.g. DeepSeek's `response_format`)
    are passed through `extra_body` so they survive the standard SDK.
    """

    def __init__(self, model: str, base_url: str, api_key: str,
                 env_key: str | None = None,
                 extra_body: dict | None = None) -> None:
        self.model = model
        self._api_key = api_key or (env_key and _require_env(env_key))  # type: ignore
        self._base_url = base_url
        self._extra_body = extra_body or {}
        from openai import OpenAI
        self._sdk = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def complete(self, turn: ChatTurn) -> str:
        messages = [{"role": "system", "content": turn.system}]
        messages.extend({"role": m.role, "content": m.content} for m in turn.history)
        messages.append({"role": "user", "content": turn.user_input})
        kwargs = {"model": self.model, "messages": messages}
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        resp = self._sdk.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


# Backwards-compatible aliases so main.py + tests keep working.
class DeepSeekClient(OpenAICompatProvider):
    """DeepSeek API (OpenAI-compatible at api.deepseek.com).

    Use as the local-dev fallback when you don't want to spin up an AMD
    droplet — same OpenAI endpoint shape as vLLM, agent code doesn't change.
    """
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com/v1", **kw) -> None:
        super().__init__(model=model, base_url=base_url,
                         api_key=api_key or "", env_key="DEEPSEEK_API_KEY", **kw)


class VLLMClient(OpenAICompatProvider):
    """vLLM's OpenAI-compatible HTTP server.

    Default base URL: http://localhost:8000/v1
    Primary backend — runs on the AMD Developer Cloud MI300X droplet.

    Exposes prefix-cache metrics on /v1/metrics that are the headline
    AMD rubric number (vllm:prefix_cache_hit_rate goes 0.0 -> ~0.95+).
    The OpenAI SDK call is identical to DeepSeek; only base_url changes.
    """
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "http://localhost:8000/v1", **kw) -> None:
        super().__init__(model=model, base_url=base_url,
                         api_key=api_key or "EMPTY", env_key=None, **kw)


# ---------- registry / helpers ----------

def _require_env(name: str) -> str:
    import os
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


# Provider registry — name -> factory(model_name, **overrides) -> ChatModelClient
#   vllm      -> AMD Developer Cloud MI300X (pre-installed; primary judge demo)
#   deepseek  -> local-dev fallback so you can iterate without spinning up a droplet
# Adding a provider = one new class + one line here.
PROVIDERS: dict[str, type] = {
    "vllm": VLLMClient,
    "deepseek": DeepSeekClient,
}


def build_chat_client(provider: str, model: str, **kwargs) -> ChatModelClient:
    """Factory. `provider` is a key in PROVIDERS."""
    try:
        cls = PROVIDERS[provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available: {sorted(PROVIDERS)}"
        ) from e
    return cls(model=model, **kwargs)