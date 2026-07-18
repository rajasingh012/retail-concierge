"""Discovery agent — extracts a structured brief from the user.

Depends on `ChatModelClient` (a Protocol), so any provider in
infrastructure/chat_clients.PROVIDERS works without code changes here.
"""
from __future__ import annotations

from agent_framework import ChatAgent

from infrastructure.chat_clients import ChatModelClient, ChatTurn


DISCOVERY_INSTRUCTIONS = """\
You are a retail discovery assistant. Your only job is to extract a
structured shopping brief from the user's free-form message.

Return JSON with these fields:
- intent: one short sentence describing what the user wants
- brands: list of brand names the user mentioned or implied (exact spelling)
- budget_max: optional number, the user's ceiling in USD, or null
- must_have: list of required features (e.g. "ergonomic", "mesh back")
- nice_to_have: list of optional features
- target_use: brief context (office, gaming, kitchen, ...)

Ask a single clarifying question if a field is ambiguous. Otherwise
return the JSON only, no prose.
"""


def build_discovery_agent(client: ChatModelClient) -> ChatAgent:
    """C# analogy: constructor injection of IChatModelClient."""
    return ChatAgent(
        chat_client=client,                # Protocol-typed; any provider works
        model=getattr(client, "model", ""),
        instructions=DISCOVERY_INSTRUCTIONS,
    )