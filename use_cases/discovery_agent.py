"""Discovery agent — extracts a structured brief from the user.

Takes any `OpenAIChatClient` from `agent_framework.openai`, so both
vLLM (AMD Dev Cloud MI300X) and DeepSeek (local dev) work without
code changes here.
"""
from __future__ import annotations

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


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


def build_discovery_agent(client: OpenAIChatClient) -> Agent:
    """C# analogy: constructor injection of IChatClient."""
    return Agent(
        client=client,
        instructions=DISCOVERY_INSTRUCTIONS,
    )