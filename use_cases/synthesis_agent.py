"""Synthesis agent — runs the structured brief against live tools.

Provider-agnostic: takes any `OpenAIChatClient` from `agent_framework.openai`.
"""
from __future__ import annotations

from typing import Any, List

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


SYNTHESIS_INSTRUCTIONS = """\
You are a retail synthesis assistant. You receive a structured
shopping brief in JSON and must produce a final ranked recommendation
using the tools available.

Workflow:
1. Call search_catalog once with the most discriminating tokens
   (brand + must-have feature) to find local candidates.
2. For the top 1-3 candidates, optionally call fetch_product_from_site
   on a live URL to verify price/availability.
3. Rank by how well each product matches `must_have`, then by price.
4. Return a final answer as JSON with:
   - ranked: list of {title, source_url, base_price, why_it_fits}
   - trade_offs: list of short strings noting compromises
   - next_step: a single concrete suggestion (e.g. which to buy first)

Never invent product data — only use what the tools returned.
"""


def build_synthesis_agent(
    client: OpenAIChatClient,
    tools: List[Any],
) -> Agent:
    return Agent(
        client=client,
        instructions=SYNTHESIS_INSTRUCTIONS,
        tools=tools,
    )