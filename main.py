"""Composition root — wire dependencies and run the multi-turn loop.

Provider selection is env-driven:
    RETAIL_PROVIDER=vllm     (default for cloud demo)   → VLLMClient
    RETAIL_PROVIDER=deepseek (local-dev fallback)       → DeepSeekClient
    RETAIL_MODEL=...         model name (provider-specific)
    RETAIL_BASE_URL=...      optional override

Run from the project root:
    cd /home/rajasingh/retail_concierge
    PYTHONPATH=. python main.py
"""
from __future__ import annotations

import asyncio
import os

from infrastructure.agent_tools import build_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ProductCatalogRepository
from infrastructure.indexer import LocalHybridSearchEngine
from infrastructure.scraper import PlaywrightScraper
from use_cases import build_discovery_agent, build_synthesis_agent

DEFAULT_PROVIDER = "vllm"      # cloud MI300X default; use "deepseek" for local dev
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_DB = "./retail_catalog.db"


def resolve_client():
    """Build the chat client from env vars. Pure factory, no global state."""
    provider = os.getenv("RETAIL_PROVIDER", DEFAULT_PROVIDER)
    model = os.getenv("RETAIL_MODEL", DEFAULT_MODEL)
    overrides: dict = {}
    if base_url := os.getenv("RETAIL_BASE_URL"):
        overrides["base_url"] = base_url
    if api_key := os.getenv("RETAIL_API_KEY"):
        overrides["api_key"] = api_key
    return build_chat_client(provider, model, **overrides)


async def _run_streaming(agent, prompt: str, label: str) -> str:
    """Stream tokens if the MAF agent supports it; otherwise fall back."""
    print(f"\n[{label}] ", end="", flush=True)
    # Try streaming first — judge demo wins if we get incremental tokens.
    try:
        chunks: list[str] = []
        async for chunk in agent.run(prompt, stream=True):
            text = getattr(chunk, "text", None) or str(chunk)
            chunks.append(text)
            print(text, end="", flush=True)
        print()
        return "".join(chunks)
    except TypeError:
        # MAF version doesn't accept stream=; do a blocking call.
        resp = await agent.run(prompt)
        text = getattr(resp, "text", None) or str(resp)
        print(text)
        return text


async def run_chat() -> None:
    # ---------- infrastructure ----------
    repo = ProductCatalogRepository(DEFAULT_DB)
    search_engine = LocalHybridSearchEngine(repo)
    scraper = PlaywrightScraper()
    await scraper.start()
    tools = build_tools(search_engine, scraper)

    # ---------- model client (provider-agnostic) ----------
    client = resolve_client()
    print(f"[boot] provider={type(client).__name__} model={client.model}\n")

    # ---------- agents ----------
    discovery = build_discovery_agent(client)
    synthesis = build_synthesis_agent(client, tools)

    # ---------- multi-turn loop ----------
    print("RetailConcierge ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user_msg or user_msg.lower() in {"quit", "exit"}:
            break

        brief_text = await _run_streaming(discovery, user_msg, "brief")

        synthesis_prompt = (
            "Here is the structured shopping brief as JSON:\n"
            f"{brief_text}\n\n"
            "Produce the final ranked recommendation."
        )
        await _run_streaming(synthesis, synthesis_prompt, "synthesis")

    # ---------- teardown ----------
    await scraper.close()
    repo.close()


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()