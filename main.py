"""Composition root for the offline three-agent retail collaboration."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from infrastructure.agent_tools import build_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ProductCatalogRepository
from use_cases import (
    build_critic_agent,
    build_discovery_agent,
    build_research_agent,
)
from use_cases.collaboration import run_collaboration

DEFAULT_PROVIDER = "vllm"
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_DB = Path("./retail_catalog.db")


def resolve_client():
    """Build the MAF OpenAI-compatible client from environment variables."""
    provider = os.getenv("RETAIL_PROVIDER", DEFAULT_PROVIDER)
    model = os.getenv("RETAIL_MODEL", DEFAULT_MODEL)
    overrides: dict[str, str] = {}
    if base_url := os.getenv("RETAIL_BASE_URL"):
        overrides["base_url"] = base_url
    if api_key := os.getenv("RETAIL_API_KEY"):
        overrides["api_key"] = api_key
    return build_chat_client(provider, model, **overrides)


async def _ask_user(question: str) -> str:
    print(f"\n[Discovery] {question}")
    return await asyncio.to_thread(input, "you> ")


async def run_chat() -> None:
    database = Path(os.getenv("RETAIL_DB", str(DEFAULT_DB)))
    repository = ProductCatalogRepository(database)
    client = resolve_client()
    tools = build_tools(repository)
    discovery = build_discovery_agent(client)
    research = build_research_agent(client, tools)
    critic = build_critic_agent(client)
    stats = repository.stats()

    print(
        f"[boot] {stats['products']:,} products / {stats['categories']} categories; "
        f"model={client.model}"
    )
    print("RetailConcierge ready. Type 'quit' to exit.\n")

    try:
        while True:
            try:
                user_message = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                break
            if not user_message or user_message.lower() in {"quit", "exit"}:
                break

            try:
                result = await run_collaboration(
                    discovery, research, critic, user_message, _ask_user
                )
            except Exception as exc:
                print(f"\n[error] {exc}")
                continue

            print("\n[Discovery brief]")
            print(json.dumps(result.brief, indent=2, ensure_ascii=False))
            print("\n[Research evidence]")
            print(json.dumps(result.research, indent=2, ensure_ascii=False))
            print("\n[Critic recommendation]")
            print(json.dumps(result.recommendation, indent=2, ensure_ascii=False))
    finally:
        repository.close()


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
