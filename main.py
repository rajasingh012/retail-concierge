"""Composition root for the offline three-agent retail collaboration."""
from __future__ import annotations

import asyncio
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
from use_cases.collaboration import RefinementChip, apply_refinement, run_collaboration

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


async def _request_clarification(question: str) -> str:
    print(f"\n[Clarification] {question}")
    return await asyncio.to_thread(input, "you> ")


def _format_product(product: dict) -> str:
    """Render one evidence-backed recommendation for the terminal."""
    rank = product.get("rank", "-")
    title = product.get("title", "Untitled product")
    price = product.get("dataset_price", 0)
    stars = product.get("dataset_stars", 0)
    price_text = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    lines = [
        f"{rank}. {title}",
        f"   Dataset price: {price_text} | Stars: {stars}",
    ]
    for reason in product.get("why_it_fits", []):
        lines.append(f"   + {reason}")
    for trade_off in product.get("trade_offs", []):
        lines.append(f"   - {trade_off}")
    if url := product.get("product_url"):
        lines.append(f"   {url}")
    return "\n".join(lines)


def _show_result(result) -> None:
    """Show products first, followed by assumptions and refinement chips."""
    recommendation = result.recommendation
    ranked = recommendation.get("ranked", [])
    print("\nRecommendations")
    if ranked:
        print("\n\n".join(_format_product(product) for product in ranked))
    else:
        print("No supported catalog matches were found.")

    if summary := recommendation.get("recommendation"):
        print(f"\n{summary}")
    assumptions = result.brief.get("assumptions", [])
    if assumptions:
        print("\nAssumptions:")
        for assumption in assumptions:
            print(f"- {assumption}")
    notes = recommendation.get("critic_notes", [])
    if notes:
        print("\nEvidence notes:")
        for note in notes:
            print(f"- {note}")
    if notice := recommendation.get("dataset_notice"):
        print(f"\n{notice}")
    if result.refinement_chips:
        print("\nRefine:")
        print(" ".join(
            f"[{index}] {chip.label}"
            for index, chip in enumerate(result.refinement_chips, start=1)
        ))


async def _next_refinement(
    current_request: str, chips: tuple[RefinementChip, ...]
) -> str | None:
    """Translate a numbered chip or free text into the next shopping request."""
    if not chips:
        return None
    reply = (
        await asyncio.to_thread(
            input, "refine> Choose a number, type your own refinement, or Enter: "
        )
    ).strip()
    if not reply:
        return None
    if reply.isdigit():
        index = int(reply) - 1
        if 0 <= index < len(chips):
            return apply_refinement(current_request, chips[index])
        print(f"Choose a number from 1 to {len(chips)}.")
        return None
    custom = RefinementChip(label=reply, instruction=reply)
    return apply_refinement(current_request, custom)


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

            current_request = user_message
            while current_request:
                try:
                    result = await run_collaboration(
                        discovery,
                        research,
                        critic,
                        current_request,
                        _request_clarification,
                    )
                except Exception as exc:
                    print(f"\n[error] {exc}")
                    break
                _show_result(result)
                current_request = await _next_refinement(
                    current_request, result.refinement_chips
                )
    finally:
        repository.close()


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
