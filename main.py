"""Composition root for the single-agent offline retail concierge."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from infrastructure.agent_tools import build_tools as _build_catalog_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ABOCatalogRepository
from use_cases import build_shopping_agent
from use_cases.shopping_agent import (
    CatalogEvidenceTracker,
    enforce_finalized_recommendation,
    finalized_candidates_from_response,
    parse_recommendation,
)

DEFAULT_PROVIDER = "minimax"
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_DB = Path("./retail_catalog.db")
MAX_REFINEMENT_CHIPS = 4


@dataclass(frozen=True)
class RefinementChip:
    label: str
    instruction: str


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


def _format_product(product: dict) -> str:
    """Render one evidence-backed recommendation for the terminal."""
    rank = product.get("rank", "-")
    title = product.get("title_en", product.get("title", "Untitled product"))
    brand = product.get("brand_en") or ""
    material = product.get("material") or ""
    color = product.get("color") or ""
    dim_note = "| Dimensions recorded" if product.get("has_dimensions", 0) else ""
    brand_text = f"| {brand}" if brand else ""
    attr_text = f"Material: {material}" if material else ""
    color_text = f"Color: {color}" if color else ""
    meta = "  ".join(filter(None, [brand_text, attr_text, color_text, dim_note]))
    lines = [f"{rank}. {title}"]
    if meta:
        lines.append(f"   {meta}")
    for reason in product.get("why_it_fits", []):
        lines.append(f"   + {reason}")
    for trade_off in product.get("trade_offs", []):
        lines.append(f"   - {trade_off}")
    if url := product.get("product_url"):
        lines.append(f"   {url}")
    return "\n".join(lines)


def _parse_refinement_chips(recommendation: dict) -> tuple[RefinementChip, ...]:
    """Validate, deduplicate, and cap user-facing refinements."""
    raw_chips = recommendation.get("refinement_chips", [])
    if not isinstance(raw_chips, list):
        return ()
    chips: list[RefinementChip] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_chips:
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        instruction = raw.get("instruction")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        key = (label.strip(), instruction.strip())
        if key in seen:
            continue
        seen.add(key)
        chips.append(RefinementChip(*key))
        if len(chips) == MAX_REFINEMENT_CHIPS:
            break
    return tuple(chips)


def _show_recommendation(recommendation: dict) -> tuple[RefinementChip, ...]:
    """Show products first, followed by assumptions and refinement chips."""
    ranked = recommendation.get("ranked", [])
    print("\nRecommendations")
    if ranked:
        print("\n\n".join(_format_product(product) for product in ranked))
    else:
        print("No supported catalog matches were found.")

    if summary := recommendation.get("recommendation"):
        print(f"\n{summary}")
    assumptions = recommendation.get("assumptions", [])
    if assumptions:
        print("\nAssumptions:")
        for assumption in assumptions:
            print(f"- {assumption}")
    notes = recommendation.get("notes", [])
    if notes:
        print("\nEvidence notes:")
        for note in notes:
            print(f"- {note}")
    if notice := recommendation.get("dataset_notice"):
        print(f"\n{notice}")

    chips = _parse_refinement_chips(recommendation)
    if chips:
        print("\nRefine:")
        print(
            " ".join(
                f"[{index}] {chip.label}" for index, chip in enumerate(chips, start=1)
            )
        )
    return chips


async def _next_message(chips: tuple[RefinementChip, ...]) -> str:
    prompt = "you> "
    if chips:
        prompt = "you> Choose a refinement number or type a message: "
    reply = (await asyncio.to_thread(input, prompt)).strip()
    if reply.isdigit() and chips:
        index = int(reply) - 1
        if 0 <= index < len(chips):
            return chips[index].instruction
    return reply


async def run_chat() -> None:
    database = Path(os.getenv("RETAIL_DB", str(DEFAULT_DB)))
    repository = ABOCatalogRepository(database)
    client = resolve_client()
    tracker = CatalogEvidenceTracker()
    catalog_tools = _build_catalog_tools(repository, catalog_tracker=tracker)
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker)
    session = agent.create_session()
    stats = repository.stats()

    print(
        f"[boot] {stats['listings']:,} products / {stats['product_types']:,} product types; "
        f"model={client.model}"
    )
    print("RetailConcierge ready (ABO catalog). Type 'quit' to exit.\n")

    chips: tuple[RefinementChip, ...] = ()
    try:
        while True:
            try:
                user_message = await _next_message(chips)
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                break
            if not user_message or user_message.lower() in {"quit", "exit"}:
                break

            tracker.reset()
            try:
                response = await agent.run(user_message, session=session)
                text = getattr(response, "text", None) or str(response)
            except Exception as exc:
                print(f"\n[error] {exc}")
                chips = ()
                continue

            try:
                parsed = parse_recommendation(text)
            except ValueError:
                # Model didn't produce JSON. Check if finalizer ran.
                finalized = finalized_candidates_from_response(response)
                if finalized is not None:
                    # Build a recommendation from the finalizer result.
                    recommendation = enforce_finalized_recommendation(
                        {
                            "kind": "recommendations",
                            "ranked": [{"item_id": c.get("item_id")} for c in finalized],
                            "assumptions": [],
                            "notes": [
                                "Model narrated instead of outputting structured JSON."
                            ],
                            "refinement_chips": [],
                        },
                        finalized,
                    )
                else:
                    print(f"\nRetailConcierge: {text.strip()}")
                    chips = ()
                    continue
            else:
                try:
                    recommendation = enforce_finalized_recommendation(
                        parsed,
                        finalized_candidates_from_response(response),
                    )
                except ValueError as exc:
                    print(f"\n[error] {exc}")
                    chips = ()
                    continue
            chips = _show_recommendation(recommendation)
    finally:
        repository.close()


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
