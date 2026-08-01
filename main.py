"""Composition root for the single-agent offline retail concierge."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agent_framework import AgentResponse

from domain.recommendation import (
    MAX_REFINEMENT_CHIPS,
    RefinementChip,
)
from infrastructure.agent_tools import build_tools as _build_catalog_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ABOCatalogRepository
from use_cases import build_shopping_agent
from use_cases.shopping_agent import (
    CatalogEvidenceTracker,
    enforce_finalized_recommendation,
    finalized_candidates_from_response,
    structured_recommendation_from_response,
)

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_DB = Path("./retail_catalog.db")
DEFAULT_AUDIT_LOG = Path("./retail_audit.jsonl")


def _load_catalog_vocabulary(repository: ABOCatalogRepository) -> dict[str, list[str]]:
    """Pull the canonical product_type / brand lists for the brief prompt.

    Bounded by ``min_listings`` to skip singleton product_types from the
    import (test rows, partial imports). Brand list capped at 200 to keep
    the prompt section under ~2k tokens; the brief validator still gates
    every value so off-vocabulary mapping attempts surface as Pydantic
    rejections.
    """
    return {
        "product_types": [
            str(row["product_type"])
            for row in repository.list_product_types(min_listings=5)
        ],
        "brands": [
            str(row["brand"])
            for row in repository.list_brands(limit=200, min_listings=1)
        ],
    }


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


def _show_recommendation(recommendation) -> tuple[RefinementChip, ...]:
    """Show products first, followed by assumptions and refinement chips."""
    ranked = recommendation.ranked
    print("\nRecommendations")
    if ranked:
        for item in ranked:
            print(_format_product(item.model_dump()))
    else:
        print("No supported catalog matches were found.")

    if recommendation.recommendation:
        print(f"\n{recommendation.recommendation}")
    if recommendation.assumptions:
        print("\nAssumptions:")
        for assumption in recommendation.assumptions:
            print(f"- {assumption}")
    if recommendation.notes:
        print("\nEvidence notes:")
        for note in recommendation.notes:
            print(f"- {note}")
    if recommendation.dataset_notice:
        print(f"\n{recommendation.dataset_notice}")

    chips = recommendation.refinement_chips[:MAX_REFINEMENT_CHIPS]
    if chips:
        print("\nRefine:")
        print(
            " ".join(
                f"[{index}] {chip.label}" for index, chip in enumerate(chips, start=1)
            )
        )
    return tuple(chips)


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


def _structured_recommendation(response: AgentResponse):
    """Return the typed RecommendationResponse from the MAF response.

    MAF exposes it as ``response.value`` when ``default_options.response_format``
    is set and the provider either enforced the schema or MAF's fallback parser
    succeeded. Returns None for clarification turns or any provider that
    delivered only narrative text.
    """
    return structured_recommendation_from_response(response)


async def run_chat() -> None:
    database = Path(os.getenv("RETAIL_DB", str(DEFAULT_DB)))
    repository = ABOCatalogRepository(database)
    provider = os.getenv("RETAIL_PROVIDER", DEFAULT_PROVIDER)
    client = resolve_client()
    tracker = CatalogEvidenceTracker()
    audit_logger = None
    audit_path = os.getenv("RETAIL_AUDIT_LOG")
    if audit_path:
        from infrastructure.audit import AuditLogger
        audit_logger = AuditLogger(Path(audit_path))
    catalog_vocabulary = _load_catalog_vocabulary(repository)
    catalog_tools = _build_catalog_tools(
        repository, catalog_tracker=tracker, audit_logger=audit_logger
    )
    agent = build_shopping_agent(
        client,
        catalog_tools,
        tracker=tracker,
        provider=provider,
        audit_logger=audit_logger,
        catalog_vocabulary=catalog_vocabulary,
    )
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
            except Exception as exc:
                print(f"\n[error] {exc}")
                chips = ()
                continue

            recommendation = _structured_recommendation(response)
            if recommendation is None:
                # Clarification turn or unparseable narrative.
                finalized = finalized_candidates_from_response(response)
                if finalized is not None:
                    recommendation = enforce_finalized_recommendation(
                        {
                            "kind": "recommendations",
                            "ranked": [{"item_id": c.item_id} for c in finalized],
                            "assumptions": [],
                            "notes": [
                                "Model narrated instead of outputting structured JSON."
                            ],
                            "refinement_chips": [],
                        },
                        finalized,
                    )
                else:
                    text = response.text or ""
                    if text.strip():
                        print(f"\nRetailConcierge: {text.strip()}")
                    chips = ()
                    continue

            try:
                recommendation = enforce_finalized_recommendation(
                    recommendation,
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