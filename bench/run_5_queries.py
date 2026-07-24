"""Harness that runs the agent against 5 different user queries and prints,
per query, exactly what the model did: tool calls (in order), the brief
returned by extract_brief, and the final recommendations.

This is a manual/one-off probe, not a CI test. Hits the deployed model.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Project root on sys.path so 'domain', 'use_cases', 'infrastructure' resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.recommendation import extract_json_object
from infrastructure.agent_tools import build_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ABOCatalogRepository
from use_cases.shopping_agent import (
    EXTRACT_BRIEF_TOOL,
    FINALIZE_RECOMMENDATIONS_TOOL,
    CatalogEvidenceTracker,
    build_shopping_agent,
    finalized_candidates_from_response,
    structured_recommendation_from_response,
)

QUERIES = [
    "a pair of wireless earbuds under 5k rupees",
    "noise-cancelling headphones for open-plan office",
    "ergonomic office chair under 120cm tall",
    "stainless steel water bottle 1L",
    "I want a black one",
]

DB = Path("./retail_catalog.db")


async def run_query(idx: int, query: str, agent, repo) -> None:
    print(f"\n{'=' * 72}")
    print(f"Query {idx}: {query!r}")
    print("=" * 72)
    session = agent.create_session()

    try:
        response = await agent.run(query, session=session)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return

    print("\nTool-call trace:")
    for message in response.messages:
        for content in getattr(message, "contents", []):
            ctype = getattr(content, "type", "")
            if ctype == "function_call":
                name = getattr(content, "name", "?")
                args = getattr(content, "arguments", None)
                if isinstance(args, str):
                    print(f"  call   {name}({args[:300]})")
                else:
                    print(f"  call   {name}({args})")
            elif ctype == "function_result":
                result = getattr(content, "result", "")
                text = result if isinstance(result, str) else str(result)
                print(f"  result {text[:300]}{'...' if len(text) > 300 else ''}")

    import json
    print("\nBrief returned by extract_brief:")
    for message in response.messages:
        for content in getattr(message, "contents", []):
            if getattr(content, "type", "") != "function_result":
                continue
            result = getattr(content, "result", None)
            if not isinstance(result, str):
                continue
            try:
                payload = extract_json_object(result)
                parsed = json.loads(payload)
                if "intent" in parsed or "search_terms" in parsed:
                    print(f"  intent='{parsed.get('intent','')}'")
                    print(f"  search_terms='{parsed.get('search_terms','')}'")
                    print(f"  product_type='{parsed.get('product_type','')}'")
                    print(f"  brand='{parsed.get('brand','')}'")
                    print(f"  budget_usd={parsed.get('budget_usd')}  max_dimension_cm={parsed.get('max_dimension_cm')}  quantity={parsed.get('quantity')}")
                    print(f"  color='{parsed.get('color','')}'  material='{parsed.get('material','')}'")
                    print(f"  must_have={parsed.get('must_have')}")
                    print(f"  nice_to_have={parsed.get('nice_to_have')}")
                    print(f"  target_use='{parsed.get('target_use','')}'  compatibility='{parsed.get('compatibility','')}'")
                    print(f"  assumptions={parsed.get('assumptions')}")
                    print(f"  evidence_gaps={parsed.get('evidence_gaps')}")
            except Exception:
                pass

    rec = structured_recommendation_from_response(response)
    if rec is None:
        text = response.text or ""
        print(f"\nNarrative response (no structured recommendation): {text[:500]}")
        return

    print(f"\nFinal kind: {rec.kind}")
    print(f"Assumptions: {rec.assumptions}")
    print(f"Notes: {rec.notes}")
    print(f"Refinement chips: {[c.label for c in rec.refinement_chips]}")
    print(f"Recommendations:")
    for item in rec.ranked:
        print(f"  {item.rank}. {item.title_en}  | {item.brand_en}")
        for reason in item.why_it_fits:
            print(f"     + {reason}")
        for trade in item.trade_offs:
            print(f"     - {trade}")


async def main() -> None:
    if not DB.exists():
        print(f"Catalog DB not found at {DB}", file=sys.stderr)
        sys.exit(1)
    repo = ABOCatalogRepository(DB)
    provider = os.getenv("RETAIL_PROVIDER", "minimax")
    model = os.getenv("RETAIL_MODEL", "MiniMax-M3")
    print(f"Provider: {provider}  Model: {model}")
    client = build_chat_client(provider, model)
    tracker = CatalogEvidenceTracker()
    catalog_tools = build_tools(repo, catalog_tracker=tracker)
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker, provider=provider)

    for idx, query in enumerate(QUERIES, start=1):
        try:
            await run_query(idx, query, agent, repo)
        except Exception as exc:
            print(f"\n[ERROR on query {idx}] {type(exc).__name__}: {exc}")

    repo.close()


if __name__ == "__main__":
    asyncio.run(main())
