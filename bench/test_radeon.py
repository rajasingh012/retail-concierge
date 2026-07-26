#!/usr/bin/env python3
"""Test 5 catalog-aligned queries against the Radeon endpoint with detailed logging."""
import asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain.recommendation import extract_json_object
from infrastructure.agent_tools import build_tools, cache_stats, clear_cache
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ABOCatalogRepository
from use_cases.shopping_agent import (
    CatalogEvidenceTracker, build_shopping_agent,
    finalized_candidates_from_response, structured_recommendation_from_response,
)

DB = Path("./retail_catalog.db")
QUERIES = [
    "Find me an office chair under $200",
    "I need a wireless mouse for my laptop",
    "show me a coffee maker for my kitchen",
    "Find noise cancelling headphones",
    "I need a monitor for work under 27 inches",
]

async def run_one(idx, query, agent, repo):
    print(f"\n{'='*72}")
    print(f"Query {idx}: {query!r}")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    t0 = time.perf_counter()

    session = agent.create_session()
    tracker = CatalogEvidenceTracker()
    clear_cache()

    try:
        response = await agent.run(query, session=session)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return

    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.2f}s")

    # Print full tool-call trace
    print("\n--- Tool calls ---")
    for msg in response.messages:
        for content in getattr(msg, "contents", []):
            ctype = getattr(content, "type", "")
            if ctype == "function_call":
                name = getattr(content, "name", "?")
                args = getattr(content, "arguments", None)
                if isinstance(args, str):
                    print(f"  CALL {name}({args[:300]})")
                else:
                    print(f"  CALL {name}({json.dumps(args, ensure_ascii=False)[:300]})")
            elif ctype == "function_result":
                result = getattr(content, "result", "")
                if isinstance(result, str):
                    print(f"  → result: {result[:300]}...")
                else:
                    print(f"  → result: {json.dumps(result, ensure_ascii=False)[:300]}")

    # Parse structured output
    rec = structured_recommendation_from_response(response)
    if rec is None:
        text = response.text or ""
        print(f"\nNo structured recommendation. Text: {text[:500]}")
        return

    print(f"\n--- Response ---")
    print(f"Kind: {rec.kind}")
    print(f"Recommendations: {len(rec.ranked)}")
    for item in rec.ranked:
        print(f"  #{item.rank}: {item.title_en[:60]} | {item.brand_en}")
        for r in item.why_it_fits:
            print(f"    + {r}")
    print(f"Assumptions: {rec.assumptions}")
    print(f"Notes: {rec.notes}")
    print(f"Refinement chips: {[c.label for c in rec.refinement_chips]}")
    print(f"Total time: {elapsed:.2f}s")

async def main():
    if not DB.exists():
        print(f"Catalog DB not found: {DB}")
        sys.exit(1)
    repo = ABOCatalogRepository(DB)
    stats = repo.stats()
    print(f"Catalog: {stats['listings']:,} products, {stats['product_types']:,} types")

    provider = os.getenv("RETAIL_PROVIDER", "deepseek")
    model = os.getenv("RETAIL_MODEL", "deepseek-v4-flash")
    print(f"Provider: {provider}  Model: {model}")
    print(f"vLLM endpoint: {os.getenv('RETAIL_BASE_URL', '(default)')}")

    client = build_chat_client(provider, model)
    tracker = CatalogEvidenceTracker()
    catalog_tools = build_tools(repo, catalog_tracker=tracker)
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker, provider=provider)

    for i, q in enumerate(QUERIES, 1):
        try:
            await run_one(i, q, agent, repo)
        except Exception as e:
            print(f"\n[ERROR on query {i}] {type(e).__name__}: {e}")

    repo.close()

if __name__ == "__main__":
    asyncio.run(main())
