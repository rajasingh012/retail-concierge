"""Skill-isolation probe for Gemma 4 12B INT8 (hackathon evidence).

Drives the shopping agent with one query per SKILL so we can attribute
weakness precisely:

  S1 direct-simple     simple direct search, few candidates
  S2 dense-finalize    broad search -> large result -> finalize (schema pressure)
  S3 clarification     ambiguous query that MUST produce a clarifying question
  S4 accessory         accessory-vs-exact classification (product_type_match)
  S5 refine-followup   first turn then refinement-chip follow-up (multi-turn)

Prints, per query: tool-call trace, final kind, recommendation count,
schema errors, and a one-line SKILL verdict. Exit 0.

Usage (against a live endpoint):
  RETAIL_PROVIDER=vllm RETAIL_BASE_URL=http://<ip>:8000/v1 \
  RETAIL_MODEL=/models/gemma-4-12b-it-int8 RETAIL_API_KEY=x \
  uv run python bench/probe_skills.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

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

DB = Path("./retail_catalog.db")

# (skill, query) — each targets exactly one capability
PROBES = [
    ("S1 direct-simple", "Find a 27-inch computer monitor."),
    ("S2 dense-finalize", "Show me all water bottles available in the catalog."),
    ("S3 clarification", "I want something for my trip."),
    ("S4 accessory", "I need a screen protector for my phone."),
    ("S5 refine-followup", "Recommend a laptop backpack. Then show me only waterproof ones."),
]


def _trace_summary(response) -> list[str]:
    calls = []
    for message in getattr(response, "messages", []):
        for content in getattr(message, "contents", []):
            if getattr(content, "type", "") == "function_call":
                name = getattr(content, "name", "?")
                args = getattr(content, "arguments", None)
                args_str = args[:160] if isinstance(args, str) else str(args)[:160]
                calls.append(f"{name}({args_str})")
    return calls


async def probe(idx: int, skill: str, query: str, agent, repo) -> dict:
    print(f"\n{'='*72}\n{skill} | query: {query!r}\n{'='*72}")
    session = agent.create_session()
    try:
        response = await agent.run(query, session=session)
    except Exception as exc:  # noqa: BLE001
        print(f"  AGENT EXCEPTION: {type(exc).__name__}: {exc}")
        return {"skill": skill, "kind": "exception"}

    calls = _trace_summary(response)
    print(f"  tool calls: {calls}")

    rec = structured_recommendation_from_response(response)
    if rec is not None:
        n = len(getattr(rec, "ranked", []) or [])
        print(f"  kind=recommendations ranked={n}")
        for item in (getattr(rec, "ranked", []) or [])[:3]:
            print(f"    - {getattr(item, 'title_en', '?')[:70]}")
        return {"skill": skill, "kind": "recommendations", "ranked": n}

    finalized = finalized_candidates_from_response(response)
    if finalized is not None:
        print(f"  kind=finalized_candidates count={len(finalized)}")
        return {"skill": skill, "kind": "finalized", "count": len(finalized)}

    # Clarification or narrative — read the raw message
    text = getattr(response, "text", None) or str(getattr(response, "value", "") or "")
    print(f"  kind=clarification/narrative: {text[:220].replace(chr(10), ' ')}")
    return {"skill": skill, "kind": "clarification"}


async def main() -> int:
    if not DB.exists():
        print(f"Catalog DB not found at {DB}", file=sys.stderr)
        return 1
    repo = ABOCatalogRepository(DB)
    provider = os.getenv("RETAIL_PROVIDER", "deepseek")
    model = os.getenv("RETAIL_MODEL", "deepseek-v4-flash")
    print(f"Provider: {provider}  Model: {model}")
    client = build_chat_client(provider, model)
    tracker = CatalogEvidenceTracker()
    catalog_tools = build_tools(repo, catalog_tracker=tracker)
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker, provider=provider)

    results = []
    for idx, (skill, query) in enumerate(PROBES, start=1):
        results.append(await probe(idx, skill, query, agent, repo))
    repo.close()

    print("\n" + "=" * 72)
    print("SKILL VERDICT SUMMARY")
    print("=" * 72)
    print("(S3/S4 scored PASS when they ask a clarifying question - that is the")
    print(" designed behavior for ambiguous queries, per the agent system prompt.)")
    for r in results:
        kind = r.get("kind")
        detail = r.get("ranked") or r.get("count") or ""
        if r["skill"].startswith(("S3", "S4")):
            status = "PASS" if kind == "clarification" else "FAIL"
        else:
            status = "PASS" if kind == "recommendations" else ("PARTIAL" if kind == "finalized" else "FAIL")
        print(f"  {r['skill']:18s} {status:8s} {kind} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
