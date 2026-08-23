"""End-to-end smoke test of the vector-search-hybrid PR against real DeepSeek.

Skipped unless DEEPSEEK_API_KEY is set in the environment AND the chair DB
has vec_items populated. When run, it exercises the real shopping agent
against the chair DB (2,173 BGE-small embeddings) using the live DeepSeek
provider and asserts:

  - search_vector was called at least 3 of 4 times
    (proves the LLM is using the new semantic search path)
  - Q4 "red velvet accent chair" activates the structured-filter pre-filter
    (proves the brief fields reach finalize_recommendations after the
    system-prompt rewrite)
  - At least 3 of 4 queries returned non-empty ranked output

Run as a plain Python script (not via pytest) because the test uses a
plain async function — no pytest-asyncio dependency.

Usage:
    DEEPSEEK_API_KEY=... uv run python tests/test_deepseek_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

CHAIR_DB = Path(__file__).resolve().parents[1] / "retail_catalog_chair.db"


def _chair_db_has_vec_items() -> bool:
    """Return True iff the chair DB exists AND vec_items is populated."""
    if not CHAIR_DB.is_file():
        return False
    try:
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(str(CHAIR_DB))
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        cur = conn.execute("SELECT COUNT(*) FROM vec_items")
        n = cur.fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set", file=sys.stderr)
        return 0
    if not _chair_db_has_vec_items():
        print(
            "SKIPPED: chair DB missing or vec_items not built "
            "(run scripts/build_vector_index.py)",
            file=sys.stderr,
        )
        return 0

    rc = asyncio.run(_run_smoke())
    return rc


async def _run_smoke() -> int:
    from infrastructure.agent_tools import build_tools
    from infrastructure.chat_clients import build_chat_client
    from infrastructure.database import ABOCatalogRepository
    from main import _load_catalog_vocabulary
    from use_cases.shopping_agent import build_shopping_agent

    os.environ.setdefault("RETAIL_DB", str(CHAIR_DB))
    os.environ.setdefault("RETAIL_PROVIDER", "deepseek")
    os.environ.setdefault("RETAIL_MODEL", "deepseek-v4-flash")

    QUERIES = [
        ("Q1 exact-keyword",   "office chair with lumbar support"),
        ("Q2 paraphrase",      "chair for back pain"),
        ("Q3 synonym",         "couch"),
        ("Q4 color+material",  "red velvet accent chair under $200"),
    ]

    def _extract_tool_calls(messages):
        calls = []
        for message in messages:
            for content in getattr(message, "contents", []):
                if getattr(content, "type", None) == "function_call":
                    args_raw = getattr(content, "arguments", None)
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {"_raw": args_raw}
                    else:
                        args = args_raw
                    calls.append({
                        "name": getattr(content, "name", None),
                        "args": args,
                    })
        return calls

    async def run_query(agent, label, text):
        from agent_framework import AgentSession

        session = AgentSession()
        t0 = time.perf_counter()
        response = await agent.run(text, session=session)
        elapsed = time.perf_counter() - t0

        tool_calls = _extract_tool_calls(getattr(response, "messages", []))
        tool_names = [c["name"] for c in tool_calls]
        structured_filter_active = False
        for c in tool_calls:
            if c["name"] == "finalize_recommendations":
                args = c.get("args") or {}
                sf = args.get("structured_filter") if isinstance(args, dict) else None
                if sf and any(v for v in sf.values() if v):
                    structured_filter_active = True
                    break

        from use_cases.shopping_agent import structured_recommendation_from_response
        rec = structured_recommendation_from_response(response)
        ranked = [r.item_id for r in (rec.ranked if rec else [])]
        titles = [r.title_en for r in (rec.ranked if rec else [])]

        print(f"\n=== {label} ===", flush=True)
        print(f"query:   {text!r}", flush=True)
        print(f"latency: {elapsed:.1f}s", flush=True)
        print(f"tools:   {tool_names}", flush=True)
        print(f"structured_filter_active: {structured_filter_active}", flush=True)
        print(f"ranked:  {ranked[:5]}", flush=True)
        for t in titles[:3]:
            print(f"  - {t[:80]}", flush=True)

        return {
            "label": label,
            "query": text,
            "latency_s": round(elapsed, 2),
            "tool_names": tool_names,
            "search_vector_called": "search_vector" in tool_names,
            "structured_filter_active": structured_filter_active,
            "ranked_item_ids": ranked,
        }

    repo = ABOCatalogRepository(str(CHAIR_DB))
    client = build_chat_client(
        os.environ["RETAIL_PROVIDER"],
        os.environ["RETAIL_MODEL"],
    )
    catalog_vocabulary = _load_catalog_vocabulary(repo)
    catalog_tools = build_tools(repo)
    agent = build_shopping_agent(
        client,
        catalog_tools,
        provider=os.environ["RETAIL_PROVIDER"],
        audit_logger=None,
        catalog_vocabulary=catalog_vocabulary,
        repository=repo,
    )

    summary = []
    for label, text in QUERIES:
        try:
            summary.append(await run_query(agent, label, text))
        except Exception as exc:
            print(f"\n=== {label} FAILED ===\n  error: {exc}", flush=True)
            summary.append({"label": label, "query": text, "error": str(exc)})

    repo.close()

    n = len(summary)
    sv_calls = sum(1 for r in summary if r.get("search_vector_called"))
    success = sum(1 for r in summary if r.get("ranked_item_ids"))
    q4 = next((r for r in summary if r.get("label") == "Q4 color+material"), {})

    print("\n" + "=" * 60, flush=True)
    print("DEEPSEEK SMOKE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"queries: {n}, ranked: {success}/{n}, vector: {sv_calls}/{n}", flush=True)
    print(f"Q4 structured_filter_active: {q4.get('structured_filter_active')}", flush=True)

    failures = []
    if sv_calls < 3:
        failures.append(
            f"search_vector was only called {sv_calls}/{n} times — "
            "semantic path is dormant"
        )
    if success < 3:
        failures.append(
            f"only {success}/{n} queries returned ranked output"
        )
    if not q4.get("structured_filter_active"):
        failures.append(
            "Q4 'red velvet accent chair' did NOT activate the structured-filter "
            "pre-filter — brief color/material fields not reaching finalize"
        )
    if failures:
        for f in failures:
            print(f"FAIL: {f}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
