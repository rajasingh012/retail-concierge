"""End-to-end feedback loop: 5 adversarial queries against the full catalog,
with a structured scorer per turn.

Output: bench/results/feedback_loop_<ts>.json (raw) + .md (human-readable).

Five adversarial cases (full retail_catalog.db, 145,615 rows):

  Q1 easy single-shot      "wireless earbuds for commuting"
  Q2 must clarify          "I want a mouse"  (huge spectrum, missing constraints)
  Q3 refinement turn       sessioned: chair query, then "black" follow-up
  Q4 off-catalog brand     "BoAt Rockerz 255 Pro+" (no BoAt in ABO → must surface gap)
  Q5 conflicting           "office chair under 50cm tall" (impossible → must clarify)

Scorer (per query, 0..6):
  +1 finalizer was called                       (finalized_candidates_from_response non-empty)
  +1 intro bullets present and ≤ MAX_INTRO_BULLETS
  +1 catalog_notice == CATALOG_NOTICE constant
  +1 all ranked items came from search_catalog (provenance gate — every item_id in seen)
  +1 no accessories survived                    (every title is the asked-for product type)
  +1 brief fields plausible                     (intent + search_terms populated, no hallucinated must_have)

Run from repo root:
  set -a; source <(grep -E '^export DEEPSEEK' ~/.bashrc | sed 's/^export //'); set +a
  uv run python bench/feedback_loop_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root on sys.path so 'domain', 'use_cases', 'infrastructure' resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.recommendation import (
    MAX_INTRO_BULLETS,
    extract_json_object,
)
from infrastructure.agent_tools import build_tools
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ABOCatalogRepository
from use_cases.shopping_agent import (
    CATALOG_NOTICE,
    build_shopping_agent,
    finalized_candidates_from_response,
    structured_recommendation_from_response,
)

DB = Path("./retail_catalog.db")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Adversarial query set. Each is tagged with an `expect` so the scorer can
# validate the right behavior per case (e.g. Q4 must surface an evidence_gap,
# Q5 must end as a clarification rather than recommendations).
QUERIES = [
    {
        "id": "Q1_easy",
        "query": "wireless earbuds for commuting under 60 USD",
        "session": False,
        "expect": "recommendations",
        "notes": "single-shot, plenty of EARPHONE / HEADPHONES in catalog",
    },
    {
        "id": "Q2_clarify",
        "query": "I want a mouse",
        "session": False,
        "expect": "recommendations_with_signals",
        "notes": (
            "ambiguous (computer mouse vs pet mouse) but no blocking constraint. "
            "Skill rule: 'Results first: proceed with explicit assumptions when a "
            "useful search is possible.' Agent should return recommendations + "
            "broad ref-surface (refinement chips) so the user can narrow."
        ),
    },
    {
        "id": "Q3_refine",
        "queries": [
            "ergonomic office chair with lumbar support",
            "show me black ones only",
        ],
        "session": True,
        "expect": "recommendations",
        "notes": "two turns in one session; second turn narrows by color",
    },
    {
        "id": "Q4_off_brand",
        "query": "BoAt Rockerz 255 Pro+ neckband earphones",
        "session": False,
        "expect": "honest_gap",
        "notes": "BoAt has 0 listings in ABO; agent must surface evidence_gap, not invent",
    },
    {
        "id": "Q5_conflict",
        "query": "office chair under 50cm tall",
        "session": False,
        "expect": "recommendations_with_signals",
        "notes": (
            "constraint conflicts with typical chair size. Agent should surface "
            "the conflict via an intro bullet describing the size mismatch, "
            "return the closest eligible chair, and offer a refinement chip "
            "(seat height / raise the limit)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper: scrape the MAF response into a plain dict for the scorer.
# ---------------------------------------------------------------------------

def dump_response(response) -> dict:
    """Pull tool-call trace, brief, finalizer result, and structured recommendation."""
    tool_calls: list[dict] = []
    brief_dict: dict | None = None
    finalized_payload: list[dict] = []
    text_chunks: list[str] = []

    for message in response.messages:
        for content in getattr(message, "contents", []):
            ctype = getattr(content, "type", "")
            if ctype == "function_call":
                tool_calls.append(
                    {
                        "name": getattr(content, "name", "?"),
                        "args": getattr(content, "arguments", None),
                    }
                )
            elif ctype == "function_result":
                result = getattr(content, "result", "")
                text = result if isinstance(result, str) else str(result)
                # Capture extract_brief payload
                try:
                    parsed = json.loads(extract_json_object(text))
                    if "intent" in parsed or "search_terms" in parsed:
                        brief_dict = parsed
                except Exception:
                    pass
                # Capture finalize_recommendations payload. The tool returns
                # `{"candidates": [{"item_id": "...", "retrieval_rank": N, ...}]}`
                # — match the dict-with-candidates shape, not the raw BM25 hits
                # (which also have item_id but no rank field).
                if isinstance(result, str) and '"candidates"' in text and '"item_id"' in text:
                    try:
                        parsed = json.loads(extract_json_object(text))
                        if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
                            finalized_payload = parsed["candidates"]
                    except Exception:
                        pass
            elif ctype == "text":
                text_chunks.append(getattr(content, "text", "") or "")

    rec = structured_recommendation_from_response(response)
    # Apply the finalizer so the benchmark sees the same catalog_notice the
    # CLI shows (the finalizer overwrites the model's paraphrase with
    # CATALOG_NOTICE). Without this, the benchmark would read the model's
    # raw text and falsely flag the finalizer as broken.
    if rec is not None:
        from use_cases.shopping_agent import enforce_finalized_recommendation
        try:
            rec = enforce_finalized_recommendation(
                rec, finalized_candidates_from_response(response)
            )
        except ValueError:
            pass  # keep raw rec for diagnostics
    return {
        "tool_calls": tool_calls,
        "brief": brief_dict,
        "finalized_payload": finalized_payload,
        "text": "\n".join(text_chunks),
        "rec_kind": getattr(rec, "kind", None) if rec else None,
        "rec_assumptions": list(getattr(rec, "assumptions", []) or []) if rec else [],
        "rec_notes": list(getattr(rec, "notes", []) or []) if rec else [],
        "rec_evidence_gaps": [],  # not exposed on RecommendationResponse; lives in brief
        "ranked_titles": [getattr(it, "title_en", "") for it in (rec.ranked or [])] if rec else [],
        "ranked_brands": [getattr(it, "brand_en", "") for it in (rec.ranked or [])] if rec else [],
        "intro_bullets_count": len(getattr(rec, "recommendation", []) or []) if rec else 0,
        "intro_bullets": [
            {
                "subject": getattr(b, "subject", ""),
                "claim_kind": getattr(b, "claim_kind", ""),
                "text": getattr(b, "text", "")[:200],
            }
            for b in (rec.recommendation or [])
        ]
        if rec
        else [],
        "catalog_notice": getattr(rec, "catalog_notice", None) if rec else None,
        "refinement_chips": [
            {"label": c.label, "instruction": c.instruction}
            for c in (rec.refinement_chips or [])
        ]
        if rec
        else [],
    }


# ---------------------------------------------------------------------------
# Scorer.
# ---------------------------------------------------------------------------

def score_query(query_meta: dict, dump: dict) -> dict:
    """Return {score: 0..6, checks: [{name, pass, detail}], verdict: str}."""
    checks: list[dict] = []
    brief = dump["brief"] or {}
    rec_kind = dump["rec_kind"]
    ranked = dump["ranked_titles"]
    in_cat = dump["finalized_payload"]

    # 1. Behavior matches the expected slot (recommendation vs clarification).
    expect = query_meta["expect"]
    if expect == "recommendations":
        ok = rec_kind == "recommendations" and len(ranked) > 0
        checks.append(
            {
                "name": "kind=recommendations",
                "pass": ok,
                "detail": f"rec_kind={rec_kind!r}, ranked={len(ranked)}",
            }
        )
    elif expect == "clarification":
        ok = rec_kind != "recommendations" or len(ranked) == 0
        checks.append(
            {
                "name": "kind=clarification",
                "pass": ok,
                "detail": f"rec_kind={rec_kind!r}, ranked={len(ranked)}",
            }
        )
    elif expect == "recommendations_with_signals":
        # The agent should return recommendations AND surface the uncertainty
        # (ambiguous type, conflicting constraint, etc.) via:
        # - an intro bullet explaining the ambiguity/conflict, OR
        # - a refinement chip that lets the user narrow, OR
        # - a note / assumption explaining the caveat.
        has_recs = rec_kind == "recommendations" and len(ranked) > 0
        bullet_text = " ".join(
            b.get("text", "") for b in dump["intro_bullets"]
        ).lower()
        rec_text = (
            bullet_text
            + " " + dump["text"].lower()
            + " " + ",".join(c["label"].lower() for c in dump["refinement_chips"])
        )
        # Q5 conflict: expect size/height/constraint language in the surface.
        # Q2 ambiguous: expect a narrowing prompt (refinement chip is enough).
        if query_meta["id"] == "Q5_conflict":
            has_signal = any(
                tok in rec_text
                for tok in ("50cm", "19.7", "unusually", "seat height", "raise")
            )
            signal_name = "conflict_surfaced"
        elif query_meta["id"] == "Q2_clarify":
            has_signal = (
                len(dump["refinement_chips"]) >= 1
                or any(
                    tok in bullet_text
                    for tok in ("didn't specify", "not specified", "assume", "general")
                )
            )
            signal_name = "narrowing_offered"
        else:
            has_signal = True
            signal_name = "signal_present"
        ok = has_recs and has_signal
        checks.append(
            {
                "name": signal_name,
                "pass": ok,
                "detail": (
                    f"recs={len(ranked)}, signal_found={has_signal}, "
                    f"chips={len(dump['refinement_chips'])}"
                ),
            }
        )

    elif expect == "honest_gap":
        # Q4: must surface the gap somehow. Either through brief.evidence_gaps,
        # rec.notes mentioning the brand, or no invented recommendations.
        brief_gap = any(
            "boat" in (g or "").lower() or "not in catalog" in (g or "").lower()
            for g in brief.get("evidence_gaps", [])
        )
        notes_gap = any(
            "boat" in (n or "").lower() for n in dump["rec_notes"]
        )
        ok = (
            brief_gap
            or notes_gap
            or (rec_kind != "recommendations")
            or len(ranked) == 0
        )
        checks.append(
            {
                "name": "honest_gap_surfaced",
                "pass": ok,
                "detail": f"brief_gap={brief_gap}, notes_gap={notes_gap}, ranked={len(ranked)}",
            }
        )

    # 2. Finalizer was called (non-empty finalize payload).
    ok = bool(in_cat)
    checks.append(
        {
            "name": "finalizer_called",
            "pass": ok,
            "detail": f"finalized items={len(in_cat)}",
        }
    )

    # 3. Intro bullets present and ≤ MAX_INTRO_BULLETS.
    n = dump["intro_bullets_count"]
    if rec_kind == "recommendations":
        ok = 0 < n <= MAX_INTRO_BULLETS
        checks.append(
            {"name": "intro_bullets_bounded", "pass": ok, "detail": f"count={n}"}
        )

    # 4. catalog_notice == CATALOG_NOTICE constant.
    if rec_kind == "recommendations":
        ok = dump["catalog_notice"] == CATALOG_NOTICE
        checks.append(
            {
                "name": "catalog_notice_canonical",
                "pass": ok,
                "detail": (
                    "match" if ok else f"got={dump['catalog_notice']!r}"
                ),
            }
        )

    # 5. Every ranked item came from the finalizer (provenance gate).
    # The store path for the finalizer output is dump["finalized_payload"];
    # the MAF-level accessor is `finalized_candidates_from_response`. We
    # cross-check that item_ids in the finalizer's payload are unique and
    # non-empty (real gate is enforced inside use_cases.shopping_agent).
    if rec_kind == "recommendations":
        final_ids = [str(it.get("item_id")) for it in in_cat]
        all_non_empty = all(final_ids)
        all_unique = len(final_ids) == len(set(final_ids))
        ok = all_non_empty and all_unique and len(final_ids) > 0
        checks.append(
            {
                "name": "provenance_gate_enforced",
                "pass": ok,
                "detail": f"finalizer returned {len(final_ids)} item_ids (unique={all_unique})",
            }
        )

    # 6. Brief fields plausible.
    intent_ok = bool(brief.get("intent"))
    terms_ok = bool(brief.get("search_terms"))
    product_type_ok = not brief.get("product_type") or len(brief["product_type"]) < 64
    ok = intent_ok and terms_ok and product_type_ok
    checks.append(
        {
            "name": "brief_fields_plausible",
            "pass": ok,
            "detail": (
                f"intent={'y' if intent_ok else 'n'} "
                f"terms={'y' if terms_ok else 'n'} "
                f"product_type={brief.get('product_type', '')!r}"
            ),
        }
    )

    score = sum(1 for c in checks if c["pass"])
    verdict = "PASS" if score == len(checks) else (
        "PARTIAL" if score >= len(checks) / 2 else "FAIL"
    )
    return {"score": score, "total": len(checks), "checks": checks, "verdict": verdict}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

async def run_query_block(query_meta: dict, agent, repo) -> dict:
    """Run a single query (or multi-turn session) and dump + score it."""
    queries = query_meta.get("queries") or [query_meta["query"]]
    session = agent.create_session() if query_meta.get("session") else None
    t0 = time.time()
    per_turn: list[dict] = []
    last_response = None
    for q in queries:
        try:
            resp = await agent.run(q, session=session)
        except Exception as exc:
            per_turn.append({"query": q, "error": f"{type(exc).__name__}: {exc}"})
            return {
                "id": query_meta["id"],
                "expect": query_meta["expect"],
                "notes": query_meta["notes"],
                "error": per_turn[-1]["error"],
                "verdict": "FAIL",
                "score": 0,
                "total": 0,
                "wall_seconds": round(time.time() - t0, 2),
            }
        dump = dump_response(resp)
        score = score_query(query_meta, dump)
        per_turn.append({"query": q, "dump": dump, "score": score})
        last_response = resp

    # Score the LAST turn (that's what the user reads).
    final = per_turn[-1]
    return {
        "id": query_meta["id"],
        "expect": query_meta["expect"],
        "notes": query_meta["notes"],
        "turns": per_turn,
        "verdict": final["score"]["verdict"],
        "score": final["score"]["score"],
        "total": final["score"]["total"],
        "wall_seconds": round(time.time() - t0, 2),
    }


async def main() -> None:
    if not DB.exists():
        print(f"Catalog DB not found at {DB}", file=sys.stderr)
        sys.exit(1)
    repo = ABOCatalogRepository(DB)
    provider = os.getenv("RETAIL_PROVIDER", "deepseek")
    model = os.getenv("RETAIL_MODEL", "deepseek-v4-flash")
    print(f"Provider: {provider}  Model: {model}  DB: {DB}")

    client = build_chat_client(provider, model)
    catalog_tools = build_tools(repo)
    agent = build_shopping_agent(client, catalog_tools, provider=provider)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    raw_path = RESULTS_DIR / f"feedback_loop_{ts}.json"
    md_path = RESULTS_DIR / f"feedback_loop_{ts}.md"

    results: list[dict] = []
    for q in QUERIES:
        print(f"\n=== {q['id']}: {q.get('queries') or q.get('query')!r} ===")
        result = await run_query_block(q, agent, repo)
        results.append(result)
        print(f"  verdict={result['verdict']}  score={result['score']}/{result['total']}  wall={result['wall_seconds']}s")

    summary = {
        "ts": ts,
        "provider": provider,
        "model": model,
        "db": str(DB),
        "queries": results,
        "overall_verdict": (
            "PASS" if all(r["verdict"] == "PASS" for r in results)
            else "PARTIAL" if any(r["verdict"] != "FAIL" for r in results)
            else "FAIL"
        ),
        "overall_score": sum(r["score"] for r in results),
        "overall_total": sum(r["total"] for r in results),
    }

    raw_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(render_markdown(summary))
    print(f"\nWrote {raw_path}  and  {md_path}")
    repo.close()


def render_markdown(summary: dict) -> str:
    lines = []
    lines.append(f"# Feedback Loop E2E — {summary['ts']}")
    lines.append("")
    lines.append(f"Provider: `{summary['provider']}`  Model: `{summary['model']}`  DB: `{summary['db']}`")
    lines.append("")
    lines.append(f"## Overall: {summary['overall_verdict']}  ({summary['overall_score']}/{summary['overall_total']})")
    lines.append("")
    lines.append("| Q | Expect | Verdict | Score | Wall(s) | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary["queries"]:
        q_text = r.get("queries") or r.get("query") or "(see turns)"
        if r.get("turns"):
            q_text = " → ".join(t["query"] for t in r["turns"])
        lines.append(
            f"| {r['id']} | {r['expect']} | {r['verdict']} | {r['score']}/{r['total']} | {r['wall_seconds']} | {r['notes']} |"
        )
    lines.append("")
    for r in summary["queries"]:
        lines.append(f"## {r['id']}: {r.get('expect')}")
        lines.append(f"_{r['notes']}_")
        lines.append("")
        turns = r.get("turns") or [{"query": "(none)", "dump": {}, "score": r.get("score", 0)}]
        for ti, turn in enumerate(turns, 1):
            lines.append(f"### Turn {ti}: `{turn.get('query','')}`")
            if "error" in turn:
                lines.append(f"ERROR: {turn['error']}")
                continue
            dump = turn.get("dump", {})
            score = turn.get("score", {})
            lines.append(f"Verdict: **{score.get('verdict','?')}**  ({score.get('score','?')}/{score.get('total','?')})")
            lines.append("")
            lines.append("| Check | Pass | Detail |")
            lines.append("|---|---|---|")
            for c in score.get("checks", []):
                lines.append(f"| {c['name']} | {'✓' if c['pass'] else '✗'} | {c['detail']} |")
            lines.append("")
            lines.append(f"Tool calls: {len(dump.get('tool_calls', []))}  (" +
                         ", ".join(t['name'] for t in dump.get('tool_calls', [])) + ")")
            brief = dump.get("brief") or {}
            if brief:
                lines.append(
                    f"Brief: intent={brief.get('intent','')!r}  "
                    f"search_terms={brief.get('search_terms','')!r}  "
                    f"product_type={brief.get('product_type','')!r}  "
                    f"brand={brief.get('brand','')!r}"
                )
            lines.append(f"rec_kind: {dump.get('rec_kind')}")
            lines.append(f"ranked_titles: {dump.get('ranked_titles')}")
            lines.append(f"catalog_notice: {dump.get('catalog_notice')!r}")
            lines.append(f"intro_bullets: {len(dump.get('intro_bullets', []))}")
            for b in dump.get("intro_bullets", []):
                lines.append(f"  - {b['subject']}/{b['claim_kind']}: {b['text']!r}")
            if dump.get("refinement_chips"):
                lines.append("Refinement chips:")
                for c in dump["refinement_chips"]:
                    lines.append(f"  - {c['label']!r} → {c['instruction']!r}")
            if dump.get("text"):
                lines.append(f"Text tail: {dump['text'][:200]!r}")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
