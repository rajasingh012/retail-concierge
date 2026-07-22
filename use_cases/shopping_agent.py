"""Single conversational shopping agent with deterministic catalog safeguards."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from use_cases.brief import (
    EXTRACT_BRIEF_TOOL,
    _extract_budget_usd,
    _extract_dimensions,
    _extract_quantity,
    _make_budget_note,
)
from use_cases.ranking import screen_and_rank_candidates

FINALIZE_RECOMMENDATIONS_TOOL = "finalize_recommendations"

SHOPPING_AGENT_INSTRUCTIONS = """\
You are RetailConcierge, one conversational shopping agent. You clarify the
user's request when necessary, search the offline Amazon Berkeley Objects (ABO)
catalog with the tools, screen product identity, and present evidence-backed
recommendations.

Conversation:
- Keep the entire conversation in the supplied AgentSession.
- Ask one concise clarification only when proceeding could select the wrong
  product type, compatibility is unknown, explicit constraints conflict, or a
  must-have would otherwise be silently relaxed.
- Missing budget, brand, color, or a nice-to-have is not blocking. Proceed with
  a clearly stated assumption when useful results are possible.
- Treat a user's next message as the answer or refinement to the current request.

Catalog workflow:
0. Call extract_brief first to produce a structured shopping brief. It handles
   budget currency conversion, dimension parsing, and the two-question budget.
1. Call find_product_types only when an exact catalog product_type will
   materially narrow retrieval.
2. Call find_brands to resolve brand names against the catalog.
3. Call search_catalog with concrete title terms and limit=50. Broaden the title
   terms once if too few useful candidates are returned.
4. Pass through every candidate returned by search_catalog. Do not invent items.
   Each candidate must include its item_id, retrieval_rank, and the catalog
   flags (has_bullet, has_dimensions, has_weight, has_material) you received.
5. Classify each item as exactly one of: exact_product, accessory, unrelated,
   uncertain. Product identity is an eligibility decision, not a preference.
   Covers, mats, pillows, replacement parts, and add-ons are not the requested
   primary product.
6. Call finalize_recommendations exactly once with the full classified list.
   Application code removes ineligible products and applies deterministic
   ranking. Only the candidates and exact order returned by the finalizer are
   shown to the user.
7. Use only the candidates and exact order returned by finalize_recommendations.
   You may omit a candidate that contradicts an explicit must-have, but never
   restore an excluded item, add an unknown item, or reorder the result.

Final response:
- For a clarification, respond with only the natural-language question. Do not
  call finalize_recommendations first.
- For recommendations, return exactly one JSON object and no surrounding prose:
{
  "kind": "recommendations",
  "ranked": [
    {
      "rank": 1,
      "item_id": "...",
      "title_en": "...",
      "brand_en": "...",
      "product_type": "...",
      "product_url": "...",
      "why_it_fits": ["catalog-backed reason"],
      "trade_offs": ["specific limitation or unknown"]
    }
  ],
  "assumptions": ["assumption used to proceed"],
  "notes": ["screening or evidence limitation"],
  "recommendation": "one concise action",
  "refinement_chips": [
    {"label": "short option", "instruction": "self-contained refinement"}
  ],
  "dataset_notice": "This is an offline product catalog snapshot with typed dimensions, material, color, and brand metadata but no prices, ratings, or live availability."
}
Return at most five ranked products and four contextual refinement chips. Never
invent specifications, prices, ratings, availability, shipping, or warranties.
"""


class CatalogEvidenceTracker:
    """Track item_ids observed by catalog tools for one shopping session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, None] = OrderedDict()

    def record(self, item_ids: list[str]) -> None:
        with self._lock:
            for item_id in item_ids:
                self._seen.setdefault(item_id, None)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._seen)

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


def build_shopping_agent(
    client: OpenAIChatCompletionClient,
    catalog_tools: list[Any],
    *,
    tracker: CatalogEvidenceTracker | None = None,
) -> Agent:
    """Build the one MAF agent used for every turn in a shopping session."""
    tracker = tracker or CatalogEvidenceTracker()
    finalize = _make_finalize_tool(tracker)
    extract_brief = _make_extract_brief_tool()
    return Agent(
        client=client,
        instructions=SHOPPING_AGENT_INSTRUCTIONS,
        tools=[extract_brief, *catalog_tools, finalize],
    )


def build_finalize_recommendations_tool():
    """Build the deterministic finalization tool and its bound tracker."""
    tracker = CatalogEvidenceTracker()
    return _make_finalize_tool(tracker), tracker


def _make_finalize_tool(tracker: CatalogEvidenceTracker):
    @tool(
        name=FINALIZE_RECOMMENDATIONS_TOOL,
        description=(
            "Remove candidates that are not exact requested products and apply the "
            "catalog's deterministic ranking. Candidates must originate from "
            "search_catalog in this session. Returns the authoritative candidate order."
        ),
    )
    def finalize_recommendations(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Screen product identity and deterministically rank exact products."""
        return screen_and_rank_candidates(
            {"candidates": candidates},
            allowed_item_ids=tracker.snapshot(),
        )

    finalize_recommendations._catalog_tracker = tracker  # type: ignore[attr-defined]
    return finalize_recommendations


def _make_extract_brief_tool():
    """Build the structured brief extraction tool."""

    @tool(
        name=EXTRACT_BRIEF_TOOL,
        description=(
            "Extract a structured shopping brief from the user's request. "
            "Call this once before any catalog search tool. Handles budget currency "
            "conversion, dimension parsing, quantity, and the two-question "
            "clarification budget. Returns a canonical brief dict with parsed fields, "
            "assumptions, and evidence gaps."
        ),
    )
    def extract_brief(
        user_request: str,
        clarifications: list[dict[str, str]] = [],
        remaining_clarification_budget: int = 2,
    ) -> dict[str, Any]:
        """Extract a structured shopping brief from the user request.

        Args:
            user_request: The user's original or refined shopping request.
            clarifications: Previous Q&A pairs as [{"q": "...", "a": "..."}].
                Passed from the caller so the agent can reference them.
            remaining_clarification_budget: How many more questions can be asked
                this turn. 0 forces the brief to complete with best assumptions.

        Returns:
            A dict with either:
              {"complete": false, "question": "..."}
            or
              {"complete": true, "brief": {...}}
              where brief contains intent, search_terms, product_type, brand,
              budget_usd, budget_source, dimensions, must_have, nice_to_have,
              color, material, compatibility, target_use, quantity, assumptions,
              evidence_gaps, budget_notes, dimension_notes.
        """
        return _build_brief_response(user_request, clarifications, remaining_clarification_budget)

    return extract_brief


def _build_brief_response(
    user_request: str,
    clarifications: list[dict[str, str]],
    remaining_budget: int,
) -> dict[str, Any]:
    """Build the brief response with application-side parsing applied.

    This is a pure-data helper (no LLM call) that constructs the brief
    structure. The actual extraction of semantic fields from the request
    happens in the model's response when it calls this tool.
    """
    budget_usd, budget_source = _extract_budget_usd(user_request)
    dimensions, dim_note = _extract_dimensions(user_request)
    quantity = _extract_quantity(user_request)
    budget_notes = _make_budget_note(user_request, budget_usd, budget_source)

    assumption_prompts = []
    if budget_notes:
        assumption_prompts.extend(budget_notes)
    if dim_note:
        assumption_prompts.append(dim_note)

    brief = {
        "intent": "",
        "search_terms": "",
        "product_type": "",
        "brand": "",
        "budget_usd": budget_usd,
        "budget_source": budget_source,
        "max_dimension_cm": dimensions,
        "must_have": [],
        "nice_to_have": [],
        "color": "",
        "material": "",
        "compatibility": "",
        "target_use": "",
        "quantity": quantity,
        "assumptions": [],
        "evidence_gaps": [],
        "budget_notes": budget_notes,
        "dimension_notes": dim_note,
    }

    return {
        "complete": True,
        "brief": brief,
        "clarifications_asked": len(clarifications),
        "clarification_budget_remaining": max(0, remaining_budget - len(clarifications)),
    }


def enforce_finalized_recommendation(
    recommendation: dict[str, Any], finalized: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Drop unknown products and restore deterministic candidate order and facts."""
    if finalized is None:
        raise ValueError(
            "Shopping agent returned recommendations without finalizing candidates"
        )
    by_id = {
        candidate.get("item_id"): candidate
        for candidate in finalized
        if candidate.get("item_id")
    }
    agent_ranked = recommendation.get("ranked", [])
    if not isinstance(agent_ranked, list):
        raise ValueError("Shopping agent recommendation needs a ranked array")
    agent_by_id = {
        item.get("item_id"): item
        for item in agent_ranked
        if isinstance(item, dict) and item.get("item_id") in by_id
    }
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in finalized:
        item_id = candidate.get("item_id")
        if item_id not in agent_by_id or item_id in seen:
            continue
        rendered = dict(candidate)
        generated = agent_by_id[item_id]
        rendered["why_it_fits"] = generated.get("why_it_fits", [])
        rendered["trade_offs"] = generated.get("trade_offs", [])
        ranked.append(rendered)
        seen.add(item_id)
        if len(ranked) == 5:
            break
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    normalized = dict(recommendation)
    normalized["ranked"] = ranked
    return normalized


def finalized_candidates_from_response(response: Any) -> list[dict[str, Any]] | None:
    """Read the latest deterministic finalizer result from a MAF response."""
    call_names: dict[str, str] = {}
    latest: list[dict[str, Any]] | None = None
    for message in getattr(response, "messages", []):
        for content in getattr(message, "contents", []):
            content_type = getattr(content, "type", None)
            if content_type == "function_call":
                call_id = getattr(content, "call_id", None)
                name = getattr(content, "name", None)
                if call_id and name:
                    call_names[call_id] = name
                continue
            if content_type != "function_result":
                continue
            call = getattr(content, "function_call", None)
            name = getattr(call, "name", None) or call_names.get(
                getattr(content, "call_id", "")
            )
            if name != FINALIZE_RECOMMENDATIONS_TOOL:
                continue
            result = getattr(content, "result", None)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(result, dict) and isinstance(result.get("candidates"), list):
                latest = result["candidates"]
    return latest


def parse_recommendation(text: str) -> dict[str, Any]:
    """Parse and validate a recommendation response from the shopping agent.

    Extracts the first top-level JSON object from the text, accepting
    fenced code blocks and free-text narration before/after the JSON.
    """
    candidate = text.strip()
    # Strip fenced code blocks if present
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    # Brace-match through prose to find the first JSON object
    for start in range(len(candidate)):
        if candidate[start] == "{":
            depth = 0
            for end in range(start, len(candidate)):
                ch = candidate[end]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        block = candidate[start : end + 1]
                        try:
                            value = json.loads(block)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if not isinstance(value, dict) or value.get("kind") != "recommendations":
                            # Nested JSON object (e.g. a product inside ranked) —
                            # skip it and keep looking for the outer contract.
                            continue
                        if not isinstance(value.get("ranked"), list):
                            raise ValueError(
                                "Shopping agent recommendation needs a ranked array"
                            )
                        if not isinstance(value.get("refinement_chips", []), list):
                            raise ValueError(
                                "Shopping agent refinement_chips must be an array"
                            )
                        return value
                    break
    raise ValueError("Shopping agent returned an invalid recommendation")
