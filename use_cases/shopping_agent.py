"""Single conversational shopping agent with deterministic catalog safeguards."""
from __future__ import annotations
import json
import threading
from collections import OrderedDict
from typing import Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from infrastructure.chat_clients import provider_extras

from domain.recommendation import (
    FinalizedCandidate,
    MAX_RANKED_PRODUCTS,
    MAX_REFINEMENT_CHIPS,
    RankedItem,
    RecommendationResponse,
    RefinementChip,
    extract_json_object,
)
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
- For recommendations, return a single JSON object — no surrounding prose, no
  fenced code blocks, no markdown. Use EXACTLY these field names:
  {
    "kind": "recommendations",
    "ranked": [{"rank": 1, "item_id": "...", "title_en": "...", "brand_en": "...",
                "product_type": "...", "product_url": "...",
                "why_it_fits": ["..."], "trade_offs": ["..."]}],
    "assumptions": ["..."],
    "notes": ["..."],
    "recommendation": "...",
    "refinement_chips": [{"label": "...", "instruction": "..."}],
    "dataset_notice": "This is an offline product catalog snapshot..."
  }
  The "ranked" field name is REQUIRED (not "recommendations"). The
  "refinement_chips" field name is REQUIRED (not "refinements"). Top-level
  "kind" must equal the literal string "recommendations".
- At most 5 entries in "ranked" and 4 entries in "refinement_chips". Never
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


def _build_agent_tools(
    catalog_tools: list[Any],
    *,
    tracker: CatalogEvidenceTracker,
) -> list[Any]:
    """Compose the tool list for the shopping agent.

    Order is part of the contract: ``extract_brief`` runs first, the three
    catalog tools in between, and ``finalize_recommendations`` last so the
    model can use the evidence it observed earlier in the turn.
    """
    return [
        _make_extract_brief_tool(),
        *catalog_tools,
        _make_finalize_tool(tracker),
    ]


def build_shopping_agent(
    client: OpenAIChatCompletionClient,
    catalog_tools: list[Any],
    *,
    tracker: CatalogEvidenceTracker | None = None,
    provider: str = "",
) -> Agent:
    """Build the one MAF agent used for every turn in a shopping session.

    Args:
        client: MAF OpenAI-compatible chat client.
        catalog_tools: Search / brand / product-type tools.
        tracker: Evidence tracker shared between catalog and finalizer tools.
        provider: Provider name (``"minimax"``, ``"vllm"``, ``"deepseek"``).
            Provider-specific request extras (e.g. MiniMax
            ``reasoning_split``) are looked up via ``provider_extras`` and
            merged into ``default_options``. Unknown extras on a different
            provider are forwarded in ``extra_body`` and silently ignored
            by the server.
    """
    tracker = tracker or CatalogEvidenceTracker()
    provider_options = provider_extras(provider) if provider else {}
    return Agent(
        client=client,
        instructions=SHOPPING_AGENT_INSTRUCTIONS,
        tools=_build_agent_tools(catalog_tools, tracker=tracker),
        default_options={
            "response_format": RecommendationResponse,
            **provider_options,
        },
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
    def finalize_recommendations(
        candidates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Screen product identity and deterministically rank exact products."""
        result = screen_and_rank_candidates(
            {"candidates": candidates},
            allowed_item_ids=tracker.snapshot(),
        )
        # MAF serializes tool returns through a generic JSON encoder that does
        # not understand Pydantic; model_dump explicitly so the wire format is
        # a plain dict (which MAF round-trips losslessly).
        result["candidates"] = [
            candidate.model_dump() for candidate in result["candidates"]
        ]
        return result

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


def structured_recommendation_from_response(response: Any) -> RecommendationResponse | None:
    """Read the typed recommendation from a MAF response.

    Tries ``response.value`` first (provider-native JSON-schema enforcement,
    e.g. vLLM and OpenAI). Falls back to extracting the JSON object from
    ``response.text`` and validating against the schema — for providers
    (MiniMax, DeepSeek) that deliver wrapped or narrated output instead.
    """
    try:
        value = getattr(response, "value", None)
    except Exception:
        value = None
    if isinstance(value, RecommendationResponse):
        return value
    text = getattr(response, "text", None)
    if not text:
        return None
    try:
        return RecommendationResponse.model_validate_json(extract_json_object(text))
    except Exception:
        return None


def finalized_candidates_from_response(response: Any) -> list[FinalizedCandidate] | None:
    """Read the latest deterministic finalizer result from a MAF response.

    Walks messages in order, collecting function_call names keyed by call_id,
    then matches each function_result to its calling tool via call_id. MAF
    stores the tool's return as a JSON string in ``Content.result``; decode it
    and validate each candidate against the ``FinalizedCandidate`` schema.
    """
    import json

    call_names: dict[str, str] = {}
    latest: list[FinalizedCandidate] | None = None
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
            call_id = getattr(content, "call_id", "")
            name = call_names.get(call_id)
            if name != FINALIZE_RECOMMENDATIONS_TOOL:
                continue
            result = getattr(content, "result", None)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(result, dict):
                continue
            raw_candidates = result.get("candidates")
            if not isinstance(raw_candidates, list):
                continue
            try:
                latest = [FinalizedCandidate.model_validate(item) for item in raw_candidates]
            except Exception:
                continue
    return latest


def enforce_finalized_recommendation(
    recommendation: RecommendationResponse | dict[str, Any],
    finalized: list[FinalizedCandidate] | None,
) -> RecommendationResponse:
    """Drop unknown products and restore deterministic candidate order and facts.

    Accepts either a ``RecommendationResponse`` (the typed result from MAF) or a
    plain dict (back-compat for callers that haven't migrated). Returns the
    typed model.
    """
    if finalized is None:
        raise ValueError(
            "Shopping agent returned recommendations without finalizing candidates"
        )
    by_id = {candidate.item_id: candidate for candidate in finalized}

    if isinstance(recommendation, RecommendationResponse):
        agent_ranked = [item.model_dump() for item in recommendation.ranked]
    else:
        agent_ranked = recommendation.get("ranked", [])
    if not isinstance(agent_ranked, list):
        raise ValueError("Shopping agent recommendation needs a ranked array")

    agent_by_id: dict[str, dict[str, Any]] = {}
    for item in agent_ranked:
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        if isinstance(item_id, str) and item_id in by_id and item_id not in agent_by_id:
            agent_by_id[item_id] = item

    ranked: list[RankedItem] = []
    seen: set[str] = set()
    for candidate in finalized:
        if candidate.item_id not in agent_by_id or candidate.item_id in seen:
            continue
        generated = agent_by_id[candidate.item_id]
        ranked.append(
            RankedItem(
                rank=len(ranked) + 1,
                item_id=candidate.item_id,
                title_en=candidate.title_en,
                brand_en=candidate.brand_en,
                product_type=candidate.product_type,
                product_url=candidate.product_url,
                why_it_fits=list(generated.get("why_it_fits", []) or []),
                trade_offs=list(generated.get("trade_offs", []) or []),
            )
        )
        seen.add(candidate.item_id)
        if len(ranked) == MAX_RANKED_PRODUCTS:
            break

    if isinstance(recommendation, RecommendationResponse):
        return recommendation.model_copy(update={"ranked": ranked})
    return RecommendationResponse(
        kind="recommendations",
        ranked=ranked,
        assumptions=list(recommendation.get("assumptions", []) or []),
        notes=list(recommendation.get("notes", []) or []),
        recommendation=str(recommendation.get("recommendation", "") or ""),
        refinement_chips=_parse_refinement_chips(recommendation.get("refinement_chips", [])),
        dataset_notice=str(
            recommendation.get("dataset_notice", RecommendationResponse.model_fields["dataset_notice"].default) or ""
        ),
    )


def _parse_refinement_chips(raw: Any) -> list[RefinementChip]:
    """Validate, deduplicate, and cap refinement chips to MAX_REFINEMENT_CHIPS."""
    if not isinstance(raw, list):
        return []
    chips: list[RefinementChip] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        instruction = item.get("instruction")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        key = (label.strip(), instruction.strip())
        if key in seen:
            continue
        seen.add(key)
        chips.append(RefinementChip(label=key[0], instruction=key[1]))
        if len(chips) == MAX_REFINEMENT_CHIPS:
            break
    return chips