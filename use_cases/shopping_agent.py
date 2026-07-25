"""Single conversational shopping agent with deterministic catalog safeguards."""
from __future__ import annotations
import json
import threading
from collections import OrderedDict
from typing import Annotated, Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

from infrastructure.chat_clients import provider_extras

from domain.recommendation import (
    FinalizedCandidate,
    MAX_RANKED_PRODUCTS,
    MAX_REFINEMENT_CHIPS,
    RankedItem,
    RecommendationResponse,
    RefinementChip,
    ShoppingBrief,
    extract_json_object,
)
from use_cases.ranking import screen_and_rank_candidates

EXTRACT_BRIEF_TOOL = "extract_brief"
FINALIZE_RECOMMENDATIONS_TOOL = "finalize_recommendations"

SHOPPING_AGENT_INSTRUCTIONS = """\
You are RetailConcierge, one conversational shopping agent. You clarify the
user's request when necessary, search the offline Amazon Berkeley Objects (ABO)
catalog with the tools, screen product identity, and present evidence-backed
recommendations.

Conversation:
- Ask one concise clarification only when constraints conflict or a must-have
  would be silently relaxed.
- Treat the user's next message as an answer or refinement to the current request.

Catalog workflow:
0. Call extract_brief first, passing a fully populated brief argument. The
   brief is the single source of truth for the rest of the turn.
1. Call find_product_types only when an exact catalog product_type will
   materially narrow retrieval.
2. Call find_brands to resolve brand names against the catalog.
3. Call search_catalog with concrete title terms and limit=50. Broaden the title
   terms once if too few useful candidates are returned. If all searches return
   empty or only unrelated items, respond with an honest note — do not invent.
4. Pass through every candidate returned by search_catalog. Each candidate
   must include its item_id, retrieval_rank, and the catalog flags
   (has_bullet, has_dimensions, has_weight, has_material) you received.
5. Classify each item as exactly one of: exact_product, accessory, unrelated,
   uncertain. Covers, mats, pillows, replacement parts, and add-ons are not
   the requested primary product.
6. Call finalize_recommendations once with the full classified list. The
   application code removes ineligible products and applies deterministic
   ranking. Never add an item the finalizer didn't return.
7. After calling finalize_recommendations, your ENTIRE response MUST be a
   single JSON object — no text before, no thinking, no code fences. The
   JSON schema is specified below in "Final response".

Brief extraction rules:
- intent: one sentence in the user's voice, what they want.
- search_terms: 2-4 concrete catalog terms, most discriminating first. Terms
  MUST be drawn from the user's words; do not invent terms the user did not say.
- product_type / brand / color / material / compatibility / target_use: empty
  string when the user did not specify. Never guess.
- budget_usd: convert to USD. Record the source currency and the conversion
  rate you used in assumptions.
- max_dimension_cm: convert to centimeters. 0 means no ceiling.
- quantity: 1 when unspecified. "pair" -> 2, "dozen" -> 12. Keep
  "set"/"pack"/"bundle" as 1 unless the user said a number.
- must_have: hard constraints the user stated. Failure to meet any is blocking.
- nice_to_have: soft preferences; missing them does not block results.
- assumptions: reasoning notes (e.g. "considered 200 EUR ~ 216 USD at 1.08").
- evidence_gaps: parts of the brief that are weak or guessed.

Brief extraction examples:
- "a pair of wireless earbuds under 5k rupees"
  intent="wireless earbuds for a pair, budget around 5k rupees",
  search_terms="wireless earbuds", product_type="HEADPHONES",
  budget_usd=60.0, quantity=2,
  assumptions=["5000 INR converted to ~60 USD at 0.012"],
  evidence_gaps=["no stated brand or color; listener must accept any"].
- "noise-cancelling headphones for open-plan office"
  intent="noise-cancelling headphones for an open-plan office",
  search_terms="noise cancelling headphones", product_type="HEADPHONES",
  target_use="open-plan office",
  nice_to_have=["noise_cancelling"],
  evidence_gaps=["budget not specified; showing full price range"].
- "I want a black one"
  intent="the previously discussed product, in black",
  search_terms="<previous product terms>", color="black",
  evidence_gaps=["no product_type restated; relying on session context"].
- "around $200, maybe a bit more"
  intent="product around $200, flexible upward",
  search_terms="<from the rest of the request>", budget_usd=200.0,
  assumptions=["$200 is a target, not a hard ceiling"],
  evidence_gaps=["no hard ceiling stated"].

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
        provider: Provider name (``"deepseek"``, ``"vllm"``, etc.).
            Provider-specific request extras (e.g. DeepSeek
            ``reasoning_split``) are looked up via ``provider_extras`` and
            merged into ``default_options``. Unknown extras on a different
            provider are forwarded in ``extra_body`` and silently ignored
            by the server.
    """
    tracker = tracker or CatalogEvidenceTracker()
    provider_options = provider_extras(provider) if provider else {}
    # DeepSeek API does not accept `json_schema` response_format (400 error),
    # and `json_object` mode is incompatible with tool calling (silently
    # disables them). Omitting response_format entirely lets the model use
    # tools freely; the extended prompt instructions + json-repair safety
    # net ensure reliable JSON extraction from the final content.
    response_format = (
        None
        if provider == "deepseek"
        else RecommendationResponse
    )
    if response_format is None:
        default_options = dict(provider_options)
    else:
        default_options = {"response_format": response_format, **provider_options}
    return Agent(
        client=client,
        instructions=SHOPPING_AGENT_INSTRUCTIONS,
        tools=_build_agent_tools(catalog_tools, tracker=tracker),
        default_options=default_options,
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
    """Build the structured brief extraction tool.

    The tool's argument is the ShoppingBrief Pydantic model. MAF exposes the
    schema to the model, the agent fills in the fields via tool calling, and
    MAF passes the parsed JSON back to the tool body as a plain dict. The
    tool body re-validates the dict through ShoppingBrief so the canonical
    typed model is the single source of truth, then returns ``model_dump()``.
    """

    @tool(
        name=EXTRACT_BRIEF_TOOL,
        description=(
            "Extract a structured shopping brief from the user's request. "
            "Call this once, before any catalog search tool. The argument "
            "is the full brief; convert any budget to USD, any dimension "
            "to centimeters, and any quantity word ('pair', 'dozen') to a "
            "number. Leave fields empty when the user did not specify them; "
            "do not invent product_type, brand, color, or material. Record "
            "non-obvious reasoning in assumptions and uncertainty in "
            "evidence_gaps."
        ),
    )
    def extract_brief(
        brief: Annotated[
            dict[str, Any],
            Field(description="Structured shopping brief extracted from the user's request."),
        ],
    ) -> dict[str, Any]:
        """Validate the brief through ShoppingBrief and return the canonical dict."""
        # MAF passes the parsed JSON dict (not a Pydantic model) to the tool
        # body. Re-validate so the canonical typed model is the single source
        # of truth; reject shape/constraint violations back to the model.
        validated = ShoppingBrief.model_validate(brief)
        return validated.model_dump()

    return extract_brief


def structured_recommendation_from_response(response: Any) -> RecommendationResponse | None:
    """Read the typed recommendation from a MAF response.

    Tries ``response.value`` first (provider-native JSON-schema enforcement,
    e.g. vLLM and OpenAI). Falls back to extracting the JSON object from
    ``response.text`` and validating against the schema — for providers
    (e.g. DeepSeek) that deliver wrapped or narrated output instead.
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
