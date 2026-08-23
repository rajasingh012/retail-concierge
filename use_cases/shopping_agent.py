"""Single conversational shopping agent with deterministic catalog safeguards."""
from __future__ import annotations
import json
from typing import Annotated, Any

from agent_framework import Agent, FunctionInvocationContext, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

from infrastructure.catalog_vocabulary_provider import (
    CatalogVocabularyProvider,
)
from infrastructure.chat_clients import provider_extras

from domain.recommendation import (
    FinalizedCandidate,
    IntroBullet,
    MAX_INTRO_BULLETS,
    MAX_RANKED_PRODUCTS,
    MAX_REFINEMENT_CHIPS,
    RankedItem,
    RecommendationResponse,
    RefinementChip,
    ShoppingBrief,
    _coerce_intro_bullets,
    extract_json_object,
)
from use_cases.ranking import screen_and_rank_candidates

EXTRACT_BRIEF_TOOL = "extract_brief"
FINALIZE_RECOMMENDATIONS_TOOL = "finalize_recommendations"

SHOPPING_AGENT_INSTRUCTIONS = """\
You are RetailConcierge, a conversational shopping agent over an offline
Amazon product catalog (Amazon Berkeley Objects). You clarify the user's
request when necessary, search the catalog with the tools, classify
products, and return evidence-backed recommendations.

Workflow (every turn, in this order):
1. extract_brief  — fill the ShoppingBrief fields from the user's words.
2. find_brands    — ONLY if the user named a brand. Three-tier resolution
                    against the live catalog (exact prefix -> FTS5 -> LIKE).
                    If it returns [], the catalog has no match; fall back to
                    BM25 on the title text. Never invent a brand.
3. search_catalog AND search_vector  — call BOTH, even on single-word queries
                    like "couch". BM25 catches exact keywords (model numbers,
                    brand names); vector catches synonyms and paraphrases
                    ("couch" for "sofa", "back pain chair" for "ergonomic").
                    You may call each once. If both return fewer than ~10
                    useful candidates, you may call search_catalog a second
                    time with broader terms.
4. For each candidate, classify as exactly one of: exact_product, accessory,
                    unrelated, uncertain. Covers, mats, pillows, replacement
                    parts, and add-ons are not the requested primary product.
                    IMPORTANT: set the `classification` field on each
                    candidate dict you pass to finalize_recommendations.
                    Without it, the finalizer rejects the call.
5. finalize_recommendations  — passes the brief's color/material/pattern fields
                    to the structured-filter pre-filter automatically.
6. Return the JSON response described below.

Brief extraction rules — copy user words verbatim into these fields:

  Always fill:
    intent        → one sentence in the user's voice.
    search_terms  → 2-4 concrete catalog terms. If the user's literal terms
                    would return zero BM25 hits (misspelling, paraphrased),
                    include a corrected form alongside the literal terms.
                    User wrote "ofice chair" -> ["ofice chair", "office chair"].
                    Literal first, corrected second.

  Copy the user's literal word if they said one (empty if they didn't):
    color         → "red", "black", "white", "velvet" (when used as color), etc.
    material      → "velvet", "leather", "mesh", "wood", "metal", etc.
    pattern       → "striped", "floral", "solid", "geometric", etc.
    fabric_type   → "velvet", "linen", "cotton", "polyester", etc.
    finish_type   → "matte", "gloss", "polished", "brushed", etc.
    style         → "modern", "rustic", "vintage", "minimalist", etc.

  Only fill if the user explicitly named one:
    product_type  → MUST be an exact value from the catalog vocabulary
                    (see CATALOG_PRODUCT_TYPES). Misspellings and paraphrases
                    get canonicalized to the closest match. Leave empty only
                    when no catalog category fits.
    brand         → write it as the user wrote it (case, transliteration,
                    misspelling all preserved). find_brands resolves it at
                    search time.

  Only fill if the user gave the signal:
    budget_usd    → stated budget converted to USD. 0 when unspecified.
    max_dimension_cm → dimension in centimeters. 0 disables the filter.
    quantity      → 1 by default. "pair" -> 2, "dozen" -> 12.
    target_use    → where/how the product will be used.
    must_have     → hard constraints the user stated.
    nice_to_have  → soft preferences; missing them does not block results.
    compatibility → stated device or system compatibility.
    assumptions   → reasoning notes (e.g. "200 EUR ~ 216 USD at 1.08").
    evidence_gaps → parts of the brief that are weak or guessed.

NEVER invent specifications, prices, ratings, availability, shipping, or
warranties that aren't in the catalog.

Brief extraction examples:

Input:  "office chair with lumbar support"
Brief:  intent="office chair with lumbar support",
        search_terms="office chair lumbar support",
        product_type="CHAIR",
        nice_to_have=["lumbar_support"]
        // color, material, etc. all empty — user didn't mention any

// For each candidate returned by search_catalog / search_vector:
//   - copy item_id, retrieval_rank, and catalog flags from the tool result
//   - add classification="exact_product"|"accessory"|"unrelated"|"uncertain"
// Pass the full list to finalize_recommendations in step 5.

Input:  "chair for back pain"
Brief:  intent="chair for back pain",
        search_terms="chair back pain ergonomic",
        product_type="CHAIR",
        target_use="back pain relief"

Input:  "red velvet accent chair"
Brief:  intent="red velvet accent chair",
        search_terms="accent chair",
        product_type="CHAIR",
        color="red",          // user said "red" → copy verbatim
        material="velvet"     // user said "velvet" → copy verbatim

Input:  "black leather chair"
Brief:  intent="black leather chair",
        search_terms="chair",
        color="black",
        material="leather"

Input:  "wireless earbuds under $60"
Brief:  intent="wireless earbuds under $60",
        search_terms="wireless earbuds",
        product_type="HEADPHONES",
        budget_usd=60.0,
        evidence_gaps=["no stated brand or color"]

Input:  "couch"                      // single ambiguous word, no category
Brief:  intent="couch",
        search_terms="couch sofa"
        // No product_type — let search_catalog + search_vector discover it.

Input:  "a black one"               // follow-up referencing a previous product
Brief:  intent="the previously discussed product, in black",
        search_terms="<previous product terms>",
        color="black",
        evidence_gaps=["no product_type restated; relying on session context"]

Final response: a single JSON object with EXACTLY these field names:

{
  "kind": "recommendations",
  "ranked": [{"rank": 1, "item_id": "...", "title_en": "...", "brand_en": "...",
              "product_type": "...", "product_url": "...",
              "why_it_fits": ["..."], "trade_offs": ["..."]}],
  "assumptions": ["..."],
  "notes": ["..."],
  "recommendation": [
    {"subject": "<one of: item|brief|assumptions|catalog_notice>",
     "claim_kind": "<one of: color|material|dimension|brand|product_type|intent_match|dataset_disclaimer|none>",
     "item_id": "<required when subject=item, omitted otherwise>",
     "text": "One sentence the user will read."}
  ],
  "refinement_chips": [{"label": "...", "instruction": "..."}],
  "catalog_notice": "This is an offline product catalog snapshot..."
}

The "kind" field MUST be the literal string "recommendations" — not the
user's intent, not a summary. At most 5 entries in "ranked", 5 in
"recommendation", 4 in "refinement_chips".

Per-item bullets must use the closed (subject, claim_kind) enums; "stock",
"price", "shipping", "rating", "warranty", and "discount" are NOT valid
claim_kinds — the schema rejects them. "subject=item" requires a non-empty
item_id that matches one of the ranked entries.

If you cannot find any exact_product, return an empty ranked list. Do not
invent items.
"""


def _build_agent_tools(
    catalog_tools: list[Any],
    *,
    audit_logger: Any = None,
    catalog_vocabulary: dict[str, list[str]] | None = None,
    repository: Any = None,
) -> list[Any]:
    """Compose the tool list for the shopping agent.

    Order is part of the contract: ``extract_brief`` runs first, the three
    catalog tools in between, and ``finalize_recommendations`` last so the
    model can use the evidence it observed earlier in the turn.

    ``repository`` is optional and only used by ``finalize_recommendations``
    to apply the brief's structured-filter attributes (color, material,
    etc.) as a pre-filter on the union of search_catalog + search_vector
    candidates. When omitted (e.g. tests that don't exercise the filter),
    the finalize tool skips the pre-filter step.
    """
    return [
        _make_extract_brief_tool(catalog_vocabulary=catalog_vocabulary),
        *catalog_tools,
        _make_finalize_tool(audit_logger=audit_logger, repository=repository),
    ]


def _seed_brief_validator(catalog_vocabulary: dict[str, list[str]] | None) -> None:
    """Push the catalog vocabulary into the brief Pydantic validator.

    Called once at agent build time. Empty vocabulary is a no-op so test
    fixtures (and the off-catalog fallback path) can keep using the bare
    brief without seeding.
    """
    if not catalog_vocabulary:
        return
    from domain.recommendation import set_catalog_vocabulary

    set_catalog_vocabulary(
        set(catalog_vocabulary.get("product_types") or []),
    )


def build_shopping_agent(
    client: OpenAIChatCompletionClient,
    catalog_tools: list[Any],
    *,
    provider: str = "",
    audit_logger: Any = None,
    catalog_vocabulary: dict[str, list[str]] | None = None,
    repository: Any = None,
) -> Agent:
    """Build the one MAF agent used for every turn in a shopping session.

    Args:
        client: MAF OpenAI-compatible chat client.
        catalog_tools: Search / brand / product-type tools.
        provider: Provider name (``"deepseek"``, ``"vllm"``, etc.).
            Provider-specific request extras (e.g. DeepSeek
            ``reasoning_split``) are looked up via ``provider_extras`` and
            merged into ``default_options``. Unknown extras on a different
            provider are forwarded in ``extra_body`` and silently ignored
            by the server.
        audit_logger: Optional ``AuditLogger``; finalize_recommendations
            records one entry per call with the screening outcomes.
        catalog_vocabulary: Optional ``{"product_types": [...]}``
            catalog terms. Used in two places, by two different layers:

            1. The brief-time Pydantic validator
               (``set_catalog_vocabulary``) is the GATE for product_type
               only — it rejects product_type values the model produces
               that are not in this set. Brands are NOT gated here.
            2. A ``CatalogVocabularyProvider`` registered on the agent
               via ``context_providers`` is the HINT — it surfaces the
               product_type list to the LLM on every model call so the
               model can pick from the right list in the first place.

            Brand canonicalization happens at search time via the
            ``find_brands`` tool (three-tier resolution against the live
            catalog), not at brief-extraction time.

            Both product_type layers share one vocabulary list so they
            cannot drift.
            The provider is the MAF-canonical pattern (ADR 0016 +
            ``samples/02-agents/context_providers/simple_context_provider.py``);
            we no longer bake the vocabulary into the static
            ``instructions=`` payload.
    """
    provider_options = provider_extras(provider) if provider else {}
    # Some vLLM versions (0.23) suppress tool calling when `response_format`
    # is set alongside `tools` — the model skips the tool-call loop and outputs
    # JSON directly, bypassing the catalog. Omitting `response_format` for all
    # providers lets tool calling work freely; the extended prompt instructions
    # + json-repair safety net ensure reliable JSON extraction from the final
    # content.
    default_options = dict(provider_options)
    _seed_brief_validator(catalog_vocabulary)
    return Agent(
        client=client,
        instructions=SHOPPING_AGENT_INSTRUCTIONS,
        tools=_build_agent_tools(
            catalog_tools,
            audit_logger=audit_logger,
            catalog_vocabulary=catalog_vocabulary,
            repository=repository,
        ),
        context_providers=[_build_vocabulary_provider(catalog_vocabulary)],
        default_options=default_options,
    )


def _build_vocabulary_provider(
    catalog_vocabulary: dict[str, list[str]] | None,
) -> CatalogVocabularyProvider:
    """Construct the MAF ContextProvider that injects the catalog vocab.

    Always returns a provider (an empty vocabulary still produces a
    provider that no-ops via ``body_has_content``), so the agent has a
    stable context_providers list regardless of catalog state.
    """
    product_types: list[str] = []
    if catalog_vocabulary:
        product_types = list(catalog_vocabulary.get("product_types") or [])
    return CatalogVocabularyProvider(
        product_types=product_types,
    )


def _make_finalize_tool(audit_logger: Any = None, repository: Any = None):
    @tool(
        name=FINALIZE_RECOMMENDATIONS_TOOL,
        description=(
            "Remove candidates that are not exact requested products and apply the "
            "catalog's deterministic ranking. Candidates must originate from "
            "search_catalog in this session. Returns the authoritative candidate order."
        ),
    )
    def finalize_recommendations(
        ctx: Annotated[FunctionInvocationContext, "MAF context (excluded from schema)"],
        candidates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Screen product identity and deterministically rank exact products.

        Reads the per-session ``seen_item_ids`` set written by
        ``search_catalog`` (via ``ctx.session.state``) and the brief's
        ``target_use`` / ``must_have`` written by ``extract_brief``.

        If the brief has any structured-attribute values (color, material,
        pattern, finish_type, fabric_type, style) and a repository was
        provided at agent-build time, the proposed candidates are first
        pre-filtered against ``listing_text_values`` via LIKE matching.
        Candidates that don't satisfy every specified filter are dropped
        before the ranking step. Filtered candidates are reported in the
        audit log so provenance stays intact.
        """
        state = ctx.session.state if ctx.session is not None else {}
        seen = set(state.get("seen_item_ids") or ())
        target_use = str(state.get("target_use") or "")
        must_have_raw = state.get("must_have") or []
        must_have_list: list[str] = [
            m for m in must_have_raw if isinstance(m, str) and m.strip()
        ]
        structured_filter_kwargs = {
            "color": str(state.get("target_color") or ""),
            "material": str(state.get("target_material") or ""),
            "pattern": str(state.get("target_pattern") or ""),
            "finish_type": str(state.get("target_finish_type") or ""),
            "fabric_type": str(state.get("target_fabric_type") or ""),
            "style": str(state.get("target_style") or ""),
        }
        has_structured_filter = any(
            v.strip() for v in structured_filter_kwargs.values()
        )

        proposed_item_ids = [
            str(c.get("item_id"))
            for c in candidates
            if isinstance(c, dict) and c.get("item_id")
        ]
        structured_filter_survivors: set[str] = set()
        structured_filter_blocked: list[str] = []
        if has_structured_filter and repository is not None:
            try:
                survivors = repository.apply_structured_filter(
                    proposed_item_ids, **structured_filter_kwargs
                )
            except Exception as exc:
                # Never let a structured-filter failure abort the finalizer;
                # log to audit, fall back to no filter so the user still
                # gets ranked results from BM25+vector.
                if audit_logger is not None:
                    record = getattr(audit_logger, "record", None)
                    if record is not None:
                        record(
                            FINALIZE_RECOMMENDATIONS_TOOL,
                            {
                                "proposed_item_ids": proposed_item_ids,
                                "structured_filter_kwargs": structured_filter_kwargs,
                            },
                            {"structured_filter_error": str(exc)},
                        )
                survivors = list(proposed_item_ids)
            structured_filter_survivors = set(survivors)
            structured_filter_blocked = [
                item_id
                for item_id in proposed_item_ids
                if item_id not in structured_filter_survivors
            ]
            # Apply the structured filter by narrowing the candidate dict list
            # to only those whose item_id survived. The model's proposed
            # candidates that don't match color/material/etc. are dropped.
            candidates = [
                c
                for c in candidates
                if isinstance(c, dict)
                and c.get("item_id") in structured_filter_survivors
            ]

        result = screen_and_rank_candidates(
            {"candidates": candidates},
            allowed_item_ids=seen,
            target_use=target_use,
            must_have=must_have_list,
        )
        accepted_item_ids = [c.item_id for c in result["candidates"]]
        if audit_logger is not None:
            record = getattr(audit_logger, "record", None)
            if record is not None:
                record(
                    FINALIZE_RECOMMENDATIONS_TOOL,
                    {
                        "proposed_item_ids": proposed_item_ids,
                        "target_use": target_use,
                        "must_have": must_have_list,
                        "structured_filter": (
                            structured_filter_kwargs
                            if has_structured_filter
                            else None
                        ),
                    },
                    {
                        "accepted_item_ids": accepted_item_ids,
                        "provenance_blocked": [
                            item_id
                            for item_id in proposed_item_ids
                            if item_id not in seen
                        ],
                        "structured_filter_blocked": structured_filter_blocked,
                        "result_count": len(accepted_item_ids),
                    },
                )
        # MAF serializes tool returns through a generic JSON encoder that does
        # not understand Pydantic; model_dump explicitly so the wire format is
        # a plain dict (which MAF round-trips losslessly).
        result["candidates"] = [
            candidate.model_dump() for candidate in result["candidates"]
        ]
        return result

    return finalize_recommendations


def _make_extract_brief_tool(catalog_vocabulary: dict[str, list[str]] | None = None):
    """Build the structured brief extraction tool.

    The tool's argument is the ShoppingBrief Pydantic model. MAF exposes the
    schema to the model, the agent fills in the fields via tool calling, and
    MAF passes the parsed JSON back to the tool body as a plain dict. The
    tool body re-validates the dict through ShoppingBrief so the canonical
    typed model is the single source of truth, then returns ``model_dump()``.

    When ``catalog_vocabulary`` is provided at agent build time, the brief's
    Pydantic model_validator (see ``domain.recommendation.ShoppingBrief``)
    is already seeded to reject off-vocabulary product_type / brand values.
    The tool body just re-validates and surfaces any rejection to MAF as a
    tool error, so the model gets a clean retry signal.

    The validated brief's ``target_use`` and ``must_have`` are written to
    ``ctx.session.state`` so the finalize tool can read them for the
    intent-match tie-breaker.
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
        ctx: Annotated[FunctionInvocationContext, "MAF context (excluded from schema)"],
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
        # Surface the intent-relevant fields to the finalize tool via
        # ``ctx.session.state``. Empty strings / lists mean the user did not
        # specify them; the tie-breaker becomes a no-op in that case.
        if ctx.session is not None:
            ctx.session.state["target_use"] = validated.target_use
            ctx.session.state["must_have"] = list(validated.must_have)
            # Structured-filter attributes — the finalize tool applies them
            # as a pre-filter via apply_structured_filter on the union of
            # search_catalog + search_vector candidates.
            ctx.session.state["target_color"] = validated.color
            ctx.session.state["target_material"] = validated.material
            ctx.session.state["target_pattern"] = validated.pattern
            ctx.session.state["target_finish_type"] = validated.finish_type
            ctx.session.state["target_fabric_type"] = validated.fabric_type
            ctx.session.state["target_style"] = validated.style
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


# Catalog-truth disclaimer. The finalizer overwrites whatever the model emitted
# and uses this string as the text of the synthetic dataset_disclaimer bullet
# it appends when the model omitted one.
CATALOG_NOTICE = (
    "This is an offline product catalog snapshot with typed dimensions, "
    "material, color, and brand metadata but no prices, ratings, or "
    "live availability."
)


def enforce_catalog_notice(_notice: str | None) -> str:
    """Return the catalog-truth disclaimer regardless of what the model wrote."""
    return CATALOG_NOTICE


def enforce_finalized_recommendation(
    recommendation: RecommendationResponse | dict[str, Any],
    finalized: list[FinalizedCandidate] | None,
) -> RecommendationResponse:
    """Drop unknown products, restore deterministic candidate order, apply finalizer guards.

    Accepts either a ``RecommendationResponse`` (the typed result from MAF) or a
    plain dict (back-compat for callers that haven't migrated). Returns the
    typed model.

    Guards applied here:

    * Strip any intro bullet whose ``item_id`` no longer maps to a
      surviving ranked item.
    * Guarantee the dataset_disclaimer bullet is present; synthesize one
      when the model omitted it.
    * Overwrite ``catalog_notice`` with the catalog-truth constant so the
      disclaimer cannot be paraphrased.
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

    # Pull the intro bullets the agent produced (recommendation field) and
    # the existing notes / assumptions so the guards can extend them.
    if isinstance(recommendation, RecommendationResponse):
        intro_bullets = list(recommendation.recommendation or [])
        existing_notes = list(recommendation.notes or [])
        existing_assumptions = list(recommendation.assumptions or [])
    else:
        raw = recommendation.get("recommendation", [])
        intro_bullets = _coerce_intro_bullets(raw)
        existing_notes = list(recommendation.get("notes", []) or [])
        existing_assumptions = list(recommendation.get("assumptions", []) or [])

    # Catalog-absent fact categories (stock, price, shipping, rating, warranty,
    # discount) are rejected at the IntroBullet schema level — the enums have
    # no slot for them. Here we make sure the dataset-disclaimer bullet is
    # present; if the model omitted it, append a synthetic one so the user
    # always sees the catalog-scope reminder.
    has_disclaimer = any(
        b.subject == "catalog_notice" and b.claim_kind == "dataset_disclaimer"
        for b in intro_bullets
    )
    if not has_disclaimer:
        intro_bullets.append(
            IntroBullet(
                subject="catalog_notice",
                claim_kind="dataset_disclaimer",
                text=CATALOG_NOTICE,
            )
        )

    # Strip bullets that reference an item the finalizer dropped (an item
    # in the agent's ranked list that the provenance gate above removed).
    # The IntroBullet schema does not check item_id validity against the
    # ranked list — it only checks that subject=item carries a non-empty
    # item_id — so this finalizer-level check is what keeps a bullet from
    # pointing at a phantom product.
    existing_item_ids = {candidate.item_id for candidate in finalized}
    stripped_item_bullets = [
        b for b in intro_bullets
        if b.subject == "item" and b.item_id not in existing_item_ids
    ]
    intro_bullets = [b for b in intro_bullets if b not in stripped_item_bullets]
    for stripped in stripped_item_bullets:
        note = (
            f"Removed intro bullet referencing unknown item "
            f"{stripped.item_id!r}"
        )
        if note not in existing_notes:
            existing_notes.append(note)

    # Cap to MAX_INTRO_BULLETS so a runaway model can't flood the UI.
    intro_bullets = intro_bullets[:MAX_INTRO_BULLETS]

    # Overwrite the dataset notice with the catalog-truth constant.
    notice = enforce_catalog_notice(
        recommendation.get("catalog_notice") if isinstance(recommendation, dict) else recommendation.catalog_notice
    )

    if isinstance(recommendation, RecommendationResponse):
        return recommendation.model_copy(
            update={
                "ranked": ranked,
                "recommendation": intro_bullets,
                "notes": existing_notes,
                "catalog_notice": notice,
            }
        )
    return RecommendationResponse(
        kind="recommendations",
        ranked=ranked,
        assumptions=existing_assumptions,
        notes=existing_notes,
        recommendation=intro_bullets,
        refinement_chips=_parse_refinement_chips(recommendation.get("refinement_chips", [])),
        catalog_notice=notice,
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
