"""Structured contracts for the shopping agent's outputs.

Two contracts:

* ``RecommendationResponse`` is the agent-level ``response_format``. MAF enforces
  this through provider-native JSON schema when available (OpenAI, vLLM) and
  falls back to prompt-side instructions + parser for other providers
  (MiniMax, DeepSeek, etc.). Read it from ``AgentResponse.value``.

* ``FinalizedCandidate`` is the typed return of the ``finalize_recommendations``
  tool. MAF serializes Pydantic returns through ``model_dump()`` into the
  ``function_result`` content, giving a hard contract the model cannot bypass:
  every ranked item must originate from this tool call.

Both are pure entities — no framework imports.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

MAX_RANKED_PRODUCTS = 5
MAX_REFINEMENT_CHIPS = 5


def _coerce_str_list(value: Any) -> list[str]:
    """Normalize provider-emitted list-ish shapes to ``list[str]``.

    Different providers encode the same intent multiple ways:

    * a real list: ``["wireless earbuds"]``
    * a JSON-Schema "array of strings" wrapped in ``{"item": [...]}``:
      ``{"item": ["wireless earbuds"]}`` (some providers emit this when the
      schema is described as ``{"type": "array", "items": {"type": "string"}}``
      and the model collapses it to a single-element object)
    * an empty string ``""`` (providers that default unfilled list fields to "")

    Anything unrecognized is dropped. The result is always a ``list[str]``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    if isinstance(value, dict):
        # Common wrapper: {"item": [...]}. Unwrap anything that looks like
        # a single list payload, regardless of the key name.
        for candidate in value.values():
            if isinstance(candidate, list):
                return [str(item) for item in candidate if isinstance(item, (str, int, float))]
            if isinstance(candidate, str) and candidate.strip():
                return [candidate]
        return []
    return []


def _coerce_search_terms(value: Any) -> str:
    """Normalize ``search_terms`` to a single space-joined string.

    Accepts a string, a list of strings, or a ``{"item": [...]}`` wrapper.
    Empty / None / unrecognized shapes return ``""`` so the agent loop can
    treat the absence as "no search terms — agent must derive them".
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item]
        return " ".join(parts)
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, list):
                return _coerce_search_terms(candidate)
            if isinstance(candidate, str):
                return candidate.strip()
    return ""


_LStr = Annotated[list[str], BeforeValidator(_coerce_str_list)]


class RefinementChip(BaseModel):
    """A single clickable refinement the user can send back as their next turn."""

    label: str = Field(min_length=1, description="Short user-facing label")
    instruction: str = Field(min_length=1, description="Self-contained refinement message")


class RankedItem(BaseModel):
    """One evidence-backed product in the final recommendation list."""

    rank: int = Field(ge=1, le=MAX_RANKED_PRODUCTS)
    item_id: str = Field(min_length=1)
    title_en: str = Field(min_length=1)
    brand_en: str = ""
    product_type: str = ""
    product_url: str = ""
    why_it_fits: list[str] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """The agent's final structured response after a recommendations turn.

    Used as ``Agent(..., default_options={"response_format": RecommendationResponse})``.
    """

    kind: Literal["recommendations"] = "recommendations"
    ranked: list[RankedItem] = Field(
        max_length=MAX_RANKED_PRODUCTS,
        description="Up to 5 evidence-backed products in deterministic order",
    )
    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    recommendation: str = ""
    refinement_chips: list[RefinementChip] = Field(
        default_factory=list,
        max_length=MAX_REFINEMENT_CHIPS,
    )
    dataset_notice: str = (
        "This is an offline product catalog snapshot with typed dimensions, "
        "material, color, and brand metadata but no prices, ratings, or "
        "live availability."
    )


class ShoppingBrief(BaseModel):
    """Structured shopping brief extracted from the user's request.

    Used as the ``extract_brief`` tool's argument type. The agent fills it
    in via MAF tool calling; the tool body packages it into a brief dict
    the rest of the agent loop can consume. Constraints on numeric fields
    encode the contract the LLM must respect — do not relax them without
    also relaxing the matching ``search_catalog`` semantics.

    List-shaped fields use ``BeforeValidator`` to accept the multiple
    shapes providers emit (a list, a ``{"item": [...]}`` wrapper, or an
    empty string). ``search_terms`` is normalized to a single string.
    """

    intent: str = Field(
        description=(
            "One sentence capturing what the user wants, in their voice. "
            "Example: 'noise-cancelling wireless earbuds for commuting.'"
        )
    )
    search_terms: Annotated[str, BeforeValidator(_coerce_search_terms)] = Field(
        default="",
        description=(
            "Concrete catalog terms to feed search_catalog. Use the most "
            "discriminating 2-4 words. Example: 'wireless earbuds noise cancelling'."
        ),
    )
    product_type: str = Field(
        default="",
        description=(
            "Canonical catalog product_type if the user implied one "
            "(e.g. 'HEADPHONES', 'CHAIR'). Empty when not specified."
        ),
    )
    brand: str = Field(
        default="",
        description="Stated brand. Empty when not specified or explicitly flexible.",
    )
    budget_usd: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Budget converted to USD. 0 when not specified. Non-USD currencies "
            "are converted at approximate market rates; note the source in assumptions."
        ),
    )
    max_dimension_cm: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Maximum dimension in centimeters. 0 disables the dimension filter. "
            "Used as a ceiling by search_catalog."
        ),
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="Quantity requested. 1 when unspecified.",
    )
    color: str = Field(default="", description="Stated color. Empty when not specified.")
    material: str = Field(default="", description="Stated material. Empty when not specified.")
    must_have: _LStr = Field(
        default_factory=list,
        description="Hard constraints the user stated; failure to meet any is blocking.",
    )
    nice_to_have: _LStr = Field(
        default_factory=list,
        description="Soft preferences; missing them does not block results.",
    )
    compatibility: str = Field(
        default="",
        description="Stated compatibility requirement (e.g. 'iPhone 15', 'ThinkPad T14').",
    )
    target_use: str = Field(
        default="",
        description="Where or how the product will be used (e.g. 'home office', 'commuting').",
    )
    assumptions: _LStr = Field(
        default_factory=list,
        description=(
            "Reasoning notes the agent made to fill the brief (e.g. \"2 = pair of "
            "earbuds\", \"15000 INR converted to ~180 USD at 0.012\"). Forwarded "
            "to the user as the brief's assumption section."
        ),
    )
    evidence_gaps: _LStr = Field(
        default_factory=list,
        description=(
            "Parts of the brief that are weak or guessed (e.g. 'budget was "
            "stated in INR with no clear USD reference rate')."
        ),
    )


class FinalizedCandidate(BaseModel):
    """Typed return of the ``finalize_recommendations`` tool.

    Built by ``screen_and_rank_candidates`` and returned to MAF as a Pydantic
    list. MAF serializes via ``model_dump()`` into the tool's
    ``function_result`` content; callers read it back through Pydantic
    validation.
    """

    item_id: str
    title_en: str = ""
    brand_en: str | None = ""
    product_type: str = ""
    product_url: str = ""
    retrieval_rank: int = 0
    ranking_score: float = 0.0
    ranking_signals: dict[str, float | int] = Field(default_factory=dict)
    ranking_position: int = 0
    has_bullet: int = 0
    has_dimensions: int = 0
    has_weight: int = 0
    has_material: int = 0


def extract_json_object(text: str) -> str:
    """Return the first balanced top-level JSON object from arbitrary text.

    MAF's structured-response parser requires pure JSON; many models wrap the
    payload in ````` or ```json``` fences, prefix `````` blocks, or
    narrate around the JSON. This helper strips that noise and returns a
    substring MAF can validate. Returns the original text when no JSON object
    can be located so callers can fall back to the framework's error path.

    Uses :mod:`json_repair` to recover from common LLM malformation: missing
    outer braces (e.g. ````json kind":"..." ````), unescaped quotes, trailing
    commas, partial truncation, and prose wrapping. Fence-stripping stays
    local so we don't hand ``json_repair`` a payload that has a fence glued
    to the JSON content.
    """
    import json_repair

    candidate = text.strip()
    if candidate.startswith("```"):
        # Strip any number of leading fence lines, and any trailing fence lines.
        lines = candidate.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        # Drop leading fence lines until the first non-fence line.
        while lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        candidate = "\n".join(lines).strip()
    import json
    try:
        parsed = json_repair.loads(candidate)
    except Exception:
        return text
    if isinstance(parsed, dict):
        return json.dumps(parsed)
    # If the JSON started with a key-value pair instead of `{`, the model
    # likely emitted ````json kind":"..." ``` and the fence strip ate the
    # opening brace. json_repair interpreted the lone key-value pair as an
    # array element. Re-parse with a prepended brace before falling back.
    if not candidate.startswith("{"):
        try:
            parsed = json_repair.loads("{" + candidate)
        except Exception:
            return text
        if isinstance(parsed, dict):
            return json.dumps(parsed)
    return text


__all__ = [
    "MAX_RANKED_PRODUCTS",
    "MAX_REFINEMENT_CHIPS",
    "RefinementChip",
    "RankedItem",
    "RecommendationResponse",
    "ShoppingBrief",
    "FinalizedCandidate",
    "extract_json_object",
]
