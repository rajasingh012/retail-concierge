"""Structured contracts for the shopping agent's outputs.

Two contracts:

* ``RecommendationResponse`` is the agent-level ``response_format``. MAF enforces
  this through provider-native JSON schema when available (OpenAI, vLLM) and
  falls back to prompt-side instructions + parser for other providers
  (DeepSeek, etc.). Read it from ``AgentResponse.value``.

* ``FinalizedCandidate`` is the typed return of the ``finalize_recommendations``
  tool. MAF serializes Pydantic returns through ``model_dump()`` into the
  ``function_result`` content, giving a hard contract the model cannot bypass:
  every ranked item must originate from this tool call.

Both are pure entities — no framework imports.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

MAX_RANKED_PRODUCTS = 5
MAX_REFINEMENT_CHIPS = 5
MAX_INTRO_BULLETS = 5

INTRO_SUBJECTS = (
    "item",          # This bullet refers to one specific ranked item. item_id is required.
    "brief",         # This bullet refers to the user's stated intent.
    "assumptions",   # This bullet surfaces a brief assumption.
    "dataset_notice",  # The catalog-scope disclaimer line.
)

INTRO_CLAIM_KINDS = (
    "color",         # Bullet mentions a color value.
    "material",      # Bullet mentions a material value.
    "dimension",     # Bullet mentions a dimension value.
    "brand",         # Bullet mentions a brand.
    "product_type",  # Bullet mentions a product category.
    "intent_match",  # Bullet states why a product fits the user's target_use / must_have.
    "dataset_disclaimer",  # Bullet is the catalog-scope disclaimer.
    "none",          # Bullet makes no catalog claim (transitions, framing, prose-only).
)

# Cached catalog vocabularies populated lazily. The brief validator uses
# these to gate the LLM-resolved product_type / brand against the catalog
# so misspelled / hallucinated mappings are rejected deterministically
# instead of silently passed through to search_catalog.
_VOCAB: dict[str, set[str]] = {}


def set_catalog_vocabulary(product_types: set[str], brands: set[str]) -> None:
    """Seed the brief validator with the catalog's known vocabulary."""
    _VOCAB["product_types"] = {t.strip() for t in product_types if t.strip()}
    _VOCAB["brands"] = {b.strip() for b in brands if b.strip()}


def _vocab(field: str) -> set[str]:
    return _VOCAB.get(field, set())


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


class IntroBullet(BaseModel):
    """One sentence in the agent's introduction prose.

    Each bullet carries a closed ``subject`` enum (item / brief / assumptions /
    dataset_notice) and a closed ``claim_kind`` enum (color / material /
    dimension / brand / product_type / intent_match / dataset_disclaimer /
    none). Pydantic rejects any value outside the enum, so a model that wants
    to write "in stock" must pick a ``claim_kind``, and no ``claim_kind`` in
    the enum corresponds to a catalog-absent fact — the schema has no slot
    for "stock" / "price" / "shipping" / "rating" / "warranty" / "discount".

    The ``text`` field is free-form natural language; the category of claim
    the sentence makes is locked at the schema level.

    ``item_id`` is required when ``subject == "item"`` and forbidden
    otherwise. ``dataset_disclaimer`` claim_kind is only valid with
    ``dataset_notice`` subject. ``intent_match`` claim_kind is only valid with
    ``brief`` subject. These cross-field rules are enforced by
    :meth:`_validate_subject_claim_combo`.
    """

    subject: Literal["item", "brief", "assumptions", "dataset_notice"]
    claim_kind: Literal[
        "color",
        "material",
        "dimension",
        "brand",
        "product_type",
        "intent_match",
        "dataset_disclaimer",
        "none",
    ]
    item_id: str = ""
    text: str = Field(min_length=1, description="The sentence the user reads.")

    @model_validator(mode="after")
    def _validate_subject_claim_combo(self) -> "IntroBullet":
        if self.subject == "item" and not self.item_id.strip():
            raise ValueError(
                "IntroBullet with subject='item' must carry an item_id "
                "pointing at one of the ranked items"
            )
        if self.subject != "item" and self.item_id.strip():
            raise ValueError(
                f"IntroBullet with subject={self.subject!r} must not carry "
                "an item_id; item_id is reserved for subject='item'"
            )
        if self.claim_kind == "dataset_disclaimer" and self.subject != "dataset_notice":
            raise ValueError(
                "IntroBullet with claim_kind='dataset_disclaimer' must use "
                "subject='dataset_notice'"
            )
        if self.claim_kind == "intent_match" and self.subject != "brief":
            raise ValueError(
                "IntroBullet with claim_kind='intent_match' must use "
                "subject='brief'"
            )
        return self


def _coerce_intro_bullets(value: Any) -> list[IntroBullet]:
    """Coerce ``recommendation`` payloads into ``list[IntroBullet]``.

    Accepts:

    * a real list of dicts that match :class:`IntroBullet`
    * a single ``IntroBullet``-shaped dict (wrapped to a one-element list)
    * a plain ``str`` (legacy shape from older model outputs) — wrapped to a
      single ``IntroBullet(subject="brief", claim_kind="none", text=...)``
      so old model outputs still parse
    * an empty string — returns ``[]``

    Anything else returns ``[]`` so the agent loop can proceed; the
    renderer treats an empty list as "no intro" the same as an absent
    field.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [IntroBullet(subject="brief", claim_kind="none", text=text)]
    if isinstance(value, dict):
        return [IntroBullet.model_validate(value)]
    if isinstance(value, list):
        bullets: list[IntroBullet] = []
        for item in value:
            if isinstance(item, IntroBullet):
                bullets.append(item)
            elif isinstance(item, dict):
                bullets.append(IntroBullet.model_validate(item))
            elif isinstance(item, str) and item.strip():
                return [IntroBullet(subject="brief", claim_kind="none", text=item.strip())]
        return bullets
    return []


_IntroBullets = Annotated[list[IntroBullet], BeforeValidator(_coerce_intro_bullets)]


class RankedItem(BaseModel):
    """One evidence-backed product in the final recommendation list."""

    rank: int = Field(ge=1, le=MAX_RANKED_PRODUCTS)
    item_id: str = Field(min_length=1)
    title_en: str = ""
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
    recommendation: _IntroBullets = Field(
        default_factory=list,
        max_length=MAX_INTRO_BULLETS,
        description=(
            "Structured intro bullets. Each bullet carries a closed "
            "(subject, claim_kind) pair so the model cannot emit a "
            "catalog-absent fact (price, rating, stock, shipping, "
            "warranty, discount). Old string-shaped payloads are coerced "
            "to a single (brief, none) bullet for back-compat."
        ),
    )
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

    @model_validator(mode="after")
    def _gate_against_catalog_vocabulary(self) -> "ShoppingBrief":
        """Reject product_type / brand values not in the seeded catalog vocabulary.

        ``extract_brief`` runs once per turn and is the single point where the
        LLM maps the user's words to canonical catalog values. Misspellings,
        foreign-language input ("chaise de bureau"), and paraphrases
        ("executive seating") all funnel through this validator.

        Gating here means the LLM cannot silently pass a wrong
        product_type / brand into search_catalog and silently get an empty
        result set. The validator rejects with a clear error message and
        MAF returns the error to the model, which then retries with the
        correct canonical value (or omits the field, falling back to
        catalog-search-time discovery).

        Empty values bypass the gate — "not specified" is a valid brief
        state. Off-vocab values the LLM wrote deliberately (e.g.
        user-restated brand that genuinely is not in the catalog)
        are rejected so the human-visible evidence_gaps list stays honest
        about why the search returned empty.
        """
        product_type = (self.product_type or "").strip()
        brand = (self.brand or "").strip()
        if product_type:
            catalog = _vocab("product_types")
            if catalog and product_type not in catalog:
                # Be lenient on case so we don't false-reject valid mappings
                # the model lowercased / uppercased.
                match = next(
                    (t for t in catalog if t.lower() == product_type.lower()),
                    None,
                )
                if match is None:
                    raise ValueError(
                        f"product_type {product_type!r} is not in the catalog "
                        "vocabulary; leave empty or pick an exact catalog value"
                    )
                # Normalize case to the catalog's canonical form so the
                # downstream search gets a stable hit.
                self.product_type = match
        if brand:
            catalog = _vocab("brands")
            if catalog and brand not in catalog:
                match = next(
                    (b for b in catalog if b.lower() == brand.lower()),
                    None,
                )
                if match is None:
                    raise ValueError(
                        f"brand {brand!r} is not in the catalog vocabulary; "
                        "leave empty or pick an exact catalog value"
                    )
                self.brand = match
        return self


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
    "MAX_INTRO_BULLETS",
    "INTRO_SUBJECTS",
    "INTRO_CLAIM_KINDS",
    "IntroBullet",
    "_coerce_intro_bullets",
    "RefinementChip",
    "RankedItem",
    "RecommendationResponse",
    "ShoppingBrief",
    "FinalizedCandidate",
    "extract_json_object",
]
