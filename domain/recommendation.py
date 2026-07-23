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

from typing import Literal

from pydantic import BaseModel, Field

MAX_RANKED_PRODUCTS = 5
MAX_REFINEMENT_CHIPS = 4


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
    substring MAF can validate. Returns the original text when no `{` is
    present so callers can fall back to the framework's error path.
    """
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
    for start, ch in enumerate(candidate):
        if ch != "{":
            continue
        depth = 0
        for end in range(start, len(candidate)):
            c = candidate[end]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start : end + 1]
    return text


__all__ = [
    "MAX_RANKED_PRODUCTS",
    "MAX_REFINEMENT_CHIPS",
    "RefinementChip",
    "RankedItem",
    "RecommendationResponse",
    "FinalizedCandidate",
    "extract_json_object",
]