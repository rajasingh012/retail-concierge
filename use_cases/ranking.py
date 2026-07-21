"""Deterministic ranking after LLM product-type eligibility screening (ABO catalog)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

ELIGIBLE_PRODUCT_TYPE = "exact_product"
MAX_RANKED_CANDIDATES = 8

_RELEVANCE_WEIGHT = 0.50
_BULLET_COVERAGE_WEIGHT = 0.15
_MATERIAL_WEIGHT = 0.15
_BRAND_WEIGHT = 0.10
_DIMENSION_WEIGHT = 0.10


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _dimension_score(candidate: dict) -> float:
    """Score 0-1: 1 if the listing has any typed dimensions, 0.5 if weight only, 0 otherwise."""
    has_dim = candidate.get("has_dimensions", 0)
    has_w = candidate.get("has_weight", 0)
    if has_dim:
        return 1.0
    if has_w:
        return 0.5
    return 0.0


def _material_score(candidate: dict) -> float:
    """Score 0-1: 1 if material is recorded, 0 otherwise."""
    return 1.0 if candidate.get("has_material", 0) else 0.0


def _bullet_coverage(candidate: dict) -> float:
    """Score 0-1: log-scaled bullet_point count visibility."""
    has_bullet = candidate.get("has_bullet", 0)
    if has_bullet:
        return 0.8
    return 0.0


def _brand_score(candidate: dict) -> float:
    """Score 0-1: 1 if brand is present, 0 otherwise."""
    brand = candidate.get("brand_en")
    return 1.0 if brand and brand.strip() else 0.0


def screen_and_rank_candidates(research: dict, limit: int = MAX_RANKED_CANDIDATES) -> dict:
    """Keep LLM-classified exact products and rerank catalog evidence.

    Signals (from highest weight):
      - FTS5 retrieval rank → log-scaled relevance
      - Bullet-point coverage → weak quality signal
      - Material presence → dimension-heuristic category (furniture, decor)
      - Brand presence → name-brand confidence
      - Dimension records → weight for dimension-constrained queries
    """
    raw_candidates = research.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Research candidates must be an array")

    classifications = Counter()
    eligible: list[dict] = []
    for index, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Each research candidate must be an object")
        classification = raw.get("product_type_match")
        if not isinstance(classification, str):
            raise ValueError("Each research candidate needs product_type_match")
        classifications[classification] += 1
        if classification != ELIGIBLE_PRODUCT_TYPE:
            continue
        candidate = dict(raw)
        candidate["retrieval_rank"] = _positive_int(
            candidate.get("retrieval_rank"), index
        )
        eligible.append(candidate)

    for candidate in eligible:
        retrieval_rank = candidate["retrieval_rank"]
        relevance = 1.0 / math.log2(retrieval_rank + 1)
        bullet = _bullet_coverage(candidate)
        material = _material_score(candidate)
        brand = _brand_score(candidate)
        dimension = _dimension_score(candidate)
        score = (
            _RELEVANCE_WEIGHT * relevance
            + _BULLET_COVERAGE_WEIGHT * bullet
            + _MATERIAL_WEIGHT * material
            + _BRAND_WEIGHT * brand
            + _DIMENSION_WEIGHT * dimension
        )
        candidate["ranking_score"] = round(score, 6)
        candidate["ranking_signals"] = {
            "text_relevance": round(relevance, 6),
            "bullet_coverage": round(bullet, 6),
            "material_present": material,
            "brand_present": brand,
            "dimension_present": dimension,
        }

    eligible.sort(
        key=lambda item: (
            -item["ranking_score"],
            item["retrieval_rank"],
            str(item.get("item_id", "")),
        )
    )
    ranked = eligible[: max(1, limit)]
    for position, candidate in enumerate(ranked, start=1):
        candidate["ranking_position"] = position

    normalized = dict(research)
    normalized["candidates"] = ranked
    summary = research.get("screening_summary", {})
    normalized["screening_summary"] = {
        **(summary if isinstance(summary, dict) else {}),
        "returned_for_ranking": len(raw_candidates),
        "eligible_exact_products": len(eligible),
        "excluded_from_ranking": len(raw_candidates) - len(eligible),
        "returned_to_critic": len(ranked),
        "returned_classifications": dict(sorted(classifications.items())),
    }
    return normalized


def enforce_recommendation_order(recommendation: dict, research: dict) -> dict:
    """Drop unknown products and preserve deterministic eligible-product order."""
    candidates = research.get("candidates", [])
    order = {
        candidate.get("item_id"): index
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, dict) and candidate.get("item_id")
    }
    raw_ranked = recommendation.get("ranked", [])
    if not isinstance(raw_ranked, list):
        raise ValueError("Critic ranked must be an array")
    ranked = [
        dict(item)
        for item in raw_ranked
        if isinstance(item, dict) and item.get("item_id") in order
    ]
    ranked.sort(key=lambda item: order[item["item_id"]])
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    normalized = dict(recommendation)
    normalized["ranked"] = ranked
    return normalized
