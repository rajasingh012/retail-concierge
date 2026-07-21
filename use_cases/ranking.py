"""Deterministic ranking after LLM product-type eligibility screening."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

ELIGIBLE_PRODUCT_TYPE = "exact_product"
MAX_RANKED_CANDIDATES = 8

_RELEVANCE_WEIGHT = 0.55
_POPULARITY_WEIGHT = 0.25
_RATING_WEIGHT = 0.15
_BESTSELLER_WEIGHT = 0.05


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _rating_confidence(stars: float, review_count: float) -> float:
    """Combine star quality with bounded review-count confidence."""
    quality = min(max(stars / 5.0, 0.0), 1.0)
    confidence = min(math.log1p(max(review_count, 0.0)) / math.log1p(1000), 1.0)
    return quality * (0.5 + 0.5 * confidence)


def screen_and_rank_candidates(research: dict, limit: int = MAX_RANKED_CANDIDATES) -> dict:
    """Keep LLM-classified exact products and rerank their catalog evidence."""
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

    max_popularity = max(
        (_number(item.get("bought_last_month")) for item in eligible),
        default=0.0,
    )
    popularity_scale = math.log1p(max_popularity) if max_popularity > 0 else 0.0

    for candidate in eligible:
        retrieval_rank = candidate["retrieval_rank"]
        relevance = 1.0 / math.log2(retrieval_rank + 1)
        bought_last_month = max(_number(candidate.get("bought_last_month")), 0.0)
        popularity = (
            math.log1p(bought_last_month) / popularity_scale
            if popularity_scale > 0
            else 0.0
        )
        rating = _rating_confidence(
            _number(candidate.get("stars")),
            _number(candidate.get("review_count")),
        )
        bestseller = 1.0 if candidate.get("is_best_seller") is True else 0.0
        score = (
            _RELEVANCE_WEIGHT * relevance
            + _POPULARITY_WEIGHT * popularity
            + _RATING_WEIGHT * rating
            + _BESTSELLER_WEIGHT * bestseller
        )
        candidate["ranking_score"] = round(score, 6)
        candidate["ranking_signals"] = {
            "text_relevance": round(relevance, 6),
            "log_popularity": round(popularity, 6),
            "rating_confidence": round(rating, 6),
            "bestseller": bestseller,
        }

    eligible.sort(
        key=lambda item: (
            -item["ranking_score"],
            item["retrieval_rank"],
            str(item.get("asin", "")),
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
        candidate.get("asin"): index
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, dict) and candidate.get("asin")
    }
    raw_ranked = recommendation.get("ranked", [])
    if not isinstance(raw_ranked, list):
        raise ValueError("Critic ranked must be an array")
    ranked = [
        dict(item)
        for item in raw_ranked
        if isinstance(item, dict) and item.get("asin") in order
    ]
    ranked.sort(key=lambda item: order[item["asin"]])
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    normalized = dict(recommendation)
    normalized["ranked"] = ranked
    return normalized
