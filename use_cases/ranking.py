"""Deterministic ranking after LLM product-type eligibility screening (ABO catalog)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from domain.recommendation import FinalizedCandidate

ELIGIBLE_PRODUCT_TYPE = "exact_product"
MAX_RANKED_CANDIDATES = 8

_RELEVANCE_WEIGHT = 0.50
_BULLET_COVERAGE_WEIGHT = 0.15
_MATERIAL_WEIGHT = 0.15
_BRAND_WEIGHT = 0.10
_DIMENSION_WEIGHT = 0.10


def _vector_distance_score(candidate: dict) -> float:
    """Score 0-1 from the sqlite-vec cosine distance (lower distance = better).

    Returns 1.0 when ``candidate['distance']`` is missing (BM25-only path) so
    candidates without a vector score don't get penalized. Distance is
    in [0, 2] for unit-norm vectors with cosine; we map distance 0 → 1.0
    and distance 1.0 → 0.0 linearly so it composes with the other 0-1 signals.

    Used as a SECONDARY tie-breaker via the intent_match-style secondary sort
    key (see screen_and_rank_candidates). The primary weighted score stays
    unchanged so the BM25-first ranking contract is preserved.
    """
    dist = candidate.get("distance")
    if dist is None:
        return 1.0  # BM25 path; no penalty
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return 1.0
    # Cosine distance on unit-norm vectors lives in [0, 2]. Most useful
    # matches fall in [0, 0.5]; clamp negatives at 0 and saturate above 1.0.
    d = max(0.0, min(d, 1.0))
    return 1.0 - d


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


def _target_use_match_score(
    candidate: dict,
    target_use: str,
    must_have: list[str],
) -> float:
    """Tie-breaker score 0-1 for prose-level intent match.

    NOT a primary ranking signal. Used only to break ties between candidates
    whose weighted score is identical. Returns 1.0 when the candidate's
    title / bullets / brand contain any token from ``target_use`` or
    ``must_have``; 0.0 otherwise. Whole-word, case-insensitive.

    Args:
        candidate: The raw candidate dict (title_en, brand_en, plus the
            ``_bullet_text`` field the search tool attaches for provenance
            scoring).
        target_use: Free-text ``target_use`` from the brief (e.g. "home
            office", "commuting"). Empty string skips the check.
        must_have: Hard-constraint strings from the brief. Any token that
            appears in the candidate's title or bullet text counts as a
            hit.

    Returns:
        A score in [0.0, 1.0]. The caller scales it by a small weight so
        the primary ranking is undisturbed.
    """
    if not target_use and not must_have:
        return 0.0
    haystack_parts: list[str] = []
    title = candidate.get("title_en") or ""
    brand = candidate.get("brand_en") or ""
    bullets = candidate.get("_bullet_text") or ""
    if title:
        haystack_parts.append(str(title))
    if brand:
        haystack_parts.append(str(brand))
    if bullets:
        haystack_parts.append(str(bullets))
    haystack = " ".join(haystack_parts).lower()
    if not haystack:
        return 0.0
    tokens: list[str] = []
    if target_use:
        tokens.extend(t.lower() for t in target_use.split() if t.strip())
    for must in must_have:
        if not isinstance(must, str) or not must.strip():
            continue
        if " " in must:
            tokens.append(must.lower())
        else:
            tokens.append(must.lower())
    if not tokens:
        return 0.0
    hits = 0
    for token in tokens:
        if re.search(r"\b" + re.escape(token) + r"\b", haystack):
            hits += 1
    return hits / len(tokens)


def screen_and_rank_candidates(
    research: dict,
    *,
    allowed_item_ids: set[str] | None = None,
    target_use: str = "",
    must_have: list[str] | None = None,
    limit: int = MAX_RANKED_CANDIDATES,
) -> dict:
    """Keep LLM-classified exact products and rerank catalog evidence.

    Signals (from highest weight):
      - FTS5 retrieval rank → log-scaled relevance
      - Bullet-point coverage → weak quality signal
      - Material presence → dimension-heuristic category (furniture, decor)
      - Brand presence → name-brand confidence
      - Dimension records → weight for dimension-constrained queries

    When ``allowed_item_ids`` is provided, only candidates whose ``item_id`` is
    in the set survive; unknown or invented IDs are dropped. When the set is
    ``None`` the catalog-evidence guarantee is not available and the function
    falls back to trusting the input.

    When ``target_use`` or ``must_have`` is non-empty, a separate
    :func:`_target_use_match_score` is computed per candidate and used as
    a SECONDARY sort key (not a weighted signal). This breaks ties between
    candidates whose primary score is identical without disturbing the
    primary ranking when the intent terms are absent or irrelevant.
    """
    raw_candidates = research.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Shopping candidates must be an array")

    classifications = Counter()
    eligible: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Each shopping candidate must be an object")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Each shopping candidate needs a non-empty item_id")
        if allowed_item_ids is not None and item_id not in allowed_item_ids:
            classifications["unknown_to_catalog"] += 1
            continue
        if item_id in seen_ids:
            classifications["duplicate"] += 1
            continue
        classification = raw.get("product_type_match") or raw.get("classification")
        if not isinstance(classification, str):
            raise ValueError("Each shopping candidate needs product_type_match")
        classifications[classification] += 1
        if classification != ELIGIBLE_PRODUCT_TYPE:
            continue
        candidate = dict(raw)
        candidate["retrieval_rank"] = _positive_int(
            candidate.get("retrieval_rank"), index
        )
        eligible.append(candidate)
        seen_ids.add(item_id)

    must_have_list: list[str] = list(must_have or [])
    for candidate in eligible:
        retrieval_rank = candidate["retrieval_rank"]
        relevance = 1.0 / math.log2(retrieval_rank + 1)
        bullet = _bullet_coverage(candidate)
        material = _material_score(candidate)
        brand = _brand_score(candidate)
        dimension = _dimension_score(candidate)
        vector_distance = _vector_distance_score(candidate)
        score = (
            _RELEVANCE_WEIGHT * relevance
            + _BULLET_COVERAGE_WEIGHT * bullet
            + _MATERIAL_WEIGHT * material
            + _BRAND_WEIGHT * brand
            + _DIMENSION_WEIGHT * dimension
        )
        intent_match = _target_use_match_score(
            candidate, target_use, must_have_list
        )
        candidate["ranking_score"] = round(score, 6)
        candidate["ranking_signals"] = {
            "text_relevance": round(relevance, 6),
            "bullet_coverage": round(bullet, 6),
            "material_present": material,
            "brand_present": brand,
            "dimension_present": dimension,
            "intent_match": round(intent_match, 6),
            "vector_distance": round(vector_distance, 6),
        }

    # Primary sort: weighted score (descending). Secondary sort: intent match
    # score (descending) — only matters when two candidates tie on the
    # primary score, so it cannot override the deterministic ranking.
    # Tertiary: vector distance score (descending) — same logic, breaks ties
    # between vector-search candidates that share an intent-match score.
    eligible.sort(
        key=lambda item: (
            -item["ranking_score"],
            -item["ranking_signals"]["intent_match"],
            -item["ranking_signals"]["vector_distance"],
            item["retrieval_rank"],
            str(item.get("item_id", "")),
        )
    )
    ranked = eligible[: max(1, limit)]
    for position, candidate in enumerate(ranked, start=1):
        candidate["ranking_position"] = position

    typed_ranked = [FinalizedCandidate.model_validate(item) for item in ranked]

    normalized = dict(research)
    normalized["candidates"] = typed_ranked
    normalized["eligible_item_ids"] = [candidate.get("item_id") for candidate in ranked]
    summary = research.get("screening_summary", {})
    normalized["screening_summary"] = {
        **(summary if isinstance(summary, dict) else {}),
        "returned_for_ranking": len(raw_candidates),
        "eligible_exact_products": len(eligible),
        "excluded_from_ranking": len(raw_candidates) - len(eligible),
        "returned_to_agent": len(ranked),
        "returned_classifications": dict(sorted(classifications.items())),
    }
    return normalized


def enforce_recommendation_order(
    recommendation: dict, research: dict
) -> dict:
    """Drop unknown products and preserve deterministic eligible-product order."""
    candidates = research.get("candidates", [])
    order = {
        candidate.get("item_id"): index
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, dict) and candidate.get("item_id")
    }
    raw_ranked = recommendation.get("ranked", [])
    if not isinstance(raw_ranked, list):
        raise ValueError("Recommendation ranked must be an array")
    seen: set[str] = set()
    ranked: list[dict] = []
    for item in raw_ranked:
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            continue
        if item_id not in order or item_id in seen:
            continue
        ranked.append(dict(item))
        seen.add(item_id)
    ranked.sort(key=lambda item: order[item["item_id"]])
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    normalized = dict(recommendation)
    normalized["ranked"] = ranked
    return normalized
