"""Tests for the ranking signal added when candidates carry a sqlite-vec
cosine distance. The vector-distance signal is a SECONDARY tie-breaker
that only affects ranking when the primary weighted score is identical.

These tests do NOT require sentence-transformers or a built vec_items
table — they exercise the ranking logic directly with synthetic candidate
dicts, which is what the deterministic ranking stage sees in production.
"""
from __future__ import annotations

from use_cases.ranking import (
    _vector_distance_score,
    screen_and_rank_candidates,
)


def _candidate(item_id: str, retrieval_rank: int = 1, distance: float | None = None):
    cand: dict = {
        "item_id": item_id,
        "title_en": "",
        "brand_en": "",
        "product_type": "CHAIR",
        "product_url": f"https://x/{item_id}",
        "retrieval_rank": retrieval_rank,
        "product_type_match": "exact_product",
        "has_bullet": 0,
        "has_dimensions": 0,
        "has_weight": 0,
        "has_material": 0,
    }
    if distance is not None:
        cand["distance"] = distance
    return cand


def test_no_distance_returns_neutral():
    assert _vector_distance_score({"item_id": "A"}) == 1.0


def test_distance_zero_returns_one():
    assert _vector_distance_score({"distance": 0.0}) == 1.0


def test_distance_one_returns_zero():
    assert _vector_distance_score({"distance": 1.0}) == 0.0


def test_distance_half_returns_half():
    assert _vector_distance_score({"distance": 0.5}) == 0.5


def test_distance_clamped_above_one():
    # Cosine distance can exceed 1 for non-unit vectors; clamp.
    assert _vector_distance_score({"distance": 1.5}) == 0.0


def test_distance_clamped_below_zero():
    # Negative distances shouldn't happen but be defensive.
    assert _vector_distance_score({"distance": -0.2}) == 1.0


def test_invalid_distance_returns_neutral():
    assert _vector_distance_score({"distance": "not a number"}) == 1.0


def test_vector_signal_present_in_ranking_output():
    """The ranking_signals dict exposed for each ranked item must carry
    a vector_distance key, even for candidates that didn't come from
    the vector path (defaulting to 1.0)."""
    result = screen_and_rank_candidates(
        {"candidates": [_candidate("A", distance=0.2), _candidate("B")]},
    )
    for c in result["candidates"]:
        # FinalizedCandidate is a Pydantic model — access via attribute.
        assert "vector_distance" in c.ranking_signals
    # A (distance=0.2) should have vector_distance=0.8
    # B (no distance) should have vector_distance=1.0
    by_id = {c.item_id: c for c in result["candidates"]}
    assert by_id["A"].ranking_signals["vector_distance"] == 0.8
    assert by_id["B"].ranking_signals["vector_distance"] == 1.0


def test_vector_distance_breaks_ties_not_overrides():
    """Two candidates with identical primary score but different vector
    distances: the closer one wins (secondary tie-breaker)."""
    # Both have retrieval_rank=1 (tied on relevance), no other signals.
    a = _candidate("A", retrieval_rank=1, distance=0.1)
    b = _candidate("B", retrieval_rank=1, distance=0.4)
    result = screen_and_rank_candidates({"candidates": [a, b]})
    ids = [c.item_id for c in result["candidates"]]
    # A has distance=0.1 (vector_distance=0.9) vs B has 0.4 (vector_distance=0.6)
    # Primary score is identical; secondary tiebreaker prefers A.
    assert ids[0] == "A"
    assert ids[1] == "B"
