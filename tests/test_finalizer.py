"""Tests for the finalizer and the schema-level intro-bullet guarantees.

Covers:

* :class:`IntroBullet` schema validation — closed (subject, claim_kind)
  enums, item_id ↔ subject coupling, dataset_disclaimer ↔ dataset_notice
  coupling, intent_match ↔ brief coupling. The model cannot emit
  catalog-absent facts (price, stock, rating, shipping, warranty,
  discount) because the schema has no slot for them.
* Dataset-notice overwrite — the finalizer's ``enforce_dataset_notice``
  returns the catalog-truth constant regardless of what the model wrote.
* Tie-breaker behavior in :func:`screen_and_rank_candidates` — candidates
  with identical primary scores are ordered by intent-match.
* Integration — :func:`enforce_finalized_recommendation` strips bullets
  that reference unknown items and appends a synthetic dataset_disclaimer
  bullet when the model omitted one.
"""
from __future__ import annotations

import pytest

from domain.recommendation import (
    FinalizedCandidate,
    IntroBullet,
    RecommendationResponse,
)
from use_cases.ranking import screen_and_rank_candidates
from use_cases.shopping_agent import (
    CATALOG_NOTICE,
    enforce_dataset_notice,
    enforce_finalized_recommendation,
)


# ─────────────────────────────────────────────────────────────────────
# IntroBullet schema — the structural guarantee
# ─────────────────────────────────────────────────────────────────────


def test_intro_bullet_accepts_item_with_item_id():
    """A bullet that references a specific ranked item carries an item_id."""
    bullet = IntroBullet(
        subject="item", claim_kind="color", item_id="A1", text="Black mesh back."
    )
    assert bullet.subject == "item"
    assert bullet.item_id == "A1"


def test_intro_bullet_accepts_brief_no_item_id():
    """A brief-scope bullet does not carry an item_id."""
    bullet = IntroBullet(
        subject="brief",
        claim_kind="intent_match",
        text="Good for open-plan office use.",
    )
    assert bullet.subject == "brief"
    assert bullet.item_id == ""


def test_intro_bullet_accepts_dataset_disclaimer_pair():
    """The dataset_disclaimer claim_kind pairs with the dataset_notice subject."""
    bullet = IntroBullet(
        subject="dataset_notice",
        claim_kind="dataset_disclaimer",
        text=CATALOG_NOTICE,
    )
    assert bullet.claim_kind == "dataset_disclaimer"


def test_intro_bullet_rejects_subject_item_without_item_id():
    """subject=item requires a non-empty item_id; the schema rejects otherwise."""
    with pytest.raises(ValueError, match="subject='item'"):
        IntroBullet(subject="item", claim_kind="color", text="Black.")


def test_intro_bullet_rejects_item_id_on_non_item_subject():
    """item_id is reserved for subject=item; the schema rejects otherwise."""
    with pytest.raises(ValueError, match="must not carry an item_id"):
        IntroBullet(
            subject="brief", claim_kind="none", item_id="A1", text="x"
        )


def test_intro_bullet_rejects_dataset_disclaimer_with_wrong_subject():
    """dataset_disclaimer is only valid with subject=dataset_notice."""
    with pytest.raises(ValueError, match="dataset_disclaimer"):
        IntroBullet(
            subject="brief", claim_kind="dataset_disclaimer", text="x"
        )


def test_intro_bullet_rejects_intent_match_with_wrong_subject():
    """intent_match is only valid with subject=brief."""
    with pytest.raises(ValueError, match="intent_match"):
        IntroBullet(
            subject="item", claim_kind="intent_match", item_id="A1", text="x"
        )


def test_intro_bullet_rejects_unknown_subject():
    """subject must be in the closed enum."""
    with pytest.raises(ValueError):
        # Cast to Any so pyright does not flag a string literal the schema
        # is supposed to reject; the runtime validation is the contract.
        IntroBullet(subject="stock", claim_kind="none", text="x")  # type: ignore[arg-type]


def test_intro_bullet_rejects_unknown_claim_kind():
    """claim_kind must be in the closed enum. 'price' / 'stock' / 'shipping'
    are not in the enum, so the schema rejects them."""
    for forbidden in ("price", "stock", "shipping", "rating", "warranty", "discount"):
        with pytest.raises(ValueError):
            IntroBullet(
                subject="item",
                claim_kind=forbidden,  # type: ignore[arg-type]
                item_id="A1",
                text="x",
            )


def test_intro_bullet_rejects_empty_text():
    """text must be at least one character — the schema enforces this so the
    renderer can rely on every bullet carrying a sentence."""
    with pytest.raises(ValueError):
        IntroBullet(subject="brief", claim_kind="none", text="")


# ─────────────────────────────────────────────────────────────────────
# RecommendationResponse — recommendation field is now a structured list
# ─────────────────────────────────────────────────────────────────────


def test_recommendation_field_accepts_legacy_string():
    """Old model outputs that still emit ``recommendation`` as a string are
    coerced into a single (brief, none) bullet so old payloads keep parsing.
    """
    rec = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "T",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": "All three are office chairs with mesh backs.",
        "assumptions": [],
        "notes": [],
    })
    assert len(rec.recommendation) == 1
    assert rec.recommendation[0].subject == "brief"
    assert rec.recommendation[0].claim_kind == "none"
    assert rec.recommendation[0].text == "All three are office chairs with mesh backs."


def test_recommendation_field_accepts_structured_list():
    """New shape: a list of dicts validated as IntroBullet objects."""
    rec = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "T",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [
            {"subject": "brief", "claim_kind": "intent_match",
             "text": "Good for open-plan office use."},
            {"subject": "item", "claim_kind": "color", "item_id": "A1",
             "text": "Black mesh back."},
            {"subject": "dataset_notice", "claim_kind": "dataset_disclaimer",
             "text": "Offline catalog snapshot."},
        ],
        "assumptions": [],
        "notes": [],
    })
    assert len(rec.recommendation) == 3
    assert rec.recommendation[0].subject == "brief"
    assert rec.recommendation[1].item_id == "A1"
    assert rec.recommendation[2].claim_kind == "dataset_disclaimer"


def test_recommendation_field_rejects_invalid_bullet_in_list():
    """If any bullet in the list is invalid, the whole response fails
    validation — the model cannot slip a bad bullet past the schema."""
    with pytest.raises(ValueError):
        RecommendationResponse.model_validate({
            "kind": "recommendations",
            "ranked": [
                {"rank": 1, "item_id": "A1", "title_en": "T",
                 "brand_en": "", "product_type": "", "product_url": ""},
            ],
            "recommendation": [
                {"subject": "item", "claim_kind": "color", "item_id": "A1",
                 "text": "Black."},
                {"subject": "item", "claim_kind": "price", "item_id": "A1",
                 "text": "$299."},  # 'price' is not in the enum
            ],
            "assumptions": [],
            "notes": [],
        })


# ─────────────────────────────────────────────────────────────────────
# enforce_dataset_notice
# ─────────────────────────────────────────────────────────────────────


def test_dataset_notice_constant_is_returned():
    """Whatever the model wrote, the guard returns the catalog-truth constant."""
    assert enforce_dataset_notice("live catalog with prices") == CATALOG_NOTICE
    assert enforce_dataset_notice("") == CATALOG_NOTICE
    assert enforce_dataset_notice(None) == CATALOG_NOTICE  # type: ignore[arg-type]


def test_catalog_notice_mentions_offline_no_prices():
    assert "offline" in CATALOG_NOTICE.lower()
    assert "no prices" in CATALOG_NOTICE.lower() or "no price" in CATALOG_NOTICE.lower()


# ─────────────────────────────────────────────────────────────────────
# Tie-breaker intent_match in screen_and_rank_candidates
# ─────────────────────────────────────────────────────────────────────


def _candidate(item_id, retrieval_rank, title, has_bullet=True, has_material=False, brand="BrandX"):
    return {
        "item_id": item_id,
        "retrieval_rank": retrieval_rank,
        "product_type_match": "exact_product",
        "has_bullet": has_bullet,
        "has_dimensions": False,
        "has_weight": False,
        "has_material": has_material,
        "brand_en": brand,
        "title_en": title,
    }


def test_intent_match_is_noop_without_target_use_or_must_have():
    """When the brief has no intent signals, ranking is unchanged."""
    research = screen_and_rank_candidates(
        {"candidates": [
            _candidate("A", 1, "Standard Office Chair", has_material=True),
            _candidate("B", 2, "Premium Office Chair", has_material=True),
        ]},
    )
    # BM25 relevance dominates: A wins (lower retrieval_rank).
    assert [c.item_id for c in research["candidates"]] == ["A", "B"]


def test_intent_match_breaks_tie_between_identical_score_candidates():
    """Two candidates with identical primary scores; the one mentioning the
    user's target_use wins via the secondary sort key."""
    # Both have identical flags and identical retrieval_rank, so the only
    # differentiator is intent_match against target_use.
    research = screen_and_rank_candidates(
        {"candidates": [
            _candidate("NO_MATCH", 1, "Office Chair Standard", has_material=True),
            _candidate("MATCHES", 1, "Office Chair for Home Office", has_material=True),
        ]},
        target_use="home office",
    )
    ids = [c.item_id for c in research["candidates"]]
    assert ids[0] == "MATCHES"
    assert ids[1] == "NO_MATCH"


def test_intent_match_does_not_override_primary_score():
    """If one candidate has a clearly higher primary score, intent_match
    on a lower-scoring candidate cannot reorder it above the winner."""
    research = screen_and_rank_candidates(
        {"candidates": [
            _candidate("LOW_PRIMARY", 1, "Living Room Chair",
                      has_bullet=True, has_material=True, brand="Premium"),
            _candidate("HIGH_PRIMARY", 5, "Living Room Chair for Home Office",
                      has_bullet=False, has_material=False, brand=None),
        ]},
        target_use="home office",
    )
    # LOW_PRIMARY has bullet+material+brand = highest primary score.
    # HIGH_PRIMARY has lower retrieval_rank + intent_match but no flags.
    # Primary wins; intent_match is a tie-breaker, not an override.
    assert research["candidates"][0].item_id == "LOW_PRIMARY"


def test_intent_match_uses_must_have_tokens():
    """must_have tokens also count toward the intent-match score."""
    research = screen_and_rank_candidates(
        {"candidates": [
            _candidate("A", 1, "Office Chair Standard", has_material=True),
            _candidate("B", 1, "Lumbar Support Office Chair", has_material=True),
        ]},
        must_have=["lumbar support"],
    )
    ids = [c.item_id for c in research["candidates"]]
    assert ids[0] == "B"


def test_intent_match_score_visible_in_signals():
    research = screen_and_rank_candidates(
        {"candidates": [
            _candidate("A", 1, "Office Chair for Home Office", has_material=True),
        ]},
        target_use="home office",
    )
    signals = research["candidates"][0].ranking_signals
    assert "intent_match" in signals
    assert signals["intent_match"] > 0.0


# ─────────────────────────────────────────────────────────────────────
# Integration: enforce_finalized_recommendation
# ─────────────────────────────────────────────────────────────────────


def _candidate_with_facts(item_id):
    return FinalizedCandidate(
        item_id=item_id,
        title_en=f"Test Product {item_id}",
        brand_en="BrandX",
        product_type="CHAIR",
        product_url="",
        retrieval_rank=1,
        ranking_score=0.5,
        ranking_signals={},
        ranking_position=1,
        has_bullet=1,
        has_dimensions=0,
        has_weight=0,
        has_material=1,
    )


def test_enforce_appends_synthetic_dataset_disclaimer_bullet():
    """If the model omitted the dataset_disclaimer bullet, the finalizer
    appends one so the user always sees the catalog-scope reminder.
    """
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test Product A1",
             "brand_en": "BrandX", "product_type": "CHAIR", "product_url": ""},
        ],
        "recommendation": [
            {"subject": "brief", "claim_kind": "intent_match",
             "text": "A good chair for the office."},
        ],
        "assumptions": [],
        "notes": [],
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    assert any(
        b.subject == "dataset_notice" and b.claim_kind == "dataset_disclaimer"
        for b in result.recommendation
    )


def test_enforce_keeps_model_dataset_disclaimer_bullet_verbatim():
    """If the model already emitted a dataset_disclaimer bullet, the
    finalizer does not duplicate it."""
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [
            {"subject": "dataset_notice", "claim_kind": "dataset_disclaimer",
             "text": "offline catalog snapshot"},
        ],
        "assumptions": [],
        "notes": [],
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    disclaimer_bullets = [
        b for b in result.recommendation
        if b.subject == "dataset_notice" and b.claim_kind == "dataset_disclaimer"
    ]
    assert len(disclaimer_bullets) == 1
    assert disclaimer_bullets[0].text == "offline catalog snapshot"


def test_enforce_strips_bullet_referencing_dropped_item():
    """A bullet with subject=item pointing at an item the finalizer dropped
    (i.e. not in the candidates list) is stripped, and a note is added so
    the user sees what happened.
    """
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test A1",
             "brand_en": "", "product_type": "", "product_url": ""},
            {"rank": 2, "item_id": "A2", "title_en": "Test A2",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [
            {"subject": "item", "claim_kind": "color", "item_id": "A2",
             "text": "Available in red."},  # A2 is unknown to the finalizer
        ],
        "assumptions": [],
        "notes": [],
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    # Bullet referencing A2 is removed.
    assert all(b.item_id != "A2" for b in result.recommendation)
    # A note explains what happened.
    assert any("A2" in n and "unknown" in n.lower() for n in result.notes)


def test_enforce_caps_bullets_at_max():
    """A response with more than MAX_INTRO_BULLETS bullets is capped by the
    finalizer. The schema-level max_length=MAX_INTRO_BULLETS catches the
    same case earlier (at model_validate time) for the typed path; this
    test exercises the finalizer's runtime cap via the dict path, which
    is the path used when the model output is untyped.
    """
    from domain.recommendation import IntroBullet, MAX_INTRO_BULLETS

    # Pre-build MAX_INTRO_BULLETS + 3 already-typed bullets. Going through
    # the dict path bypasses the schema's max_length validator (the finalizer
    # then enforces the cap).
    extra_bullets = [
        IntroBullet(subject="brief", claim_kind="none", text=f"Bullet {i}.")
        for i in range(MAX_INTRO_BULLETS + 3)
    ]
    recommendation = {
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [b.model_dump() for b in extra_bullets],
        "assumptions": [],
        "notes": [],
    }
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    assert len(result.recommendation) <= MAX_INTRO_BULLETS


def test_enforce_drops_provenance_violations_and_runs_guards():
    """The provenance drop and the safety guards both run on the same path."""
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test A1",
             "brand_en": "", "product_type": "", "product_url": ""},
            # A2 is unknown to the tracker — must be dropped.
            {"rank": 2, "item_id": "A2", "title_en": "Test A2",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [],
        "assumptions": [],
        "notes": [],
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    assert len(result.ranked) == 1
    assert result.ranked[0].item_id == "A1"
    # A disclaimer bullet was synthesized.
    assert any(
        b.subject == "dataset_notice" and b.claim_kind == "dataset_disclaimer"
        for b in result.recommendation
    )


def test_enforce_overwrites_dataset_notice():
    """The dataset_notice field on the response is overwritten with the
    catalog-truth constant. The bullet's text mirrors the same constant."""
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": [
            {"subject": "brief", "claim_kind": "none", "text": "A good chair."},
        ],
        "assumptions": [],
        "notes": [],
        "dataset_notice": "live catalog with prices and stock",
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    assert result.dataset_notice == CATALOG_NOTICE


def test_enforce_accepts_legacy_string_recommendation():
    """Old model outputs that still emit ``recommendation`` as a string are
    coerced into a single (brief, none) bullet by the schema's
    BeforeValidator. The finalizer should accept that and run the rest
    of the guards normally.
    """
    recommendation = RecommendationResponse.model_validate({
        "kind": "recommendations",
        "ranked": [
            {"rank": 1, "item_id": "A1", "title_en": "Test",
             "brand_en": "", "product_type": "", "product_url": ""},
        ],
        "recommendation": "Made of leather with a brown finish.",
        "assumptions": [],
        "notes": [],
    })
    finalized = [_candidate_with_facts("A1")]
    result = enforce_finalized_recommendation(recommendation, finalized)
    # Coerced to a single (brief, none) bullet.
    assert len(result.recommendation) == 2  # the original + the synthetic disclaimer
    assert result.recommendation[0].subject == "brief"
    assert result.recommendation[0].claim_kind == "none"
    assert "leather" in result.recommendation[0].text.lower()
