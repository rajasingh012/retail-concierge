"""Tests for the brief-time catalog-vocabulary gate.

``ShoppingBrief.model_validator`` rejects off-vocabulary product_type
values once the catalog vocabulary has been seeded via
``set_catalog_vocabulary``. Empty values bypass the gate; case
variations fold to canonical.

Brands are NOT gated at this layer (canonicalization happens at search
time via ``find_brands``). The brand field accepts any string, and
off-vocabulary brand names flow through to the search tool where
FTS5 + LIKE resolution runs against the live catalog.

The product_type validator is opt-in: without seeding, brief
construction is permissive (preserves the pre-vocab behavior so
existing callers / tests do not break).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.recommendation import (
    ShoppingBrief,
    set_catalog_vocabulary,
)


@pytest.fixture(autouse=True)
def _reset_vocab() -> None:
    """Each test gets a clean vocab — auto-reset via a fresh context."""
    set_catalog_vocabulary(set())
    yield
    set_catalog_vocabulary(set())


def _valid_brief(**overrides) -> dict:
    payload = {
        "intent": "noise-cancelling headphones for travel",
        "search_terms": "noise cancelling headphones",
        "product_type": "HEADPHONES",
        "brand": "Sony",
        "budget_usd": 200.0,
        "max_dimension_cm": 0.0,
        "quantity": 1,
        "color": "",
        "material": "",
        "must_have": [],
        "nice_to_have": ["noise_cancelling"],
        "compatibility": "",
        "target_use": "commuting",
        "assumptions": [],
        "evidence_gaps": [],
    }
    payload.update(overrides)
    return payload


def test_validator_is_noop_without_vocab() -> None:
    """No vocab seeded -> brief accepts any string (back-compat).

    This covers both product_type and brand: with no vocab, neither is
    gated. Brand gate is also disabled when vocab is seeded (deferred
    to find_brands).
    """
    brief = ShoppingBrief.model_validate(_valid_brief(product_type="NotARealType", brand="ImaginaryBrand"))
    assert brief.product_type == "NotARealType"
    assert brief.brand == "ImaginaryBrand"


def test_validator_accepts_exact_catalog_match() -> None:
    set_catalog_vocabulary({"HEADPHONES", "CHAIR"})
    brief = ShoppingBrief.model_validate(_valid_brief())
    assert brief.product_type == "HEADPHONES"
    assert brief.brand == "Sony"


def test_validator_folds_product_type_case_to_canonical() -> None:
    set_catalog_vocabulary({"HEADPHONES"})
    brief = ShoppingBrief.model_validate(
        _valid_brief(product_type="headphones", brand="logitech")
    )
    assert brief.product_type == "HEADPHONES"
    # Brand is not normalized — it stays as the LLM wrote it. The
    # search-time find_brands tool resolves against the catalog.
    assert brief.brand == "logitech"


def test_validator_rejects_unknown_product_type() -> None:
    set_catalog_vocabulary({"HEADPHONES", "CHAIR"})
    with pytest.raises(ValidationError) as exc:
        ShoppingBrief.model_validate(_valid_brief(product_type="WIDGETRON_9000"))
    assert "WIDGETRON_9000" in str(exc.value)


def test_validator_accepts_any_brand() -> None:
    """Brands are NOT gated. Any string passes through unchanged so
    the search-time find_brands tool can resolve them against the
    live catalog (FTS5 + LIKE fallback)."""
    set_catalog_vocabulary({"HEADPHONES"})
    brief = ShoppingBrief.model_validate(_valid_brief(brand="NotARealBrand"))
    assert brief.brand == "NotARealBrand"


def test_validator_accepts_arbitrary_brand_with_empty_vocab() -> None:
    """No vocab -> brand is also permissive (consistent with product_type)."""
    brief = ShoppingBrief.model_validate(_valid_brief(brand="MadeUpBrand123"))
    assert brief.brand == "MadeUpBrand123"


def test_validator_allows_empty_product_type_and_brand() -> None:
    """Empty values are valid (user did not specify)."""
    set_catalog_vocabulary({"HEADPHONES"})
    brief = ShoppingBrief.model_validate(
        _valid_brief(product_type="", brand="")
    )
    assert brief.product_type == ""
    assert brief.brand == ""


def test_validator_reseeding_changes_behavior() -> None:
    set_catalog_vocabulary(set())
    ShoppingBrief.model_validate(_valid_brief(product_type="X"))
    set_catalog_vocabulary({"HEADPHONES"})
    with pytest.raises(ValidationError):
        ShoppingBrief.model_validate(_valid_brief(product_type="X"))


def test_validator_does_not_normalize_partial_case_substrings() -> None:
    """"headphone" is not "HEADPHONES" — strict membership, not substring."""
    set_catalog_vocabulary({"HEADPHONES"})
    with pytest.raises(ValidationError):
        ShoppingBrief.model_validate(_valid_brief(product_type="headphone"))


def test_seed_skips_whitespace_only_terms() -> None:
    """Whitespace-only entries from a sloppy loader are filtered."""
    set_catalog_vocabulary({"HEADPHONES", "  ", ""})
    # only HEADPHONES survives; the brief must still accept it
    brief = ShoppingBrief.model_validate(_valid_brief(product_type="HEADPHONES", brand="Logitech"))
    assert brief.product_type == "HEADPHONES"
    assert brief.brand == "Logitech"
