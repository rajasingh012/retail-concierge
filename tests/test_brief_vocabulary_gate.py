"""Tests for the brief-time catalog-vocabulary gate.

``ShoppingBrief.model_validator`` rejects off-vocabulary product_type and
brand values once the catalog vocabulary has been seeded via
``set_catalog_vocabulary``. Empty values bypass the gate; case variations
fold to canonical.

The validator is opt-in: without seeding, brief construction is permissive
(preserves the pre-vocab behavior so existing callers / tests do not break).
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
    set_catalog_vocabulary(set(), set())
    yield
    set_catalog_vocabulary(set(), set())


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
    """No vocab seeded -> brief accepts any string (back-compat)."""
    brief = ShoppingBrief.model_validate(_valid_brief(product_type="NotARealType", brand="ImaginaryBrand"))
    assert brief.product_type == "NotARealType"
    assert brief.brand == "ImaginaryBrand"


def test_validator_accepts_exact_catalog_match() -> None:
    set_catalog_vocabulary({"HEADPHONES", "CHAIR"}, {"Sony", "Logitech"})
    brief = ShoppingBrief.model_validate(_valid_brief())
    assert brief.product_type == "HEADPHONES"
    assert brief.brand == "Sony"


def test_validator_folds_case_to_catalog_canonical() -> None:
    set_catalog_vocabulary({"HEADPHONES"}, {"Logitech"})
    brief = ShoppingBrief.model_validate(
        _valid_brief(product_type="headphones", brand="logitech")
    )
    assert brief.product_type == "HEADPHONES"
    assert brief.brand == "Logitech"


def test_validator_rejects_unknown_product_type() -> None:
    set_catalog_vocabulary({"HEADPHONES", "CHAIR"}, set())
    with pytest.raises(ValidationError) as exc:
        ShoppingBrief.model_validate(_valid_brief(product_type="WIDGETRON_9000"))
    assert "WIDGETRON_9000" in str(exc.value)


def test_validator_rejects_unknown_brand() -> None:
    set_catalog_vocabulary(set(), {"Sony", "Logitech"})
    with pytest.raises(ValidationError) as exc:
        ShoppingBrief.model_validate(_valid_brief(brand="NotARealBrand"))
    assert "NotARealBrand" in str(exc.value)


def test_validator_allows_empty_product_type_and_brand() -> None:
    """Off-vocab empty values are valid (user did not specify)."""
    set_catalog_vocabulary({"HEADPHONES"}, {"Sony"})
    brief = ShoppingBrief.model_validate(
        _valid_brief(product_type="", brand="")
    )
    assert brief.product_type == ""
    assert brief.brand == ""


def test_validator_reseeding_changes_behavior() -> None:
    set_catalog_vocabulary(set(), set())
    ShoppingBrief.model_validate(_valid_brief(product_type="X"))
    set_catalog_vocabulary({"HEADPHONES"}, set())
    with pytest.raises(ValidationError):
        ShoppingBrief.model_validate(_valid_brief(product_type="X"))


def test_validator_does_not_normalize_partial_case_substrings() -> None:
    """"headphone" is not "HEADPHONES" — strict membership, not substring."""
    set_catalog_vocabulary({"HEADPHONES"}, set())
    with pytest.raises(ValidationError):
        ShoppingBrief.model_validate(_valid_brief(product_type="headphone"))


def test_seed_skips_whitespace_only_terms() -> None:
    """Whitespace-only entries from a sloppy loader are filtered."""
    set_catalog_vocabulary({"HEADPHONES", "  ", ""}, {"  ", "Logitech", ""})
    # only HEADPHONES in product_types; only Logitech in brands
    brief = ShoppingBrief.model_validate(_valid_brief(product_type="HEADPHONES", brand="Logitech"))
    assert brief.product_type == "HEADPHONES"
    assert brief.brand == "Logitech"
