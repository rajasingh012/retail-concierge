"""Tests for the provider-shape coercion in ``ShoppingBrief``.

These cover the real shapes different providers emit (observed in
integration runs against MiniMax-M3) and assert that the
``BeforeValidator`` normalizers produce a canonical ``list[str]`` for
list fields and a single string for ``search_terms``.

The validators accept a wider shape than the constructor's static type,
so the tests use ``model_validate`` (the dynamic API) and avoid the
Pyright static checks fired by direct constructor kwargs.
"""
from __future__ import annotations

from domain.recommendation import ShoppingBrief


def test_list_field_accepts_real_list() -> None:
    """A normal list is preserved as-is."""
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": "x",
        "nice_to_have": ["a", "b"],
        "assumptions": ["1", "2"],
    })
    assert b.nice_to_have == ["a", "b"]
    assert b.assumptions == ["1", "2"]


def test_list_field_accepts_empty_string() -> None:
    """``""`` (an empty string for an empty list) is normalized to ``[]``."""
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": "x",
        "nice_to_have": "",
        "must_have": "",
        "assumptions": "",
        "evidence_gaps": "",
    })
    assert b.nice_to_have == []
    assert b.must_have == []
    assert b.assumptions == []
    assert b.evidence_gaps == []


def test_list_field_accepts_wrapped_dict() -> None:
    """``{"item": ["a", "b"]}`` is unwrapped to ``["a", "b"]``."""
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": "x",
        "nice_to_have": {"item": ["noise_cancelling", "over-ear"]},
    })
    assert b.nice_to_have == ["noise_cancelling", "over-ear"]


def test_list_field_accepts_none() -> None:
    """``None`` is normalized to ``[]``."""
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": "x",
        "nice_to_have": None,
    })
    assert b.nice_to_have == []


def test_search_terms_accepts_string() -> None:
    b = ShoppingBrief.model_validate({"intent": "x", "search_terms": "wireless earbuds"})
    assert b.search_terms == "wireless earbuds"


def test_search_terms_accepts_list() -> None:
    """A one-element list is joined to a single string."""
    b = ShoppingBrief.model_validate({"intent": "x", "search_terms": ["wireless earbuds"]})
    assert b.search_terms == "wireless earbuds"


def test_search_terms_accepts_multi_element_list() -> None:
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": ["noise cancelling", "headphones"],
    })
    assert b.search_terms == "noise cancelling headphones"


def test_search_terms_accepts_wrapped_dict() -> None:
    """``{"item": ["a", "b"]}`` is unwrapped and joined."""
    b = ShoppingBrief.model_validate({
        "intent": "x",
        "search_terms": {"item": ["noise cancelling", "headphones"]},
    })
    assert b.search_terms == "noise cancelling headphones"


def test_search_terms_accepts_empty_string() -> None:
    """An empty string for ``search_terms`` stays empty."""
    b = ShoppingBrief.model_validate({"intent": "x", "search_terms": ""})
    assert b.search_terms == ""


def test_normalizers_observed_in_production_run() -> None:
    """Exact shapes observed in the 5-query harness."""
    # Query 1: search_terms emitted as a list, nice_to_have as ""
    b = ShoppingBrief.model_validate({
        "intent": "wireless earbuds for a pair, budget around 5k rupees",
        "search_terms": ["wireless earbuds"],
        "product_type": "HEADPHONES",
        "budget_usd": 60.0,
        "quantity": 2,
        "nice_to_have": "",
        "assumptions": ["5000 INR converted to ~60 USD at 0.012"],
    })
    assert b.search_terms == "wireless earbuds"
    assert b.nice_to_have == []
    assert b.assumptions == ["5000 INR converted to ~60 USD at 0.012"]

    # Query 2: search_terms + list fields wrapped in {"item": [...]}
    b = ShoppingBrief.model_validate({
        "intent": "noise-cancelling headphones for an open-plan office",
        "search_terms": {"item": ["noise cancelling headphones", "headphones"]},
        "product_type": "HEADPHONES",
        "target_use": "open-plan office",
        "budget_usd": 0,
        "quantity": 1,
        "nice_to_have": {"item": ["noise_cancelling", "over-ear style preferred"]},
        "assumptions": {"item": ["budget not specified"]},
        "evidence_gaps": {"item": ["no budget stated"]},
    })
    assert b.search_terms == "noise cancelling headphones headphones"
    assert b.nice_to_have == ["noise_cancelling", "over-ear style preferred"]
    assert b.assumptions == ["budget not specified"]
    assert b.evidence_gaps == ["no budget stated"]
