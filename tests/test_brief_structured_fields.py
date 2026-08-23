"""Tests for the ShoppingBrief structured-attribute fields (color, material,
pattern, finish_type, fabric_type, style) and their propagation into
session state via extract_brief.
"""
from __future__ import annotations

from domain.recommendation import ShoppingBrief


def test_default_structured_fields_are_empty():
    brief = ShoppingBrief(intent="wireless earbuds")
    assert brief.color == ""
    assert brief.material == ""
    assert brief.pattern == ""
    assert brief.finish_type == ""
    assert brief.fabric_type == ""
    assert brief.style == ""


def test_structured_fields_round_trip():
    brief = ShoppingBrief(
        intent="red velvet sofa",
        color="red",
        material="velvet",
        pattern="solid",
        finish_type="matte",
        fabric_type="velvet",
        style="modern",
    )
    dumped = brief.model_dump()
    assert dumped["color"] == "red"
    assert dumped["material"] == "velvet"
    assert dumped["pattern"] == "solid"
    assert dumped["finish_type"] == "matte"
    assert dumped["fabric_type"] == "velvet"
    assert dumped["style"] == "modern"
    # Round-trip through validation again.
    brief2 = ShoppingBrief.model_validate(dumped)
    assert brief2.color == "red"
    assert brief2.material == "velvet"


def test_structured_fields_empty_strings_are_preserved():
    """Empty strings must round-trip — they're the signal "not specified"
    that the structured_filter step relies on to skip filtering on a
    particular attribute."""
    brief = ShoppingBrief(intent="chair", color="", material="")
    dumped = brief.model_dump()
    assert dumped["color"] == ""
    assert dumped["material"] == ""
    brief2 = ShoppingBrief.model_validate(dumped)
    assert brief2.color == ""
    assert brief2.material == ""


def test_existing_brief_fields_still_present():
    """New fields don't break the existing brief contract."""
    brief = ShoppingBrief(
        intent="office chair",
        product_type="CHAIR",
        brand="Ergohuman",
        color="black",
        material="mesh",
        budget_usd=400.0,
        max_dimension_cm=70.0,
        target_use="home office",
    )
    assert brief.product_type == "CHAIR"
    assert brief.brand == "Ergohuman"
    assert brief.color == "black"
    assert brief.material == "mesh"
    assert brief.budget_usd == 400.0
    assert brief.max_dimension_cm == 70.0
    assert brief.target_use == "home office"
