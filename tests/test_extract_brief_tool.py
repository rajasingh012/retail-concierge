"""Tests for the structured brief extraction tool.

These tests cover the Python-level contract of the rewritten ``extract_brief``
tool: the tool exists, accepts a brief dict (the form MAF passes after JSON
decoding), validates it through ``ShoppingBrief``, and returns the canonical
brief dict. They do NOT test the LLM's compliance with the schema — that
requires a live model.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.recommendation import ShoppingBrief
from use_cases.shopping_agent import (
    EXTRACT_BRIEF_TOOL,
    _build_agent_tools,
    _make_extract_brief_tool,
)
from use_cases.shopping_agent import CatalogEvidenceTracker


def test_extract_brief_tool_is_built_with_correct_name() -> None:
    """The tool is registered under the same name the agent references."""
    tool_obj = _make_extract_brief_tool()
    assert tool_obj.name == "extract_brief"
    assert tool_obj.name == EXTRACT_BRIEF_TOOL


def test_extract_brief_tool_returns_canonical_brief_dict() -> None:
    """The tool body re-validates and returns the canonical brief dict."""
    tool_obj = _make_extract_brief_tool()
    raw = {
        "intent": "wireless earbuds for a pair, budget around 5k rupees",
        "search_terms": "wireless earbuds",
        "product_type": "HEADPHONES",
        "budget_usd": 60.0,
        "quantity": 2,
        "assumptions": ["15000 INR converted to ~60 USD at 0.012"],
        "evidence_gaps": ["no stated brand or color"],
    }
    result = tool_obj(brief=raw)
    assert isinstance(result, dict)
    assert result["intent"] == "wireless earbuds for a pair, budget around 5k rupees"
    assert result["search_terms"] == "wireless earbuds"
    assert result["product_type"] == "HEADPHONES"
    assert result["budget_usd"] == 60.0
    assert result["quantity"] == 2
    assert result["assumptions"] == ["15000 INR converted to ~60 USD at 0.012"]
    assert result["evidence_gaps"] == ["no stated brand or color"]
    # Defaults applied for fields the model didn't supply.
    assert result["brand"] == ""
    assert result["max_dimension_cm"] == 0.0


def test_extract_brief_tool_accepts_partial_brief() -> None:
    """A brief with only some fields fills the rest from ShoppingBrief defaults."""
    tool_obj = _make_extract_brief_tool()
    raw = {
        "intent": "noise-cancelling headphones for an open-plan office",
        "search_terms": "noise cancelling headphones",
        "product_type": "HEADPHONES",
        "target_use": "open-plan office",
        "nice_to_have": ["noise_cancelling"],
    }
    result = tool_obj(brief=raw)
    assert result["nice_to_have"] == ["noise_cancelling"]
    assert result["target_use"] == "open-plan office"
    assert result["budget_usd"] == 0.0
    assert result["quantity"] == 1
    assert result["brand"] == ""


def test_extract_brief_tool_rejects_negative_budget() -> None:
    """The ShoppingBrief contract enforces ``budget_usd >= 0``; the tool
    body re-validates and raises for the model to see."""
    tool_obj = _make_extract_brief_tool()
    raw = {
        "intent": "bad request",
        "search_terms": "bad",
        "budget_usd": -10.0,
    }
    with pytest.raises(ValidationError):
        tool_obj(brief=raw)


def test_extract_brief_tool_rejects_zero_quantity() -> None:
    """The contract enforces ``quantity >= 1``."""
    tool_obj = _make_extract_brief_tool()
    raw = {
        "intent": "bad request",
        "search_terms": "bad",
        "quantity": 0,
    }
    with pytest.raises(ValidationError):
        tool_obj(brief=raw)


def test_extract_brief_tool_rejects_negative_dimension() -> None:
    """The contract enforces ``max_dimension_cm >= 0``."""
    tool_obj = _make_extract_brief_tool()
    raw = {
        "intent": "bad request",
        "search_terms": "bad",
        "max_dimension_cm": -5.0,
    }
    with pytest.raises(ValidationError):
        tool_obj(brief=raw)


def test_shopping_brief_schema_includes_all_fields() -> None:
    """The Pydantic schema MAF exposes to the agent lists every field."""
    field_names = set(ShoppingBrief.model_fields.keys())
    expected = {
        "intent", "search_terms", "product_type", "brand", "budget_usd",
        "max_dimension_cm", "quantity", "color", "material", "must_have",
        "nice_to_have", "compatibility", "target_use", "assumptions",
        "evidence_gaps",
    }
    assert expected.issubset(field_names)


def test_agent_tools_canonical_order_unchanged() -> None:
    """The extract_brief tool sits first in the canonical tool order."""
    tools = _build_agent_tools([], tracker=CatalogEvidenceTracker())
    assert tools[0].name == EXTRACT_BRIEF_TOOL
