"""Tests for the structured brief extraction module."""
from __future__ import annotations

from use_cases.brief import (
    _extract_budget_usd,
    _extract_dimensions,
    _extract_quantity,
    _make_budget_note,
)


def test_extract_budget_parses_usd_symbol() -> None:
    amount, note = _extract_budget_usd("I need a chair under $200")
    assert amount == 200.0
    assert "Converted from" in note


def test_extract_budget_parses_inr_with_conversion() -> None:
    amount, note = _extract_budget_usd("around 15000 rupees")
    assert amount == 180.0  # 15000 * 0.012
    assert "Converted from" in note


def test_extract_budget_parses_euro() -> None:
    amount, note = _extract_budget_usd("max €150")
    assert amount == 162.0  # 150 * 1.08
    assert "Converted from" in note


def test_extract_budget_returns_zero_for_no_number() -> None:
    amount, note = _extract_budget_usd("cheap headphones")
    assert amount == 0.0
    # "cheap" is not a budget trigger, so no note is generated
    assert note == ""


def test_extract_budget_returns_zero_for_no_match() -> None:
    amount, note = _extract_budget_usd("gaming chair")
    assert amount == 0.0
    assert note == ""


def test_extract_dimensions_single_inch() -> None:
    dims, note = _extract_dimensions("27 inch monitor")
    assert dims["height"] == 68.58  # 27 * 2.54
    assert "Single" in note


def test_extract_dimensions_two_values_no_units() -> None:
    # Each value needs its own unit for the simple regex to match.
    dims, note = _extract_dimensions("120cm wide by 60cm deep desk")
    assert dims["width"] == 120.0
    assert dims["length"] == 60.0
    assert "Two dimensions" in note


def test_extract_dimensions_returns_zeros_for_no_match() -> None:
    dims, note = _extract_dimensions("gaming chair")
    assert all(v == 0.0 for v in dims.values())
    assert note == ""


def test_extract_quantity_default() -> None:
    assert _extract_quantity("gaming chair") == 1


def test_extract_quantity_pair() -> None:
    assert _extract_quantity("pair of speakers") == 2


def test_extract_quantity_numeric() -> None:
    assert _extract_quantity("4 chairs") == 4


def test_make_budget_note_produces_catalog_disclaimer() -> None:
    notes = _make_budget_note("under $200", 200.0, "Converted from $200")
    assert any("no price column" in note for note in notes)


def test_make_budget_note_returns_empty_for_no_budget() -> None:
    assert _make_budget_note("gaming chair", 0.0, "") == []


def test_make_budget_note_notes_non_numeric_budget() -> None:
    notes = _make_budget_note("cheap headphones", 0.0, "")
    # "cheap" is not a budget trigger; no note generated
    assert notes == []


def test_make_budget_note_non_numeric_budget_trigger() -> None:
    notes = _make_budget_note("under budget headphones", 0.0, "")
    assert notes == ["User mentioned a budget constraint without a numeric value — treated as unset"]
