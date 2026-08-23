"""Tests for the structured-attribute pre-filter helper.

Covers the LIKE-matching pre-filter against ``listing_text_values`` that
runs in ``finalize_recommendations`` when the brief has color, material,
pattern, finish_type, fabric_type, or style set.
"""
from __future__ import annotations

import sqlite3

import pytest

from infrastructure.database import migrate
from infrastructure.structured_filter import (
    STRUCTURED_FILTER_ATTRIBUTES,
    apply_structured_filter,
)


@pytest.fixture
def catalog_db():
    """In-memory catalog with a handful of listings + attribute rows.

    Layout:

      id A (CHAIR, "Red Mesh Office Chair"): color=Red, material=Mesh
      id B (CHAIR, "Velvet Loveseat Sofa"): color=Burgundy, material=Velvet
      id C (CHAIR, "Black Leather Recliner"): color=Black, material=Leather
      id D (CHAIR, "Plain Wood Chair"): no color, material=Wood

    ``url_active = 1`` on all so the filter doesn't care.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE listings (
            id INTEGER PRIMARY KEY,
            item_id TEXT NOT NULL,
            product_type TEXT,
            title_en TEXT NOT NULL,
            url_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE listing_text_values (
            id INTEGER PRIMARY KEY,
            listing_id INTEGER NOT NULL REFERENCES listings(id),
            item_id TEXT NOT NULL,
            attribute TEXT NOT NULL,
            value TEXT NOT NULL
        );
        CREATE INDEX idx_text_values_attr ON listing_text_values(attribute, value);
        """
    )
    listings = [
        (1, "A", "CHAIR", "Red Mesh Office Chair"),
        (2, "B", "CHAIR", "Velvet Loveseat Sofa"),
        (3, "C", "CHAIR", "Black Leather Recliner"),
        (4, "D", "CHAIR", "Plain Wood Chair"),
    ]
    conn.executemany(
        "INSERT INTO listings(id, item_id, product_type, title_en) VALUES (?, ?, ?, ?)",
        listings,
    )
    text_values = [
        (1, "A", "color", "Red"),
        (1, "A", "material", "Mesh"),
        (2, "B", "color", "Burgundy"),
        (2, "B", "material", "Velvet"),
        (2, "B", "style", "Modern"),
        (3, "C", "color", "Black"),
        (3, "C", "material", "Leather"),
        (4, "D", "material", "Wood"),
    ]
    conn.executemany(
        "INSERT INTO listing_text_values(listing_id, item_id, attribute, value) "
        "VALUES (?, ?, ?, ?)",
        text_values,
    )
    conn.commit()
    return conn


def test_no_filters_returns_all_candidates(catalog_db):
    out = apply_structured_filter(catalog_db, ["A", "B", "C", "D"])
    assert sorted(out) == ["A", "B", "C", "D"]


def test_empty_candidates_returns_empty(catalog_db):
    assert apply_structured_filter(catalog_db, []) == []
    assert apply_structured_filter(catalog_db, None or []) == []  # type: ignore[arg-type]


def test_color_substring_match(catalog_db):
    # "burgundy" matches listing B's "Burgundy" value.
    out = apply_structured_filter(catalog_db, ["A", "B", "C", "D"], color="burgundy")
    assert out == ["B"]


def test_color_case_insensitive(catalog_db):
    # User types lowercase "red" → matches "Red".
    out = apply_structured_filter(catalog_db, ["A", "B", "C", "D"], color="red")
    assert out == ["A"]


def test_color_substring_matches_multiple(catalog_db):
    # "B" alone would also match "Burgundy" via LIKE %B% — confirm that's
    # how substring matching works (catalog IS the synonym dictionary).
    out = apply_structured_filter(catalog_db, ["A", "B", "C", "D"], color="b")
    assert out == ["B", "C"]


def test_material_filter(catalog_db):
    out = apply_structured_filter(catalog_db, ["A", "B", "C", "D"], material="velvet")
    assert out == ["B"]


def test_color_and_material_intersection(catalog_db):
    # No listing has color=red AND material=velvet → empty.
    out = apply_structured_filter(
        catalog_db, ["A", "B", "C", "D"], color="red", material="velvet"
    )
    assert out == []


def test_color_material_combination_that_matches(catalog_db):
    # Listing B has color=Burgundy and material=Velvet.
    out = apply_structured_filter(
        catalog_db,
        ["A", "B", "C", "D"],
        color="burgundy",
        material="velvet",
    )
    assert out == ["B"]


def test_filter_only_checks_candidates(catalog_db):
    # "Velvet" exists in listing_text_values for B, but if we only pass A,
    # C, D as candidates, B shouldn't suddenly appear.
    out = apply_structured_filter(
        catalog_db, ["A", "C", "D"], material="velvet"
    )
    assert out == []


def test_empty_string_filter_is_no_op(catalog_db):
    # Empty strings mean "don't filter on this attribute".
    out = apply_structured_filter(
        catalog_db, ["A", "B", "C", "D"], color="", material="   "
    )
    # Whitespace-only treated as empty.
    assert sorted(out) == ["A", "B", "C", "D"]


def test_attribute_whitelist_enforced(catalog_db):
    # An invalid attribute name should be rejected before any SQL runs.
    with pytest.raises(ValueError, match="unknown structured-filter attribute"):
        # Bypass the keyword-only API by calling the module function with a
        # direct attribute path. Easier: simulate via __dict__ since the
        # public API is keyword-only.
        from infrastructure.structured_filter import _valid_attribute

        _valid_attribute("not_a_real_attribute")


def test_unknown_attribute_via_public_api(catalog_db):
    # Public API takes keyword args; the only way to exercise the whitelist
    # is via the helper directly. (Public kwargs don't expose unknown names.)
    from infrastructure.structured_filter import _valid_attribute

    for attr in STRUCTURED_FILTER_ATTRIBUTES:
        # Should not raise.
        assert _valid_attribute(attr) == attr


def test_whitespace_filter_term(catalog_db):
    # Filter terms are stripped; leading/trailing whitespace doesn't matter.
    out = apply_structured_filter(
        catalog_db, ["A", "B", "C", "D"], material="  velvet  "
    )
    assert out == ["B"]
