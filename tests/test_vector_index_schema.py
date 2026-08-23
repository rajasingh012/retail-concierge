"""Tests for the database helpers supporting the sqlite-vec vector index.

Exercises:
- load_sqlite_vec: extension loads on a fresh connection
- vec_items_count: 0 on missing table, N after build
- vec_index_meta_insert / get: round-trip
- VECTOR_SCHEMA: creates the vec_items and vec_index_meta tables
- build_embedding_text: title + brand concatenation

Skipped when sqlite-vec is not installed in the active environment
(so CI without the dependency doesn't fail this test).
"""
from __future__ import annotations

import sqlite3

import pytest

from infrastructure.database import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    STRUCTURED_FILTER_ATTRIBUTES,
    VECTOR_SCHEMA,
    build_embedding_text,
    load_sqlite_vec,
    vec_index_meta_get,
    vec_index_meta_insert,
    vec_items_count,
)


def _sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason="sqlite-vec not installed",
)


def test_build_embedding_text_concatenates_title_and_brand():
    assert build_embedding_text("Red Mesh Chair", "Ergohuman") == "red mesh chair. ergohuman"


def test_build_embedding_text_lowercases_and_strips():
    assert build_embedding_text("  RED CHAIR  ", " BrandX ") == "red chair. brandx"


def test_build_embedding_text_handles_missing_brand():
    assert build_embedding_text("Red Chair", None) == "red chair"
    assert build_embedding_text("Red Chair", "") == "red chair"


def test_build_embedding_text_handles_missing_title():
    assert build_embedding_text(None, "Ergohuman") == "ergohuman"
    assert build_embedding_text("", "Ergohuman") == "ergohuman"


def test_build_embedding_text_handles_both_missing():
    assert build_embedding_text(None, None) == ""
    assert build_embedding_text("", "") == ""


def test_vec_items_count_zero_when_table_missing():
    conn = sqlite3.connect(":memory:")
    assert vec_items_count(conn) == 0
    conn.close()


def test_vec_items_count_after_create():
    conn = sqlite3.connect(":memory:")
    load_sqlite_vec(conn)
    conn.executescript(VECTOR_SCHEMA)
    assert vec_items_count(conn) == 0
    conn.execute(
        "INSERT INTO vec_items(item_id, embedding) VALUES (?, ?)",
        ("test-id", b"\x00" * (EMBEDDING_DIM * 4)),
    )
    conn.commit()
    assert vec_items_count(conn) == 1
    conn.close()


def test_vec_index_meta_round_trip():
    conn = sqlite3.connect(":memory:")
    load_sqlite_vec(conn)
    conn.executescript(VECTOR_SCHEMA)
    vec_index_meta_insert(conn, model_name=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
    conn.commit()
    meta = vec_index_meta_get(conn)
    assert meta["model"] == EMBEDDING_MODEL
    assert meta["dim"] == str(EMBEDDING_DIM)
    assert "built_at" in meta
    conn.close()


def test_vec_index_meta_empty_when_table_missing():
    conn = sqlite3.connect(":memory:")
    assert vec_index_meta_get(conn) == {}
    conn.close()


def test_structured_filter_attributes_whitelist():
    """The whitelist must contain the attributes the structured_filter
    helper accepts. If you add an attribute here, also update
    ShoppingBrief + apply_structured_filter + extract_brief session-state
    writes."""
    expected = {"color", "material", "pattern", "finish_type", "fabric_type", "style"}
    assert set(STRUCTURED_FILTER_ATTRIBUTES) == expected
