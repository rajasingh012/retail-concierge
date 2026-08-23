"""Structured attribute filtering for catalog recall candidates.

Industry-standard e-commerce pattern (Algolia, Bloomreach, Elastic,
Amazon's hybrid search): recall candidates from BM25 + vector are
pre-filtered against structured attributes (color, material, pattern,
finish, fabric, style) via SQL ``LIKE`` matching against
``listing_text_values.value``.

Why pre-filter instead of post-filter or embedding-into-vector:
  * Embedding attribute values into the vector pollutes the semantic space
    with attribute NAMES ("color:", "material:") and degrades recall for
    queries that don't specify those attributes.
  * Post-filter loses recall — if the top-50 vector hits don't include
    enough "red velvet" sofas to fill the result list, the user sees
    a thin list.
  * Pre-filter on a SELECTIVE attribute (e.g. color=maroon at 0.2% of the
    catalog) is what every production search engine does. SQLite's LIKE
    against an indexed ``(attribute, value)`` pair is fast enough for
    the candidate set sizes the agent sees (typically 50-200 items).

The catalog IS the synonym dictionary: user-typed "red" returns every
listing whose ``color`` value contains "red" as a substring. Merchants
write "Crimson Red" or "Burgundy" — if their text contains "red", it
matches. New color names added by merchants are automatically covered
without a curated synonym list. False positives (e.g. "red AND blue
stripe") are acceptable trade-off for the POC.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

from infrastructure.database import STRUCTURED_FILTER_ATTRIBUTES


def _valid_attribute(name: str) -> str:
    """Return ``name`` if it is a known structured-filter attribute,
    else raise ValueError. Whitelisting prevents SQL injection via the
    attribute name (it's not a parameter binding — it's interpolated
    into the WHERE clause).
    """
    if name not in STRUCTURED_FILTER_ATTRIBUTES:
        raise ValueError(
            f"unknown structured-filter attribute {name!r}; "
            f"allowed: {STRUCTURED_FILTER_ATTRIBUTES}"
        )
    return name


def apply_structured_filter(
    conn: sqlite3.Connection,
    item_ids: Iterable[str],
    *,
    color: str = "",
    material: str = "",
    pattern: str = "",
    finish_type: str = "",
    fabric_type: str = "",
    style: str = "",
) -> list[str]:
    """Pre-filter a set of candidate ``item_id``s by structured attributes.

    Each non-empty user term is matched via ``LIKE %term%`` against the
    corresponding attribute in ``listing_text_values``. The output is
    the intersection: a candidate must satisfy every specified filter
    to survive. Empty / missing terms are no-ops.

    Implementation: one SQL round-trip per active filter, intersecting in
    Python. The catalog has at most ~6 active filters per query and
    candidate sets are 50-200 items, so the round-trip cost is
    negligible compared to the per-query LLM call.

    Args:
        conn: Open SQLite connection to the catalog database.
        item_ids: Candidate item_ids to filter (typically from
            search_catalog + search_vector union).
        color, material, pattern, finish_type, fabric_type, style:
            User-typed filter terms. Empty string means "don't filter
            on this attribute."

    Returns:
        A list of item_ids (subset of the input) that match every
        specified filter. Empty list if no candidates survive.
        Order is NOT preserved — callers should re-rank or re-merge.
    """
    candidates: set[str] = {str(i) for i in item_ids if i}
    if not candidates:
        return []

    filters = [
        ("color", color),
        ("material", material),
        ("pattern", pattern),
        ("finish_type", finish_type),
        ("fabric_type", fabric_type),
        ("style", style),
    ]

    for attr, term in filters:
        term = term.strip()
        if not term:
            continue
        _valid_attribute(attr)

        placeholders = ",".join("?" * len(candidates))
        sql = f"""
            SELECT DISTINCT ltv.item_id
            FROM listing_text_values ltv
            WHERE ltv.attribute = ?
              AND LOWER(ltv.value) LIKE ?
              AND ltv.item_id IN ({placeholders})
        """
        rows = conn.execute(
            sql,
            (attr, f"%{term.lower()}%", *sorted(candidates)),
        ).fetchall()
        survivors = {row[0] for row in rows}
        candidates &= survivors
        # Early exit: if any filter empties the set, no point continuing.
        if not candidates:
            return []

    return sorted(candidates)
