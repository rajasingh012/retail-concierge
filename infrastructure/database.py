"""Indexed SQLite catalog over the Amazon Berkeley Objects metadata archive."""
from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

# Short English-language tag preferred for flattening. When absent, fall back
# to any English text, then any value.
ENGLISH_LANG_RE = re.compile(r"^en(_|$)", re.IGNORECASE)

BASE_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE listings (
    id                INTEGER PRIMARY KEY,
    item_id           TEXT NOT NULL,
    marketplace       TEXT NOT NULL,
    country           TEXT NOT NULL,
    product_type      TEXT,
    title_en          TEXT NOT NULL,
    brand_en          TEXT,
    main_image_id     TEXT,
    has_bullet        INTEGER NOT NULL CHECK (has_bullet IN (0, 1)),
    has_dimensions    INTEGER NOT NULL CHECK (has_dimensions IN (0, 1)),
    has_weight        INTEGER NOT NULL CHECK (has_weight IN (0, 1)),
    has_material      INTEGER NOT NULL CHECK (has_material IN (0, 1)),
    product_url       TEXT NOT NULL UNIQUE
);
"""

INDEX_SCHEMA = """
CREATE INDEX idx_listings_product_type ON listings(product_type);
CREATE INDEX idx_listings_marketplace ON listings(marketplace);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE listing_fts USING fts5(
    title_en,
    brand_en,
    content='listings',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER listings_ai AFTER INSERT ON listings BEGIN
    INSERT INTO listing_fts(rowid, title_en, brand_en) VALUES (new.id, new.title_en, new.brand_en);
END;
CREATE TRIGGER listings_ad AFTER DELETE ON listings BEGIN
    INSERT INTO listing_fts(listing_fts, rowid, title_en, brand_en)
    VALUES ('delete', old.id, old.title_en, old.brand_en);
END;
CREATE TRIGGER listings_au AFTER UPDATE OF title_en, brand_en ON listings BEGIN
    INSERT INTO listing_fts(listing_fts, rowid, title_en, brand_en)
    VALUES ('delete', old.id, old.title_en, old.brand_en);
    INSERT INTO listing_fts(rowid, title_en, brand_en)
    VALUES (new.id, new.title_en, new.brand_en);
END;
"""

TEXT_VALUES_SCHEMA = """
CREATE TABLE listing_text_values (
    id          INTEGER PRIMARY KEY,
    listing_id  INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    item_id     TEXT NOT NULL,
    attribute   TEXT NOT NULL,
    value       TEXT NOT NULL,
    language    TEXT
);
CREATE INDEX idx_text_values_listing ON listing_text_values(listing_id);
CREATE INDEX idx_text_values_item ON listing_text_values(item_id);
CREATE INDEX idx_text_values_attr ON listing_text_values(attribute, value);
"""

DIMENSIONS_SCHEMA = """
CREATE TABLE listing_dimensions (
    id              INTEGER PRIMARY KEY,
    listing_id      INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    item_id         TEXT NOT NULL,
    dimension       TEXT NOT NULL CHECK (dimension IN ('height','width','length','weight')),
    value           REAL NOT NULL CHECK (value >= 0),
    unit            TEXT NOT NULL,
    is_normalized   INTEGER NOT NULL CHECK (is_normalized IN (0, 1))
);
CREATE INDEX idx_dimensions_listing ON listing_dimensions(listing_id);
CREATE INDEX idx_dimensions_item ON listing_dimensions(item_id);
CREATE INDEX idx_dimensions_key ON listing_dimensions(dimension, value);
"""


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Conversion factors to canonical units (centimeters for length, grams for weight).
_LENGTH_TO_CM = {
    "cm": 1.0, "centimeter": 1.0, "centimeters": 1.0,
    "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
    "m": 100.0, "meter": 100.0, "meters": 100.0,
    "in": 2.54, "inch": 2.54, "inches": 2.54,
    "ft": 30.48, "foot": 30.48, "feet": 30.48,
}
_WEIGHT_TO_G = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
}


def _fts_expression(query: str) -> str:
    """Convert free text into a safe FTS5 all-token expression."""
    terms = _TOKEN_RE.findall(query)[:16]
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _best_text(values):
    """Return (text, language) from an ABO [{language_tag, value, ...}] array."""
    if not values:
        return None, None
    en = next((v for v in values if ENGLISH_LANG_RE.match(v.get("language_tag", ""))), None)
    if en is not None:
        return en.get("value"), en.get("language_tag")
    first = values[0]
    return first.get("value"), first.get("language_tag")


def _first_text(values):
    if not values:
        return None
    return values[0].get("value")


def _dimension_value(d):
    """Return (value, unit, is_normalized) from an item_dimensions sub-dict."""
    if not isinstance(d, dict):
        return None
    if d.get("normalized_value"):
        nv = d["normalized_value"]
        return (nv.get("value"), nv.get("unit"), 1)
    if d.get("value") is not None:
        return (d.get("value"), d.get("unit"), 0)
    return None


def create_schema(conn: sqlite3.Connection, *, rebuild_fts: bool = False) -> None:
    conn.executescript(BASE_SCHEMA)
    conn.executescript(INDEX_SCHEMA)
    conn.executescript(FTS_SCHEMA)
    conn.executescript(TEXT_VALUES_SCHEMA)
    conn.executescript(DIMENSIONS_SCHEMA)
    if rebuild_fts:
        conn.execute("INSERT INTO listing_fts(listing_fts) VALUES ('rebuild')")
    conn.commit()


class ABOCatalogRepository:
    """Read-only catalog queries over the imported ABO listings."""

    def __init__(self, db_path: str | Path, *, read_only: bool = True) -> None:
        path = Path(db_path).expanduser().resolve()
        if read_only:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Catalog database not found: {path}. "
                    "Run scripts/import_catalog.py first."
                )
            self._conn = sqlite3.connect(
                f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def find_product_types(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        """Return product_type buckets ordered by listing count."""
        limit = max(1, min(limit, 50))
        pattern = f"%{query.strip().lower()}%"
        rows = self._conn.execute(
            """
            SELECT product_type, COUNT(*) AS product_count
            FROM listings
            WHERE LOWER(IFNULL(product_type, '')) LIKE ?
            GROUP BY product_type
            ORDER BY product_count DESC, product_type
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [
            {"product_type": row["product_type"], "product_count": row["product_count"]}
            for row in rows
        ]

    def search(
        self,
        query: str,
        *,
        product_type: str = "",
        max_dimension_cm: float = 0.0,
        limit: int = 10,
    ) -> list[dict]:
        """Return BM25-ranked listings with structured evidence."""
        limit = max(1, min(limit, 50))
        expression = _fts_expression(query)
        where = ["l.title_en <> ''"]
        params: list[object] = []

        if expression:
            from_sql = """
                FROM (
                    SELECT rowid, bm25(listing_fts) AS text_rank
                    FROM listing_fts
                    WHERE listing_fts MATCH ?
                    LIMIT 10000
                ) AS matches
                JOIN listings AS l ON l.id = matches.rowid
            """
            params.append(expression)
            order_sql = "ORDER BY matches.text_rank, l.id"
        else:
            from_sql = "FROM listings AS l"
            order_sql = "ORDER BY l.id"

        if product_type:
            where.append("l.product_type = ?")
            params.append(product_type)

        if max_dimension_cm > 0:
            where.append(
                """EXISTS (
                    SELECT 1 FROM listing_dimensions d
                    WHERE d.item_id = l.item_id
                      AND d.dimension = 'length'
                      AND d.value <= ?
                      AND d.unit = 'cm'
                )"""
            )
            params.append(max_dimension_cm)

        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT l.item_id, l.marketplace, l.country, l.product_type,
                   l.title_en, l.brand_en, l.main_image_id,
                   l.has_bullet, l.has_dimensions, l.has_weight, l.has_material,
                   l.product_url
            {from_sql}
            WHERE {' AND '.join(where)}
            {order_sql}
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_text_values(self, item_id: str) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT attribute, value, language FROM listing_text_values "
            "WHERE item_id = ? ORDER BY attribute, id",
            (item_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_dimensions(self, item_id: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT dimension, value, unit, is_normalized FROM listing_dimensions "
            "WHERE item_id = ? ORDER BY dimension",
            (item_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        row = self._conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM listings) AS listings,
                (SELECT COUNT(DISTINCT product_type) FROM listings) AS product_types,
                (SELECT COUNT(*) FROM listing_dimensions) AS dimensions,
                (SELECT COUNT(*) FROM listing_text_values) AS text_values
            """
        ).fetchone()
        return dict(row)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    def close(self) -> None:
        self._conn.close()


# ----- Import helpers ------------------------------------------------------

def _extract_dimensions(item_dimensions, item_weight):
    """Return a list of (dimension, value, unit, is_normalized) records."""
    out = []
    for key, raw in (("length", item_dimensions.get("length") if item_dimensions else None),
                     ("width", item_dimensions.get("width") if item_dimensions else None),
                     ("height", item_dimensions.get("height") if item_dimensions else None)):
        rec = _dimension_value(raw)
        if rec is not None and rec[0] is not None:
            out.append((key, float(rec[0]), str(rec[1] or ""), int(rec[2])))
    if item_weight:
        if isinstance(item_weight, list):
            for entry in item_weight:
                rec = _dimension_value(entry)
                if rec is not None and rec[0] is not None:
                    out.append(("weight", float(rec[0]), str(rec[1] or ""), int(rec[2])))
                    break
        elif isinstance(item_weight, dict):
            rec = _dimension_value(item_weight)
            if rec is not None and rec[0] is not None:
                out.append(("weight", float(rec[0]), str(rec[1] or ""), int(rec[2])))
    return out


def normalize_dimension_value(dimension: str, value: float, unit: str) -> float:
    """Convert a typed dimension to its canonical unit (cm or grams)."""
    if dimension == "weight":
        factor = _WEIGHT_TO_G.get(unit.lower())
    else:
        factor = _LENGTH_TO_CM.get(unit.lower())
    if factor is None or value is None:
        return value
    return value * factor


def unit_to_canonical(dimension: str) -> str:
    return "g" if dimension == "weight" else "cm"


def closest_dimension_match(query_value: float, candidates: list[tuple[float, str]]) -> tuple[float, str] | None:
    """Return the candidate with the smallest absolute delta to query_value."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c[0] - query_value))
