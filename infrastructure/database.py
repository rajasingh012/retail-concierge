"""Indexed SQLite catalog over the Amazon Berkeley Objects metadata archive."""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable

# Short English-language tag preferred for flattening. When absent, fall back
# to any English text, then any value.
ENGLISH_LANG_RE = re.compile(r"^en(_|$)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Vector index (sqlite-vec) constants
# ---------------------------------------------------------------------------
# Pinned at module level so the schema, the build script, and the KNN tool
# all agree on the same model + dimension. Changing this requires dropping
# the vec_items table and rebuilding — the catalog data is immutable so the
# build is a one-shot operation.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Structured attributes the structured_filter helper accepts. These map
# 1:1 to the ``attribute`` column in ``listing_text_values``.
STRUCTURED_FILTER_ATTRIBUTES = (
    "color",
    "material",
    "pattern",
    "finish_type",
    "fabric_type",
    "style",
)


def build_embedding_text(title_en: str | None, brand_en: str | None) -> str:
    """Concatenate the two columns embedded into the vector for a listing.

    Title + brand, lowercased, stripped. Mirrors the SQL builder used in
    scripts/build_vector_index.py so the query-side encoding produces a
    vector in the same neighborhood as the indexed rows.
    """
    parts: list[str] = []
    if title_en and title_en.strip():
        parts.append(title_en.strip().lower())
    if brand_en and brand_en.strip():
        parts.append(brand_en.strip().lower())
    return ". ".join(parts)


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension on ``conn``.

    Uses sqlite3's ``enable_load_extension`` API. The extension is required
    to create ``vec0`` virtual tables and run ``MATCH ... AND k = N`` KNN
    queries.

    On Linux + Python 3.11+, ``sqlite_vec`` ships a ``sqlite_vec`` Python
    package whose entry point exposes ``loadable_path()`` returning the
    compiled shared library path. This is the portable install path and
    does not require ``SQLITE_EXTENSIONS_PATH`` overrides.
    """
    try:
        import sqlite_vec  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sqlite-vec is not installed in the active environment. "
            "Run: uv pip install sqlite-vec"
        ) from e

    ext_path = sqlite_vec.loadable_path()
    conn.enable_load_extension(True)
    try:
        conn.load_extension(ext_path)
    finally:
        conn.enable_load_extension(False)


def vec_items_count(conn: sqlite3.Connection) -> int:
    """Return the number of rows in ``vec_items`` (0 if table doesn't exist)."""
    try:
        cur = conn.execute("SELECT COUNT(*) FROM vec_items")
    except sqlite3.OperationalError:
        return 0
    return int(cur.fetchone()[0])


def vec_index_meta_insert(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    dim: int,
) -> None:
    """Insert (or replace) the sidecar metadata that records what built
    the index and when.

    A future-self reading the database needs to know: which model? what
    dimension? when? Without this row, you can't tell whether a stale
    index needs rebuilding for a model swap or is still consistent.
    """
    import time

    conn.execute("DELETE FROM vec_index_meta")
    conn.execute(
        "INSERT INTO vec_index_meta(key, value) VALUES (?, ?), (?, ?), (?, ?)",
        ("model", model_name, "dim", str(dim), "built_at", str(int(time.time()))),
    )


def vec_index_meta_get(conn: sqlite3.Connection) -> dict[str, str]:
    """Read the sidecar metadata as a dict. Empty dict if absent."""
    try:
        rows = conn.execute("SELECT key, value FROM vec_index_meta").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {row[0]: row[1] for row in rows}


VECTOR_SCHEMA = """
-- vec_items: KNN-searchable embedding for every active listing.
-- item_id is the join key against listings.item_id (no FK so sqlite-vec
-- stays simple — integrity is enforced at build time by skipping listings
-- that don't exist in `listings`).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    item_id TEXT PRIMARY KEY,
    embedding float[384]
);

-- vec_index_meta: sidecar KV recording what built the index.
CREATE TABLE IF NOT EXISTS vec_index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

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
    product_url       TEXT NOT NULL UNIQUE,
    url_active        INTEGER NOT NULL DEFAULT 1 CHECK (url_active IN (0, 1))
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
    # Vector index schema is conditional: requires sqlite-vec extension to be
    # loaded. Skipped silently if the extension isn't installed; callers that
    # need the index (build_vector_index.py) load the extension first and call
    # this script's VECTOR_SCHEMA directly.
    try:
        load_sqlite_vec(conn)
        conn.executescript(VECTOR_SCHEMA)
    except RuntimeError:
        # sqlite-vec not installed; the rest of the schema is unaffected.
        # The vector build script and the search_vector tool both raise a
        # clear error if the extension is missing at runtime.
        pass
    if rebuild_fts:
        conn.execute("INSERT INTO listing_fts(listing_fts) VALUES ('rebuild')")
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent in-place upgrades for the listings schema.

    Safe to call on every startup: each step checks current state first.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(listings)").fetchall()}
    if "url_active" not in cols:
        conn.execute(
            "ALTER TABLE listings ADD COLUMN url_active INTEGER "
            "NOT NULL DEFAULT 1 CHECK (url_active IN (0, 1))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_listings_url_active "
            "ON listings(url_active)"
        )
        conn.commit()


class ABOCatalogRepository:
    """Read-only catalog queries over the imported ABO listings."""

    def __init__(self, db_path: str | Path, *, read_only: bool = True) -> None:
        path = Path(db_path).expanduser().resolve()
        # Keep the resolved path: the read-only URI connection can't load
        # the sqlite-vec extension, so search_vector / apply_structured_filter
        # open short-lived writable connections to this same file.
        self._db_path = str(path)
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
            migrate(self._conn)
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

    def list_product_types(self, *, min_listings: int = 5) -> list[dict[str, object]]:
        """Distinct product_type buckets with enough data to be a real signal.

        ``min_listings`` filters single-listing accidentals (test rows, sparse
        imports) that would otherwise pollute the vocabulary. Used to seed
        the brief-time LLM resolver, which only needs canonical types the
        user could realistically have asked about.
        """
        rows = self._conn.execute(
            """
            SELECT product_type, COUNT(*) AS product_count
            FROM listings
            WHERE product_type IS NOT NULL AND product_type <> ''
            GROUP BY product_type
            HAVING product_count >= ?
            ORDER BY product_count DESC, product_type
            """,
            (min_listings,),
        ).fetchall()
        return [
            {"product_type": row["product_type"], "product_count": row["product_count"]}
            for row in rows
        ]

    def list_brands(self, *, limit: int = 100, min_listings: int = 1) -> list[dict[str, object]]:
        """Distinct brand buckets ordered by listing count.

        ``min_listings`` defaults to 1 (any brand) but raise it to drop sparse
        brands. ``limit`` defaults to 100 so the entire common-brand vocabulary
        fits in a brief-time prompt without bloating context (~2k tokens).
        """
        limit = max(1, min(limit, 500))
        rows = self._conn.execute(
            """
            SELECT brand_en, COUNT(*) AS product_count
            FROM listings
            WHERE brand_en IS NOT NULL AND brand_en <> ''
            GROUP BY brand_en
            HAVING product_count >= ?
            ORDER BY product_count DESC, brand_en
            LIMIT ?
            """,
            (min_listings, limit),
        ).fetchall()
        return [
            {"brand": row["brand_en"], "product_count": row["product_count"]}
            for row in rows
        ]


    def find_brands(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        """Return brand buckets ordered by listing count.

        Uses three-tier matching: exact prefix, FTS5, then LIKE fallback.
        """
        limit = max(1, min(limit, 20))
        pattern = f"%{query.strip().lower()}%"
        seen: set[str] = set()
        rows: list[dict[str, object]] = []

        # Tier 1: exact prefix match (fast, highest precision)
        tier1 = self._conn.execute(
            """
            SELECT brand_en, COUNT(*) AS product_count
            FROM listings
            WHERE LOWER(IFNULL(brand_en, '')) LIKE ? ESCAPE '\\'
            GROUP BY brand_en
            ORDER BY product_count DESC, brand_en
            LIMIT ?
            """,
            (f"{query.strip().lower()}%", limit),
        ).fetchall()
        for row in tier1:
            brand = row["brand_en"]
            if brand and brand not in seen:
                seen.add(brand)
                rows.append({"brand": brand, "product_count": row["product_count"]})
                if len(rows) >= limit:
                    break

        # Tier 2: FTS5 (handles stemming, diacritics, close misspellings)
        if len(rows) < limit:
            expression = _fts_expression(query)
            if expression:
                tier2 = self._conn.execute(
                    """
                    SELECT l.brand_en, COUNT(*) AS product_count
                    FROM listing_fts AS f
                    JOIN listings AS l ON l.id = f.rowid
                    WHERE f.brand_en MATCH ?
                      AND l.brand_en IS NOT NULL AND l.brand_en <> ''
                    GROUP BY l.brand_en
                    ORDER BY product_count DESC
                    LIMIT ?
                    """,
                    (expression, limit - len(rows)),
                ).fetchall()
                for row in tier2:
                    brand = row["brand_en"]
                    if brand and brand not in seen:
                        seen.add(brand)
                        rows.append(
                            {"brand": brand, "product_count": row["product_count"]}
                        )
                        if len(rows) >= limit:
                            break

        # Tier 3: LIKE %query% fallback
        if len(rows) < limit:
            tier3 = self._conn.execute(
                """
                SELECT brand_en, COUNT(*) AS product_count
                FROM listings
                WHERE LOWER(IFNULL(brand_en, '')) LIKE ?
                GROUP BY brand_en
                ORDER BY product_count DESC, brand_en
                LIMIT ?
                """,
                (pattern, limit - len(rows)),
            ).fetchall()
            for row in tier3:
                brand = row["brand_en"]
                if brand and brand not in seen:
                    rows.append(
                        {"brand": brand, "product_count": row["product_count"]}
                    )
                    if len(rows) >= limit:
                        break

        return rows

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

        where.append("l.url_active = 1")

        if product_type:
            where.append("l.product_type = ?")
            params.append(product_type)

        if max_dimension_cm > 0:
            where.append(
                """EXISTS (
                    SELECT 1 FROM listing_dimensions d
                    WHERE d.item_id = l.item_id
                      AND d.dimension IN ('height', 'width', 'length')
                      AND d.value <= ?
                      AND d.unit = 'cm'
                    LIMIT 1
                )"""
            )
            params.append(max_dimension_cm)

        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT l.item_id, l.marketplace, l.country, l.product_type,
                   l.title_en, l.brand_en, l.main_image_id,
                   l.has_bullet, l.has_dimensions, l.has_weight, l.has_material,
                   l.product_url, l.url_active
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
                (SELECT COUNT(*) FROM listings WHERE url_active = 1) AS listings_active,
                (SELECT COUNT(*) FROM listings WHERE url_active = 0) AS listings_inactive,
                (SELECT COUNT(DISTINCT product_type) FROM listings) AS product_types,
                (SELECT COUNT(*) FROM listing_dimensions) AS dimensions,
                (SELECT COUNT(*) FROM listing_text_values) AS text_values
            """
        ).fetchone()
        return dict(row)

    def mark_url_inactive(self, item_ids: list[str]) -> int:
        """Soft-delete listings by item_id. Returns the row count flipped."""
        if not item_ids:
            return 0
        placeholders = ",".join("?" * len(item_ids))
        cur = self._conn.execute(
            f"UPDATE listings SET url_active = 0 WHERE item_id IN ({placeholders})",
            item_ids,
        )
        self._conn.commit()
        return cur.rowcount

    def mark_url_active(self, item_ids: list[str]) -> int:
        """Restore previously soft-deleted listings. Returns the row count flipped."""
        if not item_ids:
            return 0
        placeholders = ",".join("?" * len(item_ids))
        cur = self._conn.execute(
            f"UPDATE listings SET url_active = 1 WHERE item_id IN ({placeholders})",
            item_ids,
        )
        self._conn.commit()
        return cur.rowcount

    def iter_product_urls(self, *, product_type: str = "", only_active: bool = False) -> list[tuple[str, str]]:
        """Yield (item_id, product_url) for probing. If product_type set, scope to it."""
        clauses = ["product_url LIKE 'http%'"]
        params: list[object] = []
        if only_active:
            clauses.append("url_active = 1")
        if product_type:
            clauses.append("product_type = ?")
            params.append(product_type)
        where = " AND ".join(clauses)
        return [
            (row[0], row[1])
            for row in self._conn.execute(
                f"SELECT item_id, product_url FROM listings WHERE {where} ORDER BY id",
                params,
            ).fetchall()
        ]

    def encode_query(self, text: str) -> bytes:
        """Encode a query string to a 384-dim float32 bytes vector.

        Uses the same BGE-small-en-v1.5 model that built the index (see
        ``infrastructure.database.EMBEDDING_MODEL``) via fastembed (ONNX
        runtime — no torch dependency, ~30MB model, fast cold-start).
        Cached on the repository instance so repeated identical queries
        (e.g. retries from the LLM) don't re-encode.

        Returns:
            Raw little-endian float32 bytes ready to bind into a
            ``MATCH ?`` clause against ``vec_items``.
        """
        # Lazy import + lazy model load: fastembed pulls in onnxruntime
        # (~30MB) which we don't want at import time.
        if not hasattr(self, "_encoder") or self._encoder is None:  # type: ignore[attr-defined]
            from fastembed import TextEmbedding  # type: ignore

            from infrastructure.database import EMBEDDING_MODEL

            self._encoder = TextEmbedding(model_name=EMBEDDING_MODEL)  # type: ignore[attr-defined]
        # Reuse a small cache to skip identical re-encodings inside one session.
        cache: dict[str, bytes] = getattr(self, "_encoder_cache", {})  # type: ignore[attr-defined]
        if text in cache:
            return cache[text]
        import numpy as np

        # fastembed returns a generator; pull the single embedding.
        # normalize_embeddings is the default for BGE-small — vectors are
        # unit-norm, so cosine distance == L2 distance^2 / 2.
        embeddings = list(self._encoder.embed(text.strip().lower()))  # type: ignore[attr-defined]
        if not embeddings:
            raise RuntimeError(f"fastembed returned no embeddings for {text!r}")
        vec = embeddings[0].astype("float32")
        raw = vec.tobytes()
        cache[text] = raw
        self._encoder_cache = cache  # type: ignore[attr-defined]
        return raw

    def search_vector(
        self,
        query: bytes,
        *,
        limit: int = 50,
        product_type: str = "",
    ) -> list[dict]:
        """KNN over ``vec_items`` using sqlite-vec's brute-force ``MATCH`` query.

        Returns a list of dicts with ``item_id``, ``distance``, plus a few
        denormalized columns joined from ``listings`` for ranking context.
        Distance is cosine distance on unit-norm vectors, in [0, 2].
        Lower = more similar; 0 = identical.

        sqlite-vec requires the extension to be loaded on the connection.
        The repository's main connection is opened read-only via URI, so
        extension loading is unavailable. We open a separate writable
        connection in a context manager — short-lived, single-statement.
        """
        from infrastructure.database import (
            load_sqlite_vec,
            vec_index_meta_get,
        )

        limit = max(1, min(limit, 50))
        # Open a fresh writable connection to the catalog file. The repo's
        # main connection is read-only (file:...?mode=ro) and cannot load
        # extensions; sqlite-vec must be loaded on a writable connection.
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            load_sqlite_vec(conn)
            # Build the WHERE clause for product_type if requested.
            where_clauses: list[str] = []
            params: list[object] = [query, limit]
            if product_type:
                where_clauses.append("AND l.product_type = ?")
                params.insert(-1, product_type)

            # KNN MATCH ... AND k = N returns the N closest rows.
            sql = f"""
                SELECT v.item_id, v.distance,
                       l.title_en, l.brand_en, l.product_type,
                       l.product_url, l.marketplace, l.country,
                       l.has_bullet, l.has_dimensions, l.has_weight,
                       l.has_material, l.url_active
                FROM vec_items v
                JOIN listings l ON l.item_id = v.item_id
                WHERE v.embedding MATCH ?
                  AND k = ?
                  {' '.join(where_clauses)}
                ORDER BY v.distance
            """
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        results: list[dict] = []
        for rank, row in enumerate(rows, start=1):
            results.append({
                "item_id": row[0],
                "distance": float(row[1]),
                "retrieval_rank": rank,
                "title_en": row[2] or "",
                "brand_en": row[3] or "",
                "product_type": row[4] or "",
                "product_url": row[5] or "",
                "marketplace": row[6] or "",
                "country": row[7] or "",
                "has_bullet": int(row[8] or 0),
                "has_dimensions": int(row[9] or 0),
                "has_weight": int(row[10] or 0),
                "has_material": int(row[11] or 0),
                "url_active": int(row[12] or 0),
                "retrieval_backend": "vector",
            })
        return results

    def apply_structured_filter(
        self,
        item_ids: Iterable[str],
        *,
        color: str = "",
        material: str = "",
        pattern: str = "",
        finish_type: str = "",
        fabric_type: str = "",
        style: str = "",
    ) -> list[str]:
        """Apply structured-attribute LIKE filtering to a candidate set.

        Opens a short-lived writable connection (sqlite-vec semantics
        notwithstanding — this helper only touches listing_text_values
        and uses no extensions) and runs
        :func:`infrastructure.structured_filter.apply_structured_filter`.
        Mirrors the connection pattern used by ``search_vector``.
        """
        from infrastructure.structured_filter import apply_structured_filter

        # Same pattern as search_vector: the read-only URI connection can't
        # load extensions, but this helper only needs plain SQL — open a
        # fresh writable connection to the catalog file.
        conn = sqlite3.connect(self._db_path)
        try:
            return apply_structured_filter(
                conn,
                item_ids,
                color=color,
                material=material,
                pattern=pattern,
                finish_type=finish_type,
                fabric_type=fabric_type,
                style=style,
            )
        finally:
            conn.close()

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
