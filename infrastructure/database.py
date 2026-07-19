"""Indexed SQLite catalog over the offline Amazon dataset."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from domain.entities import CatalogProduct

BASE_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE categories (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE products (
    id                    INTEGER PRIMARY KEY,
    asin                  TEXT NOT NULL UNIQUE,
    title                 TEXT NOT NULL,
    image_url             TEXT NOT NULL,
    product_url           TEXT NOT NULL UNIQUE,
    stars                 REAL NOT NULL CHECK (stars BETWEEN 0 AND 5),
    review_count          INTEGER NOT NULL CHECK (review_count >= 0),
    price                 REAL NOT NULL CHECK (price >= 0),
    list_price            REAL NOT NULL CHECK (list_price >= 0),
    category_id           INTEGER NOT NULL REFERENCES categories(id),
    is_best_seller        INTEGER NOT NULL CHECK (is_best_seller IN (0, 1)),
    bought_last_month     INTEGER NOT NULL CHECK (bought_last_month >= 0)
);
"""

INDEX_SCHEMA = """
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_stars ON products(stars);
CREATE INDEX idx_products_popularity ON products(bought_last_month DESC);
CREATE INDEX idx_products_category_price_stars
    ON products(category_id, price, stars DESC);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE product_fts USING fts5(
    title,
    content='products',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER products_ai AFTER INSERT ON products BEGIN
    INSERT INTO product_fts(rowid, title) VALUES (new.id, new.title);
END;
CREATE TRIGGER products_ad AFTER DELETE ON products BEGIN
    INSERT INTO product_fts(product_fts, rowid, title)
    VALUES ('delete', old.id, old.title);
END;
CREATE TRIGGER products_au AFTER UPDATE OF title ON products BEGIN
    INSERT INTO product_fts(product_fts, rowid, title)
    VALUES ('delete', old.id, old.title);
    INSERT INTO product_fts(rowid, title) VALUES (new.id, new.title);
END;
"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def create_schema(conn: sqlite3.Connection, *, rebuild_fts: bool = False) -> None:
    """Create the normalized catalog schema on an empty connection."""
    conn.executescript(BASE_SCHEMA)
    conn.executescript(INDEX_SCHEMA)
    conn.executescript(FTS_SCHEMA)
    if rebuild_fts:
        conn.execute("INSERT INTO product_fts(product_fts) VALUES ('rebuild')")
    conn.commit()


def _fts_expression(query: str) -> str:
    """Convert free text into a safe FTS5 all-token expression."""
    terms = _TOKEN_RE.findall(query)[:16]
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


class ProductCatalogRepository:
    """Read-only catalog queries used by the research agent tools."""

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

    def find_categories(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        """Find category IDs by case-insensitive name fragment."""
        limit = max(1, min(limit, 50))
        pattern = f"%{query.strip().lower()}%"
        rows = self._conn.execute(
            """
            SELECT c.id, c.name, COUNT(p.id) AS product_count
            FROM categories AS c
            LEFT JOIN products AS p ON p.category_id = c.id
            WHERE LOWER(c.name) LIKE ?
            GROUP BY c.id, c.name
            ORDER BY product_count DESC, c.name
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [
            {"category_id": row["id"], "category_name": row["name"],
             "product_count": row["product_count"]}
            for row in rows
        ]

    def search(
        self,
        query: str,
        *,
        category_id: int = 0,
        max_price: float = 0,
        min_stars: float = 0,
        bestseller_only: bool = False,
        limit: int = 10,
    ) -> list[CatalogProduct]:
        """Search titles, then apply exact structured catalog filters."""
        limit = max(1, min(limit, 50))
        expression = _fts_expression(query)
        params: list[object] = []
        where = ["p.price > 0"]

        if expression:
            from_sql = """
                FROM (
                    SELECT rowid, bm25(product_fts) AS text_rank
                    FROM product_fts
                    WHERE product_fts MATCH ?
                    LIMIT 10000
                ) AS matches
                JOIN products AS p ON p.id = matches.rowid
                JOIN categories AS c ON c.id = p.category_id
            """
            params.append(expression)
            order_sql = (
                "ORDER BY matches.text_rank, p.is_best_seller DESC, "
                "p.stars DESC, p.bought_last_month DESC"
            )
        else:
            from_sql = """
                FROM products AS p
                JOIN categories AS c ON c.id = p.category_id
            """
            order_sql = (
                "ORDER BY p.is_best_seller DESC, p.stars DESC, "
                "p.bought_last_month DESC, p.review_count DESC"
            )

        if category_id > 0:
            where.append("p.category_id = ?")
            params.append(category_id)
        if max_price > 0:
            where.append("p.price <= ?")
            params.append(max_price)
        if min_stars > 0:
            where.append("p.stars >= ?")
            params.append(min_stars)
        if bestseller_only:
            where.append("p.is_best_seller = 1")

        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT p.id, p.asin, p.title, p.image_url, p.product_url,
                   p.stars, p.review_count, p.price, p.list_price,
                   p.category_id, c.name AS category_name,
                   p.is_best_seller, p.bought_last_month
            {from_sql}
            WHERE {' AND '.join(where)}
            {order_sql}
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_product(row) for row in rows]

    def get_by_asin(self, asin: str) -> CatalogProduct | None:
        """Return one product by its stable Amazon identifier."""
        row = self._conn.execute(
            """
            SELECT p.id, p.asin, p.title, p.image_url, p.product_url,
                   p.stars, p.review_count, p.price, p.list_price,
                   p.category_id, c.name AS category_name,
                   p.is_best_seller, p.bought_last_month
            FROM products AS p
            JOIN categories AS c ON c.id = p.category_id
            WHERE p.asin = ?
            """,
            (asin,),
        ).fetchone()
        return self._row_to_product(row) if row else None

    def stats(self) -> dict[str, int]:
        """Return compact catalog counts for boot diagnostics."""
        row = self._conn.execute(
            """
            SELECT (SELECT COUNT(*) FROM products) AS products,
                   (SELECT COUNT(*) FROM categories) AS categories
            """
        ).fetchone()
        return {"products": row["products"], "categories": row["categories"]}

    @staticmethod
    def _row_to_product(row: sqlite3.Row) -> CatalogProduct:
        return CatalogProduct(
            id=row["id"],
            asin=row["asin"],
            title=row["title"],
            image_url=row["image_url"],
            product_url=row["product_url"],
            stars=float(row["stars"]),
            review_count=row["review_count"],
            price=float(row["price"]),
            list_price=float(row["list_price"]),
            category_id=row["category_id"],
            category_name=row["category_name"],
            is_best_seller=bool(row["is_best_seller"]),
            bought_last_month=row["bought_last_month"],
        )

    def close(self) -> None:
        self._conn.close()
