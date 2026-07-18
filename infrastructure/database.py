"""SQLite persistence layer with native JSON path filtering.

Schema:
    products(product_id TEXT PK, platform TEXT, title TEXT,
             base_price REAL, structured_data TEXT)
The `structured_data` column is a JSON blob; queries use
`json_extract(structured_data, '$.field')` to filter on nested
fields without re-parsing in Python.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Optional

from domain.entities import ItemVariant, ProductPayload


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,
    platform        TEXT NOT NULL,
    title           TEXT NOT NULL,
    base_price      REAL,
    structured_data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_platform ON products(platform);
"""


class ProductCatalogRepository:
    """Thin wrapper over a single SQLite file with JSON1 path queries."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # `detect_types` keeps date columns friendly; `PARSE_DECLTYPES` not needed here.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---------- writes ----------

    def upsert(self, product: ProductPayload, platform: str, product_id: str) -> None:
        """Insert or replace a product. JSON blob holds variants + dynamic attrs."""
        base_price = min((v.price for v in product.variants), default=0.0)
        blob = json.dumps(
            {
                "variants": [v.__dict__ for v in product.variants],
                "reviews": product.reviews,
                "dynamic_attributes": product.dynamic_attributes,
            },
            ensure_ascii=False,
        )
        self._conn.execute(
            """
            INSERT INTO products(product_id, platform, title, base_price, structured_data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                platform=excluded.platform,
                title=excluded.title,
                base_price=excluded.base_price,
                structured_data=excluded.structured_data
            """,
            (product_id, platform, product.title, base_price, blob),
        )
        self._conn.commit()

    def bulk_upsert(
        self, products: Iterable[tuple[ProductPayload, str, str]]
    ) -> None:
        for product, platform, product_id in products:
            self.upsert(product, platform, product_id)

    # ---------- reads ----------

    def all(self) -> List[ProductPayload]:
        cur = self._conn.execute("SELECT * FROM products")
        return [self._row_to_payload(r) for r in cur.fetchall()]

    def get(self, product_id: str) -> Optional[ProductPayload]:
        cur = self._conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        )
        row = cur.fetchone()
        return self._row_to_payload(row) if row else None

    def find_by_brand(self, brand: str) -> List[ProductPayload]:
        """Native JSON lookup: case-insensitive brand equality."""
        cur = self._conn.execute(
            """
            SELECT * FROM products
            WHERE LOWER(json_extract(structured_data, '$.dynamic_attributes.brand'))
                  = LOWER(?)
            """,
            (brand,),
        )
        return [self._row_to_payload(r) for r in cur.fetchall()]

    def find_by_variant_attribute(
        self, key: str, value: Any
    ) -> List[ProductPayload]:
        """Filter on a variant attribute via JSON path.

        e.g. find_by_variant_attribute('color', 'graphite')
        """
        cur = self._conn.execute(
            """
            SELECT * FROM products
            WHERE EXISTS (
                SELECT 1 FROM json_each(
                    json_extract(structured_data, '$.variants')
                )
                WHERE LOWER(json_extract(value, '$.attributes.' || ?)) = LOWER(?)
            )
            """,
            (key, str(value)),
        )
        return [self._row_to_payload(r) for r in cur.fetchall()]

    def price_under(self, ceiling: float) -> List[ProductPayload]:
        cur = self._conn.execute(
            "SELECT * FROM products WHERE base_price <= ?", (ceiling,)
        )
        return [self._row_to_payload(r) for r in cur.fetchall()]

    # ---------- internals ----------

    @staticmethod
    def _row_to_payload(row: sqlite3.Row) -> ProductPayload:
        blob = json.loads(row["structured_data"])
        variants = [
            ItemVariant(
                sku=v["sku"],
                label=v["label"],
                price=v["price"],
                in_stock=v.get("in_stock", True),
                attributes=v.get("attributes", {}),
            )
            for v in blob.get("variants", [])
        ]
        return ProductPayload(
            title=row["title"],
            source_url=row["product_id"],  # we key products by canonical URL
            variants=variants,
            reviews=blob.get("reviews", []),
            dynamic_attributes=blob.get("dynamic_attributes", {}),
        )

    def close(self) -> None:
        self._conn.close()