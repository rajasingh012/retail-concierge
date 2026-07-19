#!/usr/bin/env python3
"""Import the Amazon product CSVs into an indexed SQLite catalog."""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import BASE_SCHEMA, FTS_SCHEMA, INDEX_SCHEMA

CATEGORY_FIELDS = ["id", "category_name"]
PRODUCT_FIELDS = [
    "asin", "title", "imgUrl", "productURL", "stars", "reviews", "price",
    "listPrice", "category_id", "isBestSeller", "boughtInLastMonth",
]
INSERT_PRODUCTS = """
INSERT INTO products(
    asin, title, image_url, product_url, stars, review_count, price,
    list_price, category_id, is_best_seller, bought_last_month
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _require_fields(reader: csv.DictReader, expected: list[str], path: Path) -> None:
    if reader.fieldnames != expected:
        raise ValueError(
            f"Unexpected columns in {path}: {reader.fieldnames}; expected {expected}"
        )


def _read_categories(path: Path) -> list[tuple[int, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_fields(reader, CATEGORY_FIELDS, path)
        rows = [(int(row["id"]), row["category_name"].strip()) for row in reader]
    if not rows:
        raise ValueError(f"No categories found in {path}")
    return rows


def _product_rows(
    path: Path, category_ids: set[int], limit: int = 0
) -> Iterable[tuple[object, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_fields(reader, PRODUCT_FIELDS, path)
        for index, row in enumerate(reader, start=1):
            if limit and index > limit:
                break
            category_id = int(row["category_id"])
            if category_id not in category_ids:
                raise ValueError(
                    f"Unknown category_id={category_id} at product row {index}"
                )
            best_seller = row["isBestSeller"].strip().lower()
            if best_seller not in {"true", "false"}:
                raise ValueError(
                    f"Invalid isBestSeller={row['isBestSeller']!r} at row {index}"
                )
            yield (
                row["asin"].strip(),
                row["title"].strip(),
                row["imgUrl"].strip(),
                row["productURL"].strip(),
                float(row["stars"]),
                int(row["reviews"]),
                float(row["price"]),
                float(row["listPrice"]),
                category_id,
                int(best_seller == "true"),
                int(row["boughtInLastMonth"]),
            )


def _batched(rows: Iterable[tuple[object, ...]], size: int = 10_000):
    batch: list[tuple[object, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def import_catalog(
    products_csv: Path,
    categories_csv: Path,
    database: Path,
    *,
    limit: int = 0,
) -> dict[str, int | float]:
    """Build a new catalog beside the target and atomically replace it."""
    products_csv = products_csv.expanduser().resolve()
    categories_csv = categories_csv.expanduser().resolve()
    database = database.expanduser().resolve()
    if not products_csv.is_file() or not categories_csv.is_file():
        raise FileNotFoundError("Both product and category CSV files are required")

    database.parent.mkdir(parents=True, exist_ok=True)
    temp_db = database.with_name(f".{database.name}.importing")
    temp_db.unlink(missing_ok=True)
    categories = _read_categories(categories_csv)
    category_ids = {row[0] for row in categories}
    started = time.perf_counter()
    inserted = 0
    conn: sqlite3.Connection | None = None

    try:
        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.executescript(BASE_SCHEMA)
        conn.executemany("INSERT INTO categories(id, name) VALUES (?, ?)", categories)

        for batch in _batched(_product_rows(products_csv, category_ids, limit)):
            conn.executemany(INSERT_PRODUCTS, batch)
            inserted += len(batch)
            if inserted % 100_000 == 0:
                print(f"[import] {inserted:,} products", flush=True)
        conn.commit()

        print("[import] building indexes and FTS5 catalog", flush=True)
        conn.executescript(INDEX_SCHEMA)
        conn.executescript(FTS_SCHEMA)
        conn.execute("INSERT INTO product_fts(product_fts) VALUES ('rebuild')")
        conn.commit()

        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if fk_errors or integrity != "ok":
            raise RuntimeError(
                f"Catalog verification failed: foreign_keys={fk_errors[:3]}, "
                f"integrity={integrity}"
            )
        conn.close()
        conn = None
        os.replace(temp_db, database)
    except Exception:
        if conn is not None:
            conn.close()
        temp_db.unlink(missing_ok=True)
        raise

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "products": inserted,
        "categories": len(categories),
        "seconds": elapsed,
        "database_bytes": database.stat().st_size,
    }


def cli() -> None:
    default_archive = Path.home() / "Downloads" / "archive"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--products", type=Path,
        default=default_archive / "amazon_products.csv",
    )
    parser.add_argument(
        "--categories", type=Path,
        default=default_archive / "amazon_categories.csv",
    )
    parser.add_argument("--database", type=Path, default=Path("retail_catalog.db"))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Import only the first N products; 0 imports the full dataset.",
    )
    args = parser.parse_args()
    result = import_catalog(
        args.products, args.categories, args.database, limit=max(args.limit, 0)
    )
    print(
        f"[import] complete: {result['products']:,} products, "
        f"{result['categories']} categories, {result['seconds']}s, "
        f"{result['database_bytes'] / (1024 ** 2):.1f} MiB"
    )


if __name__ == "__main__":
    cli()
