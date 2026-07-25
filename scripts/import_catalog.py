"""Import the Amazon Berkeley Objects listings archive into the SQLite catalog."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import create_schema

SHARD_PATTERN = "listings_*.json.gz"


def _read_shards(archive_root: Path) -> Iterable[dict]:
    metadata_dir = archive_root / "listings" / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {metadata_dir}. Extract the ABO archive first."
        )
    for shard in sorted(metadata_dir.glob(SHARD_PATTERN)):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def _product_url(item_id: str, marketplace: str, country: str) -> str:
    return f"https://www.amazon.com/dp/{item_id}"


def _first_product_type(product_type) -> str | None:
    if not product_type:
        return None
    if isinstance(product_type, list) and product_type:
        v = product_type[0].get("value") if isinstance(product_type[0], dict) else None
        return v or None
    return None


_TEXT_VALUE_ATTRS = (
    "bullet_point",
    "color",
    "material",
    "style",
    "model_name",
    "model_number",
    "pattern",
    "finish_type",
    "fabric_type",
    "item_shape",
    "item_keywords",
)


def _text_records(item_id: str, raw: dict) -> list[tuple[str, str, str | None, str | None]]:
    out: list[tuple[str, str, str | None, str | None]] = []
    for attr in _TEXT_VALUE_ATTRS:
        values = raw.get(attr)
        if not isinstance(values, list):
            continue
        for entry in values:
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            language = entry.get("language_tag")
            if value:
                out.append((item_id, attr, str(value), language))
    return out


def _dimension_records(item_id: str, raw: dict) -> list[tuple[str, float, str, int]]:
    out: list[tuple[str, float, str, int]] = []
    item_dimensions = raw.get("item_dimensions")
    if isinstance(item_dimensions, dict):
        for key in ("height", "width", "length"):
            entry = item_dimensions.get(key)
            if not isinstance(entry, dict):
                continue
            if entry.get("normalized_value"):
                nv = entry["normalized_value"]
                v = nv.get("value")
                unit = nv.get("unit")
                if v is not None:
                    out.append((key, float(v), str(unit or ""), 1))
                continue
            v = entry.get("value")
            unit = entry.get("unit")
            if v is not None:
                out.append((key, float(v), str(unit or ""), 0))
    weight = raw.get("item_weight")
    if isinstance(weight, list) and weight:
        entry = weight[0]
        if isinstance(entry, dict):
            if entry.get("normalized_value"):
                nv = entry["normalized_value"]
                v = nv.get("value")
                unit = nv.get("unit")
                if v is not None:
                    out.append(("weight", float(v), str(unit or ""), 1))
            else:
                v = entry.get("value")
                unit = entry.get("unit")
                if v is not None:
                    out.append(("weight", float(v), str(unit or ""), 0))
    elif isinstance(weight, dict):
        if weight.get("normalized_value"):
            nv = weight["normalized_value"]
            v = nv.get("value")
            unit = nv.get("unit")
            if v is not None:
                out.append(("weight", float(v), str(unit or ""), 1))
        else:
            v = weight.get("value")
            unit = weight.get("unit")
            if v is not None:
                out.append(("weight", float(v), str(unit or ""), 0))
    return out


def _english_text(values):
    if not isinstance(values, list):
        return None
    for entry in values:
        if not isinstance(entry, dict):
            continue
        lang = entry.get("language_tag") or ""
        if lang.lower().startswith("en"):
            v = entry.get("value")
            if v:
                return str(v)
    if values and isinstance(values[0], dict):
        v = values[0].get("value")
        if v:
            return str(v)
    return None


def _populate(
    conn: sqlite3.Connection, archive_root: Path
) -> tuple[int, int, int]:
    """Insert all listings and their text/dimension evidence into the schema."""
    conn.execute("PRAGMA foreign_keys = ON")

    insert_listing = """
    INSERT INTO listings(
        item_id, marketplace, country, product_type, title_en, brand_en,
        main_image_id, has_bullet, has_dimensions, has_weight, has_material,
        product_url
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    insert_text = (
        "INSERT INTO listing_text_values(listing_id, item_id, attribute, value, language) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    insert_dim = (
        "INSERT INTO listing_dimensions(listing_id, item_id, dimension, value, unit, is_normalized) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )

    BATCH = 5000
    listings = 0
    text_values = 0
    dimension_rows = 0
    text_batch: list[tuple] = []
    dim_batch: list[tuple] = []

    for raw in _read_shards(archive_root):
        item_id = raw.get("item_id")
        if not item_id:
            continue
        title_en = _english_text(raw.get("item_name")) or ""
        brand_en = _english_text(raw.get("brand"))
        marketplace = str(raw.get("marketplace") or "")
        country = str(raw.get("country") or "")
        product_type = _first_product_type(raw.get("product_type"))
        main_image_id = raw.get("main_image_id")
        url = _product_url(item_id, marketplace, country)

        # Check if this URL already exists (ABO has 1,805 duplicate ASIN+marketplace pairs).
        existing = conn.execute(
            "SELECT id FROM listings WHERE product_url = ?", (url,)
        ).fetchone()
        if existing is not None:
            continue

        conn.execute(
            insert_listing,
            (
                item_id, marketplace, country, product_type, title_en, brand_en,
                main_image_id,
                1 if raw.get("bullet_point") else 0,
                1 if isinstance(raw.get("item_dimensions"), dict) else 0,
                1 if raw.get("item_weight") else 0,
                1 if raw.get("material") else 0,
                url,
            ),
        )
        listing_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        listings += 1

        for tv in _text_records(item_id, raw):
            text_batch.append((listing_id, tv[0], tv[1], tv[2], tv[3]))
        for d in _dimension_records(item_id, raw):
            dim_batch.append((listing_id, item_id, d[0], d[1], d[2], d[3]))

        if len(text_batch) >= BATCH:
            conn.executemany(insert_text, text_batch)
            text_values += len(text_batch)
            text_batch.clear()
        if len(dim_batch) >= BATCH:
            conn.executemany(insert_dim, dim_batch)
            dimension_rows += len(dim_batch)
            dim_batch.clear()

    if text_batch:
        conn.executemany(insert_text, text_batch)
        text_values += len(text_batch)
    if dim_batch:
        conn.executemany(insert_dim, dim_batch)
        dimension_rows += len(dim_batch)

    return listings, text_values, dimension_rows


def import_catalog(
    archive_root: Path,
    database: Path,
) -> dict[str, int | float]:
    """Build a new catalog beside the target and atomically replace it."""
    archive_root = archive_root.expanduser().resolve()
    database = database.expanduser().resolve()
    if not (archive_root / "listings" / "metadata").is_dir():
        raise FileNotFoundError(
            f"ABO archive not extracted at {archive_root}. "
            "Run: tar -xf abo-listings.tar -C <archive_root>."
        )
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_db = database.with_name(f".{database.name}.importing")
    temp_db.unlink(missing_ok=True)
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None
    listings = 0
    text_values = 0
    dimension_rows = 0

    try:
        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        create_schema(conn)
        listings, text_values, dimension_rows = _populate(conn, archive_root)
        conn.commit()
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if fk_errors or integrity != "ok":
            raise RuntimeError(
                f"Catalog verification failed: foreign_keys={fk_errors[:3]}, integrity={integrity}"
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
        "listings": listings,
        "text_values": text_values,
        "dimensions": dimension_rows,
        "seconds": elapsed,
        "database_bytes": database.stat().st_size,
    }


def cli() -> None:
    default_archive = Path("data/abo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--database", type=Path, default=Path("retail_catalog.db"))
    args = parser.parse_args()
    result = import_catalog(args.archive, args.database)
    print(
        f"[import] complete: {result['listings']:,} listings, "
        f"{result['text_values']:,} text values, "
        f"{result['dimensions']:,} dimensions, "
        f"{result['seconds']}s, "
        f"{result['database_bytes'] / (1024 ** 2):.1f} MiB"
    )


if __name__ == "__main__":
    cli()
