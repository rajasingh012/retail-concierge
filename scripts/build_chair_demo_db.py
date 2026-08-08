"""Build a single-category SQLite demo catalog for free-tier hosting.

Filters the full ABO catalog to one or more product_types, copies their
text_values and dimensions, rebuilds the FTS5 index against the slimmed
listings table, then VACUUMs. Output is ~5-30 MB and fits in any free-tier
deploy target (Streamlit Community Cloud, HF Spaces, GitHub repo).

By default the script keeps CHAIR + BEAN_BAG_CHAIR listings (the category
chosen for the RetailConcierge VP demo). Use --product-types to filter to
any other category, or --list-top to see what's available before picking.

Usage:
    # Default (chairs, the VP demo category)
    uv run python scripts/build_chair_demo_db.py \\
        --source retail_catalog.db \\
        --output retail_catalog_chair.db

    # See the top 10 largest categories in the catalog
    uv run python scripts/build_chair_demo_db.py \\
        --source retail_catalog.db \\
        --list-top 10

    # Build a demo DB for a different category
    uv run python scripts/build_chair_demo_db.py \\
        --source retail_catalog.db \\
        --output retail_catalog_shoes.db \\
        --product-types SHOES BOOT SANDAL
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import create_schema, migrate

DEFAULT_PRODUCT_TYPES = ("CHAIR", "BEAN_BAG_CHAIR")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", help="Path to the full ABO SQLite catalog")
    p.add_argument(
        "--output",
        help="Path to write the slim demo catalog (required unless --list-top is set)",
    )
    p.add_argument(
        "--product-types",
        nargs="+",
        default=list(DEFAULT_PRODUCT_TYPES),
        help="Product types to keep (default: CHAIR BEAN_BAG_CHAIR)",
    )
    p.add_argument(
        "--list-top",
        type=int,
        metavar="N",
        help="List the top N product_types by listing count and exit (no DB written). "
        "Use this to pick a category before running a real build.",
    )
    args = p.parse_args()
    if args.list_top is None and (args.source is None or args.output is None):
        p.error("--source and --output are required unless --list-top is set")
    return args


def list_top_categories(source_path: Path, n: int) -> None:
    """Print the top N product_types in the catalog, ordered by listing count."""
    conn = sqlite3.connect(str(source_path))
    rows = list(
        conn.execute(
            "SELECT product_type, COUNT(*) AS n "
            "FROM listings WHERE product_type IS NOT NULL "
            "GROUP BY product_type ORDER BY n DESC LIMIT ?",
            (n,),
        )
    )
    conn.close()
    if not rows:
        print("(no product_types found in catalog)", file=sys.stderr)
        return
    width = max(len(row[0]) for row in rows)
    print(f"Top {len(rows)} product_types in {source_path}:")
    for row in rows:
        print(f"  {row[1]:>7}  {row[0]}")
    print(
        f"\nUse --product-types <NAME>... to build a demo DB for any of these.",
        file=sys.stderr,
    )


def copy_subset(source_path: Path, output_path: Path, product_types: list[str]) -> int:
    """Copy filtered listings + referencing rows to a fresh demo DB.

    Returns the number of listings copied.
    """
    if output_path.exists():
        output_path.unlink()

    dst = sqlite3.connect(str(output_path))
    create_schema(dst, rebuild_fts=False)
    migrate(dst)

    # Single connection: ATTACH source, do everything via INSERT...SELECT,
    # DETACH, then VACUUM. Two open connections on the same file acquire
    # conflicting locks under WAL and DETACH fails with "database is locked".
    dst.execute("ATTACH DATABASE ? AS src", (str(source_path),))

    placeholders = ",".join("?" * len(product_types))
    # Stage subset IDs in a temp table on the dst side, then INSERT...SELECT
    # using a join. The temp table breaks the open read transaction on src,
    # so we can commit + DETACH cleanly. Two open connections on the same
    # file acquire conflicting locks under WAL and DETACH fails with
    # "database is locked".
    dst.execute("CREATE TEMP TABLE _subset_ids (id INTEGER PRIMARY KEY)")
    dst.execute(
        f"INSERT INTO _subset_ids(id) "
        f"SELECT id FROM src.listings WHERE product_type IN ({placeholders}) "
        f"ORDER BY id",
        product_types,
    )
    n_subset = dst.execute("SELECT COUNT(*) FROM _subset_ids").fetchone()[0]
    if n_subset == 0:
        dst.execute("DETACH DATABASE src")
        dst.close()
        raise RuntimeError(f"No listings matched product_types={product_types}")

    dst.execute(
        "INSERT INTO main.listings "
        "SELECT l.* FROM src.listings l JOIN _subset_ids s ON s.id = l.id"
    )
    dst.execute(
        "INSERT INTO main.listing_text_values "
        "SELECT t.* FROM src.listing_text_values t "
        "JOIN _subset_ids s ON s.id = t.listing_id"
    )
    dst.execute(
        "INSERT INTO main.listing_dimensions "
        "SELECT d.* FROM src.listing_dimensions d "
        "JOIN _subset_ids s ON s.id = d.listing_id"
    )
    dst.execute("DROP TABLE _subset_ids")
    dst.commit()  # releases the read lock on src before DETACH
    dst.execute("DETACH DATABASE src")
    dst.commit()

    # Rebuild FTS5 against the slimmed listings. Triggers fired during the
    # INSERTs already populated listing_fts, but rebuilding is the safe
    # canonical form (matches create_schema(rebuild_fts=True)).
    dst.execute("INSERT INTO listing_fts(listing_fts) VALUES ('rebuild')")
    dst.commit()

    n_listings = dst.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    n_text = dst.execute("SELECT COUNT(*) FROM listing_text_values").fetchone()[0]
    n_dim = dst.execute("SELECT COUNT(*) FROM listing_dimensions").fetchone()[0]
    print(
        f"copied listings={n_listings} text_values={n_text} dimensions={n_dim}",
        file=sys.stderr,
    )

    dst.execute("VACUUM")
    dst.execute("ANALYZE")
    dst.close()
    return n_listings


def verify_demo_db(output_path: Path, expected_types: list[str]) -> None:
    """Sanity-check the demo DB: schema, FTS5 index, row counts."""
    conn = sqlite3.connect(str(output_path))
    n = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    assert n > 0, f"empty listings table in {output_path}"

    distinct_types = {
        row[0]
        for row in conn.execute("SELECT DISTINCT product_type FROM listings")
    }
    unexpected = distinct_types - set(expected_types)
    assert not unexpected, f"unexpected product_types in demo DB: {unexpected}"

    # Every row in listing_dimensions / listing_text_values references an
    # existing listings row. (No FK-enforced referential integrity, but the
    # demo DB must be self-consistent — orphan rows would crash ranking.)
    n_orphan = conn.execute(
        "SELECT COUNT(*) FROM listing_dimensions d "
        "WHERE NOT EXISTS (SELECT 1 FROM listings l WHERE l.id = d.listing_id)"
    ).fetchone()[0]
    assert n_orphan == 0, f"{n_orphan} orphan rows in listing_dimensions"

    n_orphan = conn.execute(
        "SELECT COUNT(*) FROM listing_text_values t "
        "WHERE NOT EXISTS (SELECT 1 FROM listings l WHERE l.id = t.listing_id)"
    ).fetchone()[0]
    assert n_orphan == 0, f"{n_orphan} orphan rows in listing_text_values"

    # FTS5 sanity check: use the longest product_type as the search term.
    # For CHAIR/BEAN_BAG_CHAIR this is "BEAN_BAG_CHAIR" or "CHAIR" itself,
    # both of which match real listings. For any other category this picks
    # the most-specific token and still matches real rows.
    fts_term = max(expected_types, key=len)
    n_listings = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    n_fts = conn.execute(
        "SELECT COUNT(*) FROM listing_fts WHERE listing_fts MATCH ?",
        (fts_term,),
    ).fetchone()[0]
    assert n_fts > 0, f"FTS5 index returned 0 hits for {fts_term!r}"
    print(
        f"verify OK: {n_listings} listings, "
        f"FTS5 hits for {fts_term!r} = {n_fts}, "
        f"zero orphan rows in referencing tables",
        file=sys.stderr,
    )
    conn.close()


def main() -> int:
    args = parse_args()
    if args.list_top is not None:
        if args.source is None:
            print("--source is required with --list-top", file=sys.stderr)
            return 1
        source = Path(args.source).resolve()
        if not source.exists():
            print(f"source DB not found: {source}", file=sys.stderr)
            return 1
        list_top_categories(source, args.list_top)
        return 0

    if args.source is None or args.output is None:
        print("--source and --output are required for a build", file=sys.stderr)
        return 1
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.exists():
        print(f"source DB not found: {source}", file=sys.stderr)
        return 1

    t0 = time.monotonic()
    n = copy_subset(source, output, args.product_types)
    verify_demo_db(output, args.product_types)
    size_mb = output.stat().st_size / (1024 * 1024)
    elapsed = time.monotonic() - t0
    types_label = "+".join(args.product_types)
    print(
        f"wrote {output} ({n} listings [{types_label}], {size_mb:.1f} MB) in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())