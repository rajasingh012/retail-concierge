"""Mark listings as verified-active after you've confirmed them in Firefox.

Verify-alive model: url_active=1 means "a real browser session saw a real
product page here." This script flips 0→1 for ASINs you've personally checked.

Common workflows:
  - Pre-demo: pull a random sample, open each in Firefox, mark the keepers.
  - Post-cron: review data/dead_candidates.log, restore any false positives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import ABOCatalogRepository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./retail_catalog.db")
    parser.add_argument("--asins", nargs="+", default=[], help="ASINs to mark active")
    parser.add_argument("--deactivate", action="store_true",
                        help="Reverse: set url_active=0 (mark dead)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N random inactive ASINs with their URLs")
    parser.add_argument("--product-type", default="",
                        help="Filter --sample to a single product_type")
    args = parser.parse_args()

    repo = ABOCatalogRepository(args.db, read_only=False)

    if args.sample:
        # Print random inactive rows for manual Firefox verification
        sql = "SELECT item_id, product_type, product_url FROM listings WHERE url_active=0"
        params: list[object] = []
        if args.product_type:
            sql += " AND product_type=?"
            params.append(args.product_type)
        rows = repo._conn.execute(sql, params).fetchall()
        import random
        random.shuffle(rows)
        for r in rows[: args.sample]:
            print(f"  {r[0]}  [{r[1]}]  {r[2]}")
        print(f"\n(Showing {min(args.sample, len(rows))} of {len(rows)} inactive rows)")
        return 0

    if not args.asins:
        print("Pass --asins B0XXX B0YYY ... or --sample N to list inactive rows.")
        return 1

    if args.deactivate:
        n = repo.mark_url_inactive(args.asins)
        print(f"Deactivated {n} listings (url_active=0)")
    else:
        n = repo.mark_url_active(args.asins)
        print(f"Activated {n} listings (url_active=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
