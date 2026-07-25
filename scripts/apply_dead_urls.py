"""Apply manually-confirmed dead URLs to url_active=0.

Workflow:
  1. Cron logs new candidates to data/dead_candidates.log
     (format: date,item_id,asin,outcome,status,url)
  2. You open each URL in Firefox, confirm it's truly 404
  3. List the confirmed asins below (or via --asins a b c)
  4. Run this script; it flips url_active=0 for those listings
  5. To restore later: --restore --asins <...>
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
    parser.add_argument("--asins", nargs="+", default=[], help="ASINs to flip")
    parser.add_argument("--restore", action="store_true", help="set url_active=1 instead")
    parser.add_argument("--list", action="store_true", help="print candidates that are still url_active=1")
    args = parser.parse_args()

    repo = ABOCatalogRepository(args.db, read_only=False)

    if args.list:
        log = Path("data/dead_candidates.log")
        if not log.exists():
            print("No dead_candidates.log found.")
            return 1
        seen = {}
        for line in log.read_text().splitlines():
            parts = line.split(",")
            if len(parts) < 6:
                continue
            asin = parts[2]
            seen[asin] = parts[5]
        # Print only those still url_active=1 (not yet flipped)
        for asin, url in seen.items():
            r = repo._conn.execute(
                "SELECT url_active FROM listings WHERE item_id=?", (asin,)
            ).fetchone()
            if r and r[0] == 1:
                print(f"  {asin}  {url}")
        return 0

    if not args.asins:
        print("Pass --asins B07NRWJPGT B0XXXX ... or --list to view unconfirmed candidates.")
        return 1

    if args.restore:
        n = repo.mark_url_active(args.asins)
        print(f"Restored {n} listings to url_active=1")
    else:
        n = repo.mark_url_inactive(args.asins)
        print(f"Marked {n} listings as url_active=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
