"""Probe every product_url and soft-delete (url_active=0) any that return 404.

Run once after `import_catalog.py` to clean dead links from the catalog.
Re-runnable: rows already at url_active=0 are skipped unless --include-inactive is set.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

# Allow running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import ABOCatalogRepository

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) RetailConcierge/0.1 "
    "(+https://github.com/rajasingh012/retail-concierge)"
)
TIMEOUT_SEC = 8


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        # Don't follow 3xx — Amazon redirects dead ASINs to search pages (200),
        # which would falsely flag them as alive. We want the raw ASIN status.
        # HTTPRedirectHandler default follows; we want to STOP at the first response.
    )


_OPENER = _build_opener()


def probe(item_id: str, url: str) -> tuple[str, str, int, str]:
    """Return (item_id, url, http_status, reason).

    2xx = alive. 404 = dead (soft-delete). 4xx (other) = suspect, retry as GET.
    3xx followed manually up to 5 hops — Amazon redirects dead ASINs to search
    pages which return 200 and would falsely flag them as alive, so we treat
    "redirected off the canonical ASIN path" as dead.
    """
    current_url = url
    redirect_count = 0
    while True:
        for method in ("HEAD", "GET"):
            try:
                req = urllib.request.Request(
                    current_url,
                    method=method,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                )
                with _OPENER.open(req, timeout=TIMEOUT_SEC) as r:
                    status = r.status
                    if method == "GET" and status == 200:
                        # Consume a tiny bit so the connection releases cleanly
                        r.read(64)
                    if 300 <= status < 400:
                        location = r.headers.get("Location")
                        if not location:
                            return item_id, url, status, "redirect-no-location"
                        current_url = urllib.parse.urljoin(current_url, location)
                        redirect_count += 1
                        if redirect_count > 5:
                            return item_id, url, -1, "redirect-too-many"
                        # Restart method loop with the new URL
                        break
                    # Canonical /dp/<ASIN> redirected elsewhere = probably dead ASIN
                    if redirect_count > 0 and status == 200:
                        return item_id, url, status, "redirect-then-200"
                    return item_id, url, status, "ok" if status == 200 else "non-2xx"
            except urllib.error.HTTPError as e:
                if e.code in (405, 403) and method == "HEAD":
                    # HEAD blocked, retry as GET
                    continue
                return item_id, url, e.code, e.reason or "http-error"
            except urllib.error.URLError as e:
                return item_id, url, -1, f"url-error:{e.reason}"
            except (TimeoutError, ssl.SSLError) as e:
                return item_id, url, -1, f"net-error:{type(e).__name__}"
            except Exception as e:  # noqa: BLE001 — best-effort probe
                return item_id, url, -1, f"unexpected:{type(e).__name__}"
        else:
            # HEAD and GET both failed without raising (shouldn't happen)
            return item_id, url, -1, "no-result"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./retail_catalog.db", help="catalog DB path")
    parser.add_argument("--product-type", default="", help="limit probe to one product_type (e.g. CHAIR)")
    parser.add_argument("--workers", type=int, default=32, help="concurrent probes")
    parser.add_argument("--limit", type=int, default=0, help="max URLs to probe (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="report only, don't update DB")
    parser.add_argument("--include-inactive", action="store_true", help="re-probe already-inactive rows")
    parser.add_argument("--batch-commit", type=int, default=200, help="flush DB writes every N deaths")
    args = parser.parse_args()

    repo = ABOCatalogRepository(args.db, read_only=False)
    pairs = repo.iter_product_urls(product_type=args.product_type, only_active=not args.include_inactive)
    if args.limit:
        pairs = pairs[: args.limit]
    total = len(pairs)
    print(f"Probing {total} URLs ({args.workers} workers, timeout={TIMEOUT_SEC}s)")
    if args.dry_run:
        print("DRY RUN — no DB writes")
    started = time.time()

    dead_ids: list[str] = []
    counts: Counter[tuple[int, str]] = Counter()
    done = 0
    last_log = started

    def is_alive(status: int, reason: str) -> bool:
        return status == 200 and reason == "ok"

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_id = {ex.submit(probe, iid, url): iid for iid, url in pairs}
        for fut in cf.as_completed(future_to_id):
            iid, url, status, reason = fut.result()
            counts[(status, reason)] += 1
            done += 1
            if not is_alive(status, reason):
                dead_ids.append(iid)
                if not args.dry_run and len(dead_ids) >= args.batch_commit:
                    n = repo.mark_url_inactive(dead_ids)
                    dead_ids.clear()
            if time.time() - last_log > 5:
                pct = done * 100 / total
                rate = done / max(1, time.time() - started)
                print(f"  [{done}/{total}] {pct:5.1f}%  {rate:5.1f}/s  dead-so-far={sum(c for (s, r), c in counts.items() if not is_alive(s, r))}", flush=True)
                last_log = time.time()

    if dead_ids and not args.dry_run:
        repo.mark_url_inactive(dead_ids)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.0f}s ({done / max(1, elapsed):.1f} URLs/s)")
    print("Status breakdown:")
    for (status, reason), count in counts.most_common():
        print(f"  {status:>4} {reason:<25} {count:>7}")
    alive = counts.get((200, "ok"), 0)
    dead = total - alive
    print(f"\nAlive: {alive}   Dead: {dead}   ({dead / total * 100:.1f}% dead)")
    if not args.dry_run:
        print(f"Marked {dead} listings as url_active=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
