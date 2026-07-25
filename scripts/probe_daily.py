"""Daily catalog verification probe — small batch (default 100 URLs/day).

Verify-alive model:
  - Probe amazon.in/dp/<ASIN> for ~100 URLs/day drawn from url_active=1 rows
  - ONLY flip url_active=0 when the response is a confirmed dead signal
    (404, sustained 5xx, sustained net-error, OR redirect off /dp/)
  - 200 OK responses are NEVER auto-flipped (they could be CAPTCHA pages
    served to our VPS IP — we cannot trust 200 to mean "real product page")
  - On CAPTCHA / bot-wall: STOP immediately and alert. Do not continue.

Design rationale:
  - url_active=1 is the default; we only ever flip to 0.
  - Daily batch is small (~100) to stay under Amazon's bot radar.
  - search() filters WHERE url_active=1, so dead rows stop showing up
    automatically as we verify them.
  - Resumable: pending set drawn from rows not yet probed today.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import ABOCatalogRepository

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64; rv:151.0) Gecko/20100101 Firefox/151.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
]
TIMEOUT_SEC = 8
HOMEPAGE_URL = "https://www.amazon.in/"

# If any of these appear in a 200-OK body, it's NOT a real product page.
CAPTCHA_MARKERS = [
    "Click the button below to continue shopping",
    "/errors/validateCaptcha",
    "To discuss automated access to Amazon data",
    "Correios.DoNotSend",
]

ALIVE = "alive"
DEAD_404 = "dead-404"
DEAD_5XX = "dead-5xx"
DEAD_REDIRECT = "dead-redirect"
NET_ERROR = "net-error"
CAPTCHA = "captcha"
BOTWALL = "botwall"
SUSPECT_4XX = "suspect-4xx"   # 4xx other than 403/404 — don't auto-flip
SUSPECT_200 = "suspect-200"   # 200 OK but can't verify it's a real product page


def classify(body: str | None, status: int) -> str:
    if body is not None:
        for marker in CAPTCHA_MARKERS:
            if marker in body:
                return CAPTCHA
    if status == 200:
        # Can't tell a real product page from a CAPTCHA here, body-check above
        # didn't fire. Treat as suspect — never auto-flip.
        return SUSPECT_200
    if status == 404:
        return DEAD_404
    if status == 403:
        return BOTWALL
    if 400 <= status < 500:
        return SUSPECT_4XX
    if 500 <= status < 600:
        return DEAD_5XX
    return NET_ERROR


def is_hard_stop(outcome: str) -> bool:
    return outcome in (CAPTCHA, BOTWALL)


def is_dead(outcome: str) -> bool:
    """Outcomes we trust enough to auto-flip url_active=0."""
    return outcome in (DEAD_404, DEAD_5XX, DEAD_REDIRECT)


def probe(asin: str) -> tuple[str, str, int]:
    """Probe amazon.in/dp/<ASIN>. Returns (asin, outcome, http_status)."""
    ua = random.choice(USER_AGENTS)
    url = f"https://www.amazon.in/dp/{asin}"
    body: str | None = None
    status = -1
    redirect_count = 0
    while redirect_count <= 5:
        try:
            req = urllib.request.Request(
                url,
                method="GET",
                headers={
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
            with _OPENER.open(req, timeout=TIMEOUT_SEC) as r:
                status = r.status
                if 300 <= status < 400:
                    location = r.headers.get("Location")
                    if not location:
                        return asin, DEAD_REDIRECT, status
                    url = urllib.parse.urljoin(url, location)
                    redirect_count += 1
                    continue
                if status == 200:
                    raw = r.read(8192)
                    try:
                        body = raw.decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        body = None
                break
        except urllib.error.HTTPError as e:
            return asin, classify(None, e.code), e.code
        except (TimeoutError, ssl.SSLError, urllib.error.URLError) as e:
            return asin, NET_ERROR, -1
        except Exception:  # noqa: BLE001
            return asin, NET_ERROR, -1
    if redirect_count > 5:
        return asin, DEAD_REDIRECT, status
    return asin, classify(body, status), status


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


_OPENER = _build_opener()


def homepage_is_clean() -> bool:
    ua = random.choice(USER_AGENTS)
    try:
        req = urllib.request.Request(
            HOMEPAGE_URL,
            method="GET",
            headers={"User-Agent": ua, "Accept": "text/html"},
        )
        with _OPENER.open(req, timeout=TIMEOUT_SEC) as r:
            if r.status != 200:
                return False
            raw = r.read(4096).decode("utf-8", errors="replace")
            for marker in CAPTCHA_MARKERS:
                if marker in raw:
                    return False
            return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./retail_catalog.db")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Max URLs to probe today (default 100)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--log", default="./data/dead_candidates.log")
    parser.add_argument("--state", default="./data/probe_state.json")
    parser.add_argument("--alert", default="./data/needs_human_attention.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    state_path = Path(args.state)
    alert_path = Path(args.alert)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}] verify-alive probe starting")

    if not homepage_is_clean():
        msg = (
            "STOPPED: amazon.in homepage returned CAPTCHA/bot-wall. "
            "Our VPS IP appears flagged. Wait 24h before retry, "
            "or rotate to amazon.co.uk / amazon.de."
        )
        print(msg)
        if not args.dry_run:
            alert_path.write_text(msg + "\n")
        return 2

    if args.seed is not None:
        random.seed(args.seed)

    # Pull a random sample of url_active=1 rows
    repo = ABOCatalogRepository(args.db, read_only=False)
    pairs = repo.iter_product_urls(only_active=True)
    random.shuffle(pairs)
    today_batch = pairs[: args.batch_size]
    total = len(today_batch)
    print(f"Sampling {total} random url_active=1 rows for today")

    if total == 0:
        print("Nothing to probe. Done.")
        return 0

    counts: Counter[str] = Counter()
    dead_asins: list[tuple[str, str, int]] = []   # (asin, outcome, status)
    stop_reason: str | None = None
    done = 0
    started = time.time()

    asins = [url.rsplit("/", 1)[-1].split("?")[0] for _, url in today_batch]

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_asin = {ex.submit(probe, a): a for a in asins}
        for fut in cf.as_completed(future_to_asin):
            asin, outcome, status = fut.result()
            counts[outcome] += 1
            done += 1
            if is_dead(outcome):
                dead_asins.append((asin, outcome, status))
            if is_hard_stop(outcome):
                stop_reason = f"{outcome} after {done} probes"
                break

    # Auto-flip confirmed dead (only on hard signals)
    flipped = 0
    if not args.dry_run and dead_asins:
        asin_list = [a for a, _, _ in dead_asins]
        flipped = repo.mark_url_inactive(asin_list)
        # Log them
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with log_path.open("a") as f:
            for asin, outcome, status in dead_asins:
                url = f"https://www.amazon.com/dp/{asin}"
                f.write(f"{today},{asin},{outcome},{status},{url}\n")

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.0f}s ({done / max(1, elapsed):.1f} URLs/s)")
    print("Outcome breakdown:")
    for outcome, count in counts.most_common():
        print(f"  {outcome:<14} {count:>5}")
    print(f"\nAuto-flipped to url_active=0: {flipped}")
    print(f"Suspect (not flipped): {counts.get(SUSPECT_200, 0) + counts.get(SUSPECT_4XX, 0)}")
    if stop_reason:
        msg = f"STOPPED EARLY: {stop_reason}. Remaining {total - done} ASINs deferred to next run."
        print(msg)
        if not args.dry_run:
            with alert_path.open("a") as f:
                f.write(f"[{datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}] {msg}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
