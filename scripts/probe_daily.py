"""Daily catalog freshness probe — top product types against amazon.in.

Design principles:
  - Probes amazon.in/dp/<ASIN> (cleaner VPS edge than amazon.com)
  - Probes only the top product types (~25k rows, not the full 145k catalog)
  - Stops immediately on CAPTCHA / bot-wall signals (do NOT continue when flagged)
  - Resumable: picks up from where yesterday's run left off
  - Outputs candidates for HUMAN verification only — never auto-flips url_active
  - No last_verified_at column (per design choice)

Cron:  runs weekday mornings. Reads progress from data/dead_candidates.log.
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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]
TIMEOUT_SEC = 8

# Hard-stop markers. If any of these appear in the response body, this is NOT
# a real product page and we should NOT classify the ASIN based on it.
CAPTCHA_MARKERS = [
    "Click the button below to continue shopping",
    "/errors/validateCaptcha",
    "To discuss automated access to Amazon data",
    "Correios.DoNotSend",
]
HOMEPAGE_URL = "https://www.amazon.in/"


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


_OPENER = _build_opener()

# Outcome codes
ALIVE = "alive"
DEAD_404 = "dead-404"
DEAD_4XX = "dead-4xx"
DEAD_5XX = "dead-5xx"
DEAD_REDIRECT = "dead-redirect"
NET_ERROR = "net-error"
CAPTCHA = "captcha"
BOTWALL = "botwall"


def classify(body: str | None, status: int) -> str:
    """Body-aware classifier. None means body wasn't read (HEAD or bot-wall short-circuit)."""
    if body is not None:
        for marker in CAPTCHA_MARKERS:
            if marker in body:
                return CAPTCHA
    if status == 200:
        return ALIVE
    if status == 404:
        return DEAD_404
    if status == 403:
        return BOTWALL
    if 400 <= status < 500:
        return DEAD_4XX
    if 500 <= status < 600:
        return DEAD_5XX
    return NET_ERROR


def probe(item_id: str, asin: str) -> tuple[str, str, str, int]:
    """Probe amazon.in/dp/<ASIN>. Returns (item_id, asin, outcome, http_status).

    Always reads body for CAPTCHA classification when status is 200.
    """
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
                        return item_id, asin, DEAD_REDIRECT, status
                    url = urllib.parse.urljoin(url, location)
                    redirect_count += 1
                    continue
                if status == 200:
                    # Read enough to detect CAPTCHA markers but not the whole page
                    raw = r.read(8192)
                    try:
                        body = raw.decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        body = None
                break
        except urllib.error.HTTPError as e:
            return item_id, asin, classify(None, e.code), e.code
        except (TimeoutError, ssl.SSLError, urllib.error.URLError) as e:
            return item_id, asin, NET_ERROR, -1
        except Exception as e:  # noqa: BLE001
            return item_id, asin, NET_ERROR, -1

    if redirect_count > 5:
        return item_id, asin, DEAD_REDIRECT, status

    outcome = classify(body, status)
    return item_id, asin, outcome, status


def is_hard_stop(outcome: str) -> bool:
    """Outcomes that mean our IP is being challenged and we MUST stop."""
    return outcome in (CAPTCHA, BOTWALL)


def homepage_is_clean() -> bool:
    """Quick pre-flight: hit amazon.in homepage. If we get CAPTCHA, our IP is flagged."""
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
    parser.add_argument("--product-types", nargs="+", default=[
        "CHAIR", "SHOES", "BOOT", "SANDAL", "HOME", "GROCERY",
        "HOME_BED_AND_BATH", "CELLULAR_PHONE_CASE",
    ])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-urls", type=int, default=0, help="0 = all pending")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-min", type=float, default=1.0)
    parser.add_argument("--sleep-max", type=float, default=3.0)
    parser.add_argument("--log", default="./data/dead_candidates.log")
    parser.add_argument("--state", default="./data/probe_state.json")
    parser.add_argument("--alert", default="./data/needs_human_attention.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    state_path = Path(args.state)
    alert_path = Path(args.alert)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}] daily probe starting")

    if not homepage_is_clean():
        msg = (
            "STOPPED: amazon.in homepage returned CAPTCHA/bot-wall. "
            "Our IP appears to be flagged. Wait 24h before next attempt, "
            "or rotate to amazon.co.uk / amazon.de."
        )
        print(msg)
        alert_path.write_text(msg + "\n")
        return 2

    # Load prior state to resume
    state = {"probed_asins": []}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            pass
    probed_set = set(state.get("probed_asins", []))

    repo = ABOCatalogRepository(args.db, read_only=True)
    pending: list[tuple[str, str]] = []
    for pt in args.product_types:
        for iid, url in repo.iter_product_urls(product_type=pt, only_active=True):
            asin = url.rsplit("/", 1)[-1].split("?")[0]
            if asin and asin not in probed_set:
                pending.append((iid, asin))
    random.shuffle(pending)
    if args.max_urls:
        pending = pending[: args.max_urls]
    total = len(pending)
    print(f"Probing {total} ASINs across {len(args.product_types)} product types ({args.workers} workers)")

    if total == 0:
        print("Nothing pending. Run done.")
        return 0

    counts: Counter[str] = Counter()
    candidates: list[tuple[str, str, str, int]] = []  # (date, item_id, asin, status)
    stop_reason: str | None = None
    done = 0
    started = time.time()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_pair = {ex.submit(probe, iid, asin): (iid, asin) for iid, asin in pending}
        for fut in cf.as_completed(future_to_pair):
            iid, asin, outcome, status = fut.result()
            counts[outcome] += 1
            done += 1
            probed_set.add(asin)
            if outcome in (DEAD_404, DEAD_4XX, DEAD_5XX, DEAD_REDIRECT):
                candidates.append((iid, asin, outcome, status))
            if is_hard_stop(outcome):
                stop_reason = f"{outcome} on {asin} after {done} probes"
                break

    # Save state so next run resumes from here
    if not args.dry_run:
        state_path.write_text(json.dumps({"probed_asins": sorted(probed_set)}, indent=2))

    # Append candidates to log (dedupe by asin)
    new_candidates = 0
    if not args.dry_run and candidates:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        existing = log_path.read_text() if log_path.exists() else ""
        seen_asins = {line.split(",")[1] for line in existing.splitlines() if line.count(",") >= 3}
        with log_path.open("a") as f:
            for iid, asin, outcome, status in candidates:
                if asin in seen_asins:
                    continue
                url = f"https://www.amazon.com/dp/{asin}"
                f.write(f"{today},{iid},{asin},{outcome},{status},{url}\n")
                new_candidates += 1

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.0f}s ({done / max(1, elapsed):.1f} URLs/s)")
    print("Outcome breakdown:")
    for outcome, count in counts.most_common():
        print(f"  {outcome:<14} {count:>6}")
    print(f"\nNew candidates logged: {new_candidates}  (append-only, never auto-flipped)")
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
