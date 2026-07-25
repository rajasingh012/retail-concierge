"""Daily catalog freshness probe.

Auto-flips listings.url_active 1->0 on hard dead signals (404, 5xx,
redirect off /dp/). Stops immediately on CAPTCHA / bot-wall. Appends every
flip to data/dead_candidates.log for grep.

Run via cron. Exit codes:
  0 = clean run
  1 = stopped early (CAPTCHA / bot-wall)
  2 = homepage pre-flight failed
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
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
LOG_PATH = Path("./data/dead_candidates.log")
ALERT_PATH = Path("./data/needs_human_attention.txt")
BATCH_SIZE = 100

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
SUSPECT_4XX = "suspect-4xx"
SUSPECT_200 = "suspect-200"


def classify(body: str | None, status: int) -> str:
    if body is not None:
        for marker in CAPTCHA_MARKERS:
            if marker in body:
                return CAPTCHA
    if status == 200:
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
    return outcome in (DEAD_404, DEAD_5XX, DEAD_REDIRECT)


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )


_OPENER = _build_opener()


def probe(asin: str) -> tuple[str, str, int]:
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
        except (TimeoutError, ssl.SSLError, urllib.error.URLError):
            return asin, NET_ERROR, -1
        except Exception:  # noqa: BLE001
            return asin, NET_ERROR, -1
    if redirect_count > 5:
        return asin, DEAD_REDIRECT, status
    return asin, classify(body, status), status


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
            return not any(m in raw for m in CAPTCHA_MARKERS)
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./retail_catalog.db")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    if not homepage_is_clean():
        msg = f"[{ts}] STOPPED: amazon.in homepage returned CAPTCHA/bot-wall. VPS IP flagged. Wait 24h."
        print(msg)
        with ALERT_PATH.open("a") as f:
            f.write(msg + "\n")
        return 2

    repo = ABOCatalogRepository(args.db, read_only=False)
    pairs = repo.iter_product_urls(only_active=True)
    random.shuffle(pairs)
    today_batch = pairs[: args.batch_size]
    print(f"[{ts}] probing {len(today_batch)} random url_active=1 rows")

    if not today_batch:
        return 0

    asins = [url.rsplit("/", 1)[-1].split("?")[0] for _, url in today_batch]
    counts: Counter[str] = Counter()
    dead_asins: list[tuple[str, str, int]] = []
    stop_reason: str | None = None
    done = 0
    started = time.time()

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

    flipped = 0
    if dead_asins:
        asin_list = [a for a, _, _ in dead_asins]
        flipped = repo.mark_url_inactive(asin_list)
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with LOG_PATH.open("a") as f:
            for asin, outcome, status in dead_asins:
                f.write(f"{today},{asin},{outcome},{status},https://www.amazon.com/dp/{asin}\n")

    elapsed = time.time() - started
    print(f"done in {elapsed:.0f}s, flipped {flipped} to url_active=0")
    for outcome, count in counts.most_common():
        print(f"  {outcome:<14} {count:>5}")
    if stop_reason:
        msg = f"[{ts}] STOPPED EARLY: {stop_reason}. {len(today_batch) - done} ASINs deferred."
        print(msg)
        with ALERT_PATH.open("a") as f:
            f.write(msg + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
