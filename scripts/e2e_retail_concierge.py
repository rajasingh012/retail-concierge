"""E2E browser harness for the RetailConcierge Streamlit UI (Playwright + Chromium).

Opens the app (local or deployed), runs a fixed set of queries, parses the
rendered DOM into structured results, and writes:

  bench/results/e2e_retail_concierge.json   — structured per-query results
  bench/results/e2e_retail_concierge.log    — human-readable summary

Usage:
  python3 scripts/e2e_retail_concierge.py [URL]

Defaults to http://localhost:8501 (local run against the chair demo DB, which
carries the sqlite-vec embedding index). Requires: pip install playwright
and `playwright install chromium`.

Queries cover the PR #1 hybrid-search claims: BM25 baseline, paraphrase
(→vector), synonym (→vector), multilingual (→vector), and structured
attribute pre-filter (color/material via listing_text_values LIKE).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URL = "http://localhost:8501"

QUERIES = [
    "office chair with lumbar support",   # BM25 baseline (keyword hit)
    "chair for back pain",                # paraphrase -> vector catches "ergonomic"
    "couch",                              # synonym -> vector catches "sofa"
    "sofá reclinable",                    # multilingual -> vector catches reclining sofa
    "black leather chair",                # structured filter: material=leather, color=black
    "grey mesh chair",                    # structured filter: material=mesh, color=grey
    "wooden chair",                       # structured filter: material=wood
    "red velvet accent chair under $200", # should return 0 cards (no red+velvet overlap)
]

PER_QUERY_TIMEOUT_S = 240
SETTLE_S = 2


def _cards_from_html(html: str) -> list[dict]:
    """Parse product cards out of one assistant message's inner HTML.

    Streamlit's DOM is stable enough for regex (verified Aug 2026 against the
    chair-only demo). Card bodies live between consecutive h3 anchors.
    """
    anchors = list(re.finditer(r'<h3 id="(\d+)"[^>]*>.*?#(\d+).*?</h3>', html, re.S))
    if not anchors:
        return []
    cards: list[dict] = []
    for i, m in enumerate(anchors):
        start = m.end()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(html)
        block = html[start:end]
        brand_m = re.search(r"<strong>\s*([^<]+?)\s*</strong>", block)
        link_m = re.search(r'<a\s+href="([^"]+)"[^>]*>\s*([^<]+?)\s*</a>', block)
        id_m = re.search(r"<code[^>]*>\s*([A-Z0-9]{8,})\s*</code>", block)
        card: dict = {
            "rank": m.group(2),
            "brand": brand_m.group(1).strip() if brand_m else "",
            "title": link_m.group(2).strip() if link_m else "",
            "url": link_m.group(1) if link_m else "",
            "item_id": id_m.group(1) if id_m else "",
        }
        # Pros: anchored on "Pros" heading, capture until Trade-offs heading.
        pros_m = re.search(r"Pros</span>.*?</h4>(.*?)(?:Trade-?offs</span>|$)", block, re.S)
        if pros_m:
            card["pros"] = [p.strip() for p in re.findall(r"<p[^>]*>•\s*([^<]+)", pros_m.group(1))]
        else:
            card["pros"] = []
        # Cons: anchored on "Trade-offs" heading, capture until Assumptions/Refine.
        cons_m = re.search(r"Trade-?offs</span>.*?</h4>(.*?)(?:Assumptions|Refine|$)", block, re.S)
        if cons_m:
            card["cons"] = [p.strip() for p in re.findall(r"<p[^>]*>•\s*([^<]+)", cons_m.group(1))]
        else:
            card["cons"] = []
        cards.append(card)
    return cards


def _strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def run_query(page, query: str) -> dict:
    # Fresh session: clear chat history via the sidebar button.
    page.get_by_role("button", name="New Session").first.click()
    page.wait_for_timeout(800)

    chat = page.locator('[data-testid="stChatInput"] textarea').first
    chat.fill(query)
    chat.press("Enter")

    started = time.time()
    deadline = started + PER_QUERY_TIMEOUT_S
    while time.time() < deadline:
        spinners = page.locator("[data-testid='stSpinner']").count()
        msgs = page.locator("[data-testid='stChatMessage']").count()
        if spinners == 0 and msgs >= 2:
            break
        time.sleep(2)
    timed_out = time.time() >= deadline
    time.sleep(SETTLE_S)  # let the final rerun land

    # Expand any collapsed expanders (Assumptions) in the last message.
    last = page.locator("[data-testid='stChatMessage']").nth(-1)
    for summary in last.locator("details summary").all():
        try:
            summary.click()
        except Exception:
            pass
    time.sleep(0.5)

    html = last.inner_html()
    cards = _cards_from_html(html)
    if cards:
        first_anchor = re.search(r'<h3 id="\d+"', html)
        intro_html = html[: first_anchor.start()] if first_anchor else ""
        intro = _strip_tags(intro_html)
    else:
        intro = _strip_tags(html)
        for marker in ("Assumptions", "Refine"):
            idx = intro.find(marker)
            if idx != -1:
                intro = intro[:idx].rstrip()
    assumptions: list[str] = []
    am = re.search(r"Assumptions</(?:span|strong)[^>]*>(.*?)(?:Refine|$)", html, re.S)
    if am:
        assumptions = [s.strip() for s in re.findall(r"<p[^>]*>•\s*([^<]+)", am.group(1))]

    return {
        "query": query,
        "elapsed_seconds": round(time.time() - started, 1),
        "timed_out": timed_out,
        "cards": cards,
        "intro": intro[:600],
        "assumptions": assumptions,
    }


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    results: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        # Wait for the chat input to mount.
        page.locator('[data-testid="stChatInput"] textarea').first.wait_for(timeout=120_000)
        print(f"app loaded at {url}", flush=True)
        for q in QUERIES:
            print(f"query: {q}", flush=True)
            try:
                r = run_query(page, q)
            except Exception as exc:  # noqa: BLE001 — one bad query must not kill the run
                r = {"query": q, "elapsed_seconds": 0, "timed_out": True,
                     "cards": [], "intro": "", "assumptions": [], "error": str(exc)}
            results.append(r)
            print(f"  -> {len(r['cards'])} cards in {r['elapsed_seconds']}s", flush=True)
        browser.close()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    json_path = RESULTS_DIR / "e2e_retail_concierge.json"
    log_path = RESULTS_DIR / "e2e_retail_concierge.log"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    lines = [f"RetailConcierge E2E run — {stamp}", f"URL: {url}", ""]
    lines.append(f"{'query':<38} {'cards':>5} {'secs':>6}  status")
    for r in results:
        status = "TIMEOUT" if r.get("timed_out") else ("ERROR: " + r.get("error", "") if r.get("error") else "ok")
        lines.append(f"{r['query'][:38]:<38} {len(r['cards']):>5} {r['elapsed_seconds']:>6.1f}  {status}")
        for c in r["cards"]:
            lines.append(f"    #{c['rank']} {c['brand']} | {c['title'][:60]} | {c['item_id']}")
    lines.append("")
    for r in results:
        if r["cards"]:
            lines.append(f"[{r['query']}] intro: {r['intro'][:150]}")
    log_path.write_text("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
