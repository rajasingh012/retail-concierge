"""End-to-end browser test of the RetailConcierge Streamlit app.

Drives the app with Playwright + Chromium (headed), runs a fixed set of
shopping queries, parses the rendered recommendation cards from the chat
HTML, and prints + saves a structured report.

Usage:
    python3 scripts/e2e_streamlit_chair.py [URL]

URL defaults to the deployed chair-only demo:
    https://retail-concierge-9fz4fe3znfxcvqiqsncwxn.streamlit.app/~/+

To test the hybrid-search flow (BM25 + sqlite-vec) locally, run the app on
the PR branch against the chairs DB (which carries the embedding index) and
pass the local URL:
    RETAIL_DB=./retail_catalog_chair.db streamlit run app.py
    python3 scripts/e2e_streamlit_chair.py http://localhost:8501

The script writes:
    bench/results/e2e_streamlit_chair.json   - structured per-query results
    bench/results/e2e_streamlit_chair.log    - human-readable run log

Each query starts a fresh AgentSession ("New Session" button), sends the
query, waits for the assistant response (up to 180s ceiling), and records:
- elapsed time
- intro prose (text before first card)
- ranked cards: rank, brand, title, ASIN, Amazon URL, pros, cons
- assumptions the agent disclosed
- any errors / timeouts

Query sets:
- SHOPPING_QUERIES: common chair intents — smoke tests the deployed demo.
- HYBRID_QUERIES: PR #1 claims — paraphrase / synonym / multilingual
  caught by vector search, structured attribute pre-filter, and the
  "red velvet" empty-result case (chair subset has zero red+velvet
  overlap). These only make sense against the chairs DB with embeddings.

This is a smoke/quality test, not a benchmark. Use the Streamlit Cloud
"Manage" button (bottom-left of the app) to view deployment logs if a
query times out or errors here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright


APP_URL = "https://retail-concierge-9fz4fe3znfxcvqiqsncwxn.streamlit.app/~/+/"

SHOPPING_QUERIES = [
    "I need an office chair for long hours at my desk",
    "Show me ergonomic chairs with lumbar support",
    "Looking for a comfy gaming chair under $300",
    "I want a leather recliner for my living room",
    "Find me a kid's chair for studying, around $100",
    "Outdoor patio chair, weather-resistant, set of 4",
]

HYBRID_QUERIES = [
    "chair for back pain",                # paraphrase -> vector catches "ergonomic"
    "couch",                              # synonym -> vector catches "sofa"
    "sofá reclinable",                    # multilingual -> vector catches reclining sofa
    "black leather chair",                # structured filter: material=leather, color=black
    "red velvet accent chair under $200", # 0 cards: catalog has no red+velvet chairs
]

PER_QUERY_TIMEOUT_SEC = 180
COLD_START_TIMEOUT_SEC = 90


@dataclass
class Card:
    rank: str = ""
    brand: str = ""
    title: str = ""
    item_id: str = ""
    url: str = ""
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass
class Result:
    query: str
    elapsed_seconds: float
    cards: list[Card] = field(default_factory=list)
    intro: str = ""
    assumptions: list[str] = field(default_factory=list)
    timed_out: bool = False
    error: str = ""


def _last_message_html(page) -> str:
    """Return the inner HTML of the last assistant message body.

    Scoped to [data-testid="stChatMessageContent"] — the stable testid
    Streamlit's own Playwright e2e suite targets. The assistant avatar is a
    SIBLING div whose Material icon glyph label ("smart_toy") would pollute
    the text; reading the body container avoids it without any element
    stripping. Falls back to the whole message element for very old
    Streamlit versions.

    Reading inner HTML (not text) preserves the <strong>, <a href>, and
    <code> tags that mark brand, title+URL, and ASIN.
    """
    msgs = page.locator('[data-testid="stChatMessage"]')
    n = msgs.count()
    if n == 0:
        return page.locator("body").inner_html(timeout=5000)
    msg_el = msgs.nth(n - 1)
    content_el = msg_el.locator('[data-testid="stChatMessageContent"]')
    if content_el.count():
        return content_el.first.inner_html(timeout=5000)
    return msg_el.inner_html(timeout=5000)


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _parse_cards(html: str) -> tuple[list[Card], str, list[str]]:
    """Parse the rendered assistant message HTML.

    Returns:
        cards:     one Card per recommendation block
        intro:     prose text before the first card (the agent's summary)
        assumps:   bullets inside the Assumptions expander
    """
    cards: list[Card] = []

    # Find every card heading in order. Streamlit renders each "### #N" as:
    #   <h3 id="N" ...><span ...>#N</span><svg/></h3>
    matches = list(re.finditer(r'<h3 id="(\d+)"[^>]*>.*?#(\d+).*?</h3>', html, re.DOTALL))

    if matches:
        pre = html[: matches[0].start()]
        intro = _strip_tags(pre)
    else:
        # No cards — the whole assistant text is prose intro.
        # Trim the trailing Assumptions/Refine blocks if present.
        body_end = len(html)
        for marker in (r"Assumptions</", r"Refine[^<]*</"):
            m = re.search(marker, html)
            if m and m.start() < body_end:
                body_end = m.start()
        intro = _strip_tags(html[:body_end])

    for i, m in enumerate(matches):
        rank = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        body = html[body_start:body_end]

        c = Card(rank=rank)
        m_brand = re.search(r"<strong>\s*([^<]+?)\s*</strong>", body)
        if m_brand:
            c.brand = m_brand.group(1).strip()
        m_link = re.search(r'<a\s+href="([^"]+)"[^>]*>\s*([^<]+?)\s*</a>', body)
        if m_link:
            c.url = m_link.group(1).strip()
            c.title = m_link.group(2).strip()
        m_id = re.search(r"<code[^>]*>\s*([A-Z0-9]{8,})\s*</code>", body)
        if m_id:
            c.item_id = m_id.group(1).strip()

        pros_block = re.search(
            r"Pros</span>.*?</h4>(.*?)(?:Trade-?offs</span>|$)", body, re.DOTALL
        )
        if pros_block:
            c.pros = [
                b.strip()
                for b in re.findall(r"<p>\s*(?:•|\u2023)\s*([^<]+?)\s*</p>", pros_block.group(1))
            ]
        cons_block = re.search(
            r"Trade-?offs</span>.*?</h4>(.*?)(?:Assumptions|Refine|$)", body, re.DOTALL
        )
        if cons_block:
            c.cons = [
                b.strip()
                for b in re.findall(r"<p>\s*(?:•|\u2023)\s*([^<]+?)\s*</p>", cons_block.group(1))
            ]
        cards.append(c)

    # Assumptions (Streamlit renders this inside an expander; content is in
    # the DOM even when collapsed).
    assumps: list[str] = []
    asm = re.search(
        r"Assumptions[^<]*</[^>]+>(.*?)(?:Refine[^<]*</[^>]+>|$)", html, re.DOTALL
    )
    if asm:
        assumps = [
            b.strip()
            for b in re.findall(r"<p>\s*(?:•|\u2023)\s*([^<]+?)\s*</p>", asm.group(1))
        ]

    return cards, intro, assumps


def _wait_for_response(page, deadline_s: float) -> bool:
    """Wait until the spinner disappears AND >= 2 chat messages exist."""
    while time.time() < deadline_s:
        try:
            spinners = page.locator("[data-testid='stSpinner']").count()
            msgs = page.locator("[data-testid='stChatMessage']").count()
            if spinners == 0 and msgs >= 2:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_query(page, query: str) -> Result:
    res = Result(query=query, elapsed_seconds=0.0)
    print(f"\n>>> {query!r}")

    # Start a fresh AgentSession via the sidebar button.
    try:
        new_btn = page.get_by_role("button", name=re.compile(r"New Session"))
        if new_btn.count():
            new_btn.first.click()
            time.sleep(1)
    except Exception:
        pass

    t0 = time.time()
    try:
        box = page.locator('[data-testid="stChatInput"] textarea').first
        box.click()
        box.fill(query)
        box.press("Enter")
        print("    sent, waiting...")

        if not _wait_for_response(page, t0 + PER_QUERY_TIMEOUT_SEC):
            res.timed_out = True
        time.sleep(2)  # let final render settle

        html = _last_message_html(page)
        cards, intro, assumps = _parse_cards(html)
        res.elapsed_seconds = round(time.time() - t0, 1)
        res.cards = cards
        res.intro = intro[:600]
        res.assumptions = assumps
        print(
            f"    done in {res.elapsed_seconds}s — {len(cards)} cards, "
            f"{len(assumps)} assumptions"
        )
    except Exception as e:
        res.elapsed_seconds = round(time.time() - t0, 1)
        res.error = f"{type(e).__name__}: {e}"
        print(f"    ERROR after {res.elapsed_seconds}s: {res.error}")

    return res


def _print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        top = r.cards[0].title if r.cards else "(none)"
        print(
            f"  {r.elapsed_seconds:>5.1f}s | cards={len(r.cards):<2} "
            f"| assumps={len(r.assumptions):<2} | err={r.error or '—'}"
        )
        print(f"        Q: {r.query!r}")
        print(f"        top: {top[:65]!r}")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else APP_URL
    # Streamlit Cloud wraps apps in an iframe at /~/+/ that auth-redirects
    # via the share URL. Bypass the wrapper by going direct.
    if "streamlit.app" in url and "/~/" not in url:
        url = url.rstrip("/") + "/~/+/"
        print(f"using iframe-direct URL: {url}")
    queries = SHOPPING_QUERIES + HYBRID_QUERIES
    results: list[Result] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        print(f"Loading {url}")
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.locator('[data-testid="stChatInput"] textarea').first.wait_for(
                timeout=COLD_START_TIMEOUT_SEC * 1000
            )
            print("Chat input visible — app loaded.")
        except PWTimeout:
            print(
                f"WARNING: chat input never appeared in {COLD_START_TIMEOUT_SEC}s. "
                f"Open {url.rsplit('/', 1)[0]} and use the 'Manage' button "
                f"(bottom-left) to view logs."
            )
        time.sleep(3)

        for q in queries:
            results.append(run_query(page, q))

        browser.close()

    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "e2e_streamlit_chair.json"
    with json_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nWrote {json_path}")

    log_path = out_dir / "e2e_streamlit_chair.log"
    with log_path.open("w") as f:
        f.write("RetailConcierge E2E chair test\n")
        f.write(f"App: {url}\n")
        f.write(f"Queries: {len(queries)}\n\n")
        for r in results:
            f.write(f"Q: {r.query}\n")
            f.write(f"  time={r.elapsed_seconds}s cards={len(r.cards)} "
                    f"timed_out={r.timed_out} err={r.error}\n")
            f.write(f"  intro: {r.intro}\n")
            for c in r.cards:
                f.write(f"  #{c.rank} {c.brand} {c.item_id} {c.title}\n")
                f.write(f"        url: {c.url}\n")
                for p in c.pros:
                    f.write(f"        pro: {p}\n")
                for c_ in c.cons:
                    f.write(f"        con: {c_}\n")
            for a in r.assumptions:
                f.write(f"  assumption: {a}\n")
            f.write("\n")
    print(f"Wrote {log_path}")

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
