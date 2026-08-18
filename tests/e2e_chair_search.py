"""
End-to-end test of the deployed Retail Concierge app.
Uses Playwright + Chromium (headed, as the user prefers) to run 6 keyword queries.

Streamlit Community Cloud cold start can take 3-5 minutes; we poll patiently.
The share URL embeds the app in a single-page React shell, so we wait for the
chat input inside the iframe / shadow DOM.
"""

import re
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout, Frame

URL = "https://share.streamlit.io/rajasingh012/retail-concierge/main/app.py"
SCREENSHOT_DIR = Path("/home/rajasingh/retail-concierge/tests/e2e_screens")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    ("office chair",         "broad recall",     "many results expected"),
    ("ergonomic chair",      "specific style",   "subset of office chairs"),
    ("IKEA",                 "brand filter",     "brand match expected"),
    ("leather chair",        "material + type",  "FTS relevance test"),
    ("bean bag",             "subcategory",      "BEAN_BAG_CHAIR rows"),
    ("xyzqwertynonsense",    "no-result case",   "graceful empty expected"),
]


def find_chat_input_in_frame(frame):
    """Look for the Streamlit chat input across selectors."""
    for sel in [
        "textarea[data-testid='stChatInputTextArea']",
        "textarea",
        "input[type='text']",
        "[contenteditable='true']",
        "[data-testid='stChatInput']",
        "form input",
        "form textarea",
        "[placeholder*='looking for' i]",
        "[placeholder*='looking' i]",
        "[aria-label*='chat' i]",
    ]:
        try:
            loc = frame.locator(sel).first
            if loc.count() > 0:
                # Try to be visible
                try:
                    if loc.is_visible():
                        return sel, loc
                except Exception:
                    return sel, loc
        except Exception:
            pass
    return None, None


def all_frames(page):
    """Yield page + every frame (descendants included)."""
    yield page
    for fr in page.frames:
        yield fr


def is_app_text(text):
    """Heuristic: does the page contain proof the Streamlit app rendered?"""
    if not text:
        return False
    t = text.lower()
    markers = [
        "retailconcierge", "amd ai devmaster",
        "what are you looking for", "new session",
        "ask me anything", "deepseek-v4",
        "2,173 products", "2,170 products",
    ]
    return any(m in t for m in markers)


def wait_for_app_ready(page, max_seconds=480):
    """Wait for the app to actually render — check every iframe too."""
    print(f"  Waiting for app render (up to {max_seconds//60} min)…", flush=True)
    deadline = time.time() + max_seconds
    last_log = 0
    while time.time() < deadline:
        # Check page + every frame
        for fr in all_frames(page):
            try:
                txt = fr.evaluate("e => e.innerText || e.textContent || ''")
            except Exception:
                txt = ""
            if is_app_text(txt):
                # Find the chat input
                sel, loc = find_chat_input_in_frame(fr)
                if sel:
                    print(f"  Found chat input in {'page' if fr is page else 'iframe'} via: {sel}")
                    return fr, sel
        if time.time() - last_log > 30:
            n = len(page.frames)
            print(f"    [{int(deadline-time.time())}s remaining] checking {n} frames…", flush=True)
            last_log = time.time()
        page.wait_for_timeout(2500)
    return None, None


def run():
    results = []
    console_msgs = []

    # ── Pre-warm with curl: hit both URLs and the iframe backend ─────────
    import urllib.request
    print(f"Pre-warming {URL} (and iframe)…", flush=True)
    try:
        # The share.streamlit.io React shell eventually mounts an iframe to
        # the actual app backend. Pre-fetching the shell warms the CDN edge.
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print(f"  Pre-warm error (non-fatal): {e}")
    # Also poke the private streamlit.app URL — this might be the real deployed entry
    try:
        req = urllib.request.Request(
            "https://retail-concierge-9fz4fe3znfxcvqiqsncwxn.streamlit.app/",
            headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print(f"  streamlit.app pre-warm error (non-fatal): {e}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.on("console", lambda m: console_msgs.append(f"[{m.type}] {m.text[:200]}"))
        page.on("pageerror", lambda e: console_msgs.append(f"[pageerror] {str(e)[:200]}"))

        print(f"\n=== Navigating to {URL} ===", flush=True)
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        except Exception as e:
            print(f"  goto warning: {e}")

        ctx_obj, sel = wait_for_app_ready(page, max_seconds=420)
        if not ctx_obj:
            page.screenshot(path=str(SCREENSHOT_DIR / "TIMEOUT.png"), full_page=True)
            print("TIMEOUT — chat input never appeared. See TIMEOUT.png")
            browser.close()
            return results

        # Initial render screenshot
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SCREENSHOT_DIR / "00_initial.png"), full_page=False)

        for idx, (kw, intent, hint) in enumerate(QUERIES, 1):
            print(f"\n--- Query {idx}/6: {kw!r}  ({intent}) ---", flush=True)
            t0 = time.time()

            # Re-locate each time (Streamlit re-renders)
            # Re-locate each time (Streamlit re-renders DOM after submit)
            ctx_obj, sel = wait_for_app_ready(page, max_seconds=60)
            if not ctx_obj or not sel:
                print("  ! Lost the chat input — skipping")
                results.append({"kw": kw, "intent": intent, "hint": hint,
                                "elapsed_s": 0, "user_msgs": 0,
                                "count_text": None, "n_product_links": 0,
                                "shot": "(none)", "error": "lost input"})
                continue

            inp = ctx_obj.locator(sel).first
            try:
                inp.click(timeout=10_000)
                inp.fill("")
                inp.fill(kw, timeout=10_000)
                inp.press("Enter")
            except Exception as e:
                print(f"  ! Input/Enter failed: {e}")
                results.append({"kw": kw, "intent": intent, "hint": hint,
                                "elapsed_s": time.time() - t0, "user_msgs": 0,
                                "count_text": None, "n_product_links": 0,
                                "shot": "(none)", "error": f"input: {e}"})
                continue

            # Wait for an assistant message
            print("  Waiting for assistant reply…", flush=True)
            try:
                # Poll for any of these success signals in page + iframes
                def got_reply():
                    for fr in all_frames(page):
                        try:
                            n_msgs = fr.locator("[data-testid='stChatMessage']").count()
                            if n_msgs >= 2:
                                return True
                        except Exception:
                            pass
                        try:
                            t = (fr.evaluate("e => e.innerText || e.textContent || ''") or "").lower()
                        except Exception:
                            t = ""
                        if "no results" in t or "i couldn't find" in t or "found" in t:
                            return True
                    return False
                deadline = time.time() + 180
                while time.time() < deadline:
                    if got_reply():
                        break
                    page.wait_for_timeout(2500)
                else:
                    print("  ! No assistant response within 180s")
            except Exception as e:
                print(f"  ! Reply wait error: {e}")

            # Let the agent finish responding (long tool-call chains possible)
            page.wait_for_timeout(8000)

            elapsed = round(time.time() - t0, 1)

            # Aggregate results across all frames
            user_msgs = 0
            full_text = ""
            n_links = 0
            for fr in all_frames(page):
                try:
                    user_msgs += fr.locator("[data-testid='stChatMessage']").count()
                except Exception:
                    pass
                try:
                    txt = (fr.evaluate("e => e.innerText || e.textContent || ''") or "")
                    full_text += "\n" + txt
                except Exception:
                    pass
                try:
                    n_links += fr.locator(
                        "a[href*='/dp/'], a[href*='amazon'], a[href*='primenow'], "
                        "a[href*='item_id'], a[target='_blank']"
                    ).count()
                except Exception:
                    pass

            page_text = full_text.lower()
            m = re.search(r"(\d+)\s+(results|chairs?|listings?|products?|matches?|items?|found)", page_text)
            count_text = m.group(0) if m else None

            shot = SCREENSHOT_DIR / f"q{idx:02d}_{kw.replace(' ', '_')}.png"
            try:
                page.screenshot(path=str(shot), full_page=False)
            except Exception as e:
                print(f"  screenshot failed: {e}")

            print(f"  elapsed:        {elapsed}s")
            print(f"  chat msgs:      {user_msgs}")
            print(f"  parsed count:   {count_text or '(not parsed)'}")
            print(f"  product links:  {n_links}")
            print(f"  screenshot:     {shot}")

            results.append({
                "kw": kw, "intent": intent, "hint": hint,
                "elapsed_s": elapsed, "user_msgs": user_msgs,
                "count_text": count_text, "n_product_links": n_links,
                "shot": str(shot),
            })

        (SCREENSHOT_DIR / "console.log").write_text("\n".join(console_msgs))
        print(f"\nConsole log saved → {SCREENSHOT_DIR/'console.log'}  ({len(console_msgs)} msgs)")
        browser.close()

    return results


def verdict_table(results):
    print("\n\n================ RESULTS TABLE ================\n")
    headers = ["#", "Keyword", "Intent", "Elapsed", "Results found", "Prod links"]
    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            str(i), r["kw"], r["intent"],
            f"{r['elapsed_s']}s",
            r["count_text"] or "(not parsed)",
            str(r["n_product_links"]),
        ])
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(f"{c:<{w}}" for c, w in zip(row, widths)) + " |")
    print(sep)


def quality_verdict(results):
    print("\n\n================ QUALITY VERDICT ================\n")
    if not results:
        print("No results captured — app never became ready.")
        return

    for i, r in enumerate(results, 1):
        kw = r["kw"]
        elapsed = r["elapsed_s"]
        n = r["n_product_links"]
        ct = r["count_text"]
        user_msgs = r["user_msgs"]
        err = r.get("error")

        if err:
            grade, msg = "ERROR", err
        elif "nonsense" in kw or "xyz" in kw:
            ok = (n == 0) or (ct is not None and ct.split()[0] == "0") or user_msgs >= 2
            grade = "PASS" if ok else "FAIL"
            msg = "Empty state handled" if ok else "Junk keyword misbehaved"
        else:
            if n >= 5:
                grade, msg = "PASS", f"{n} product links rendered"
            elif n >= 1:
                grade, msg = "WEAK", f"only {n} links"
            elif ct and int(ct.split()[0]) > 0:
                grade, msg = "PASS-TEXT", f"{ct} (links not in DOM)"
            else:
                grade, msg = "FAIL", "no product results visible"

        lat = "OK" if elapsed < 20 else ("SLOW" if elapsed < 40 else "TOO SLOW")

        print(f"[{i}] {kw!r:<22}  intent={r['intent']:<18}  "
              f"latency={lat:<8}  rows={n:<4}  → {grade}  ({msg})")

    print("\nKey:")
    print("  PASS      - relevant product links rendered")
    print("  WEAK      - some results but few")
    print("  PASS-TEXT - text said results but links not detectable in DOM")
    print("  FAIL      - no results shown for a query that should match the chair DB")
    print()
    print("Screenshots per query are in tests/e2e_screens/")


if __name__ == "__main__":
    try:
        results = run()
        verdict_table(results)
        quality_verdict(results)
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
