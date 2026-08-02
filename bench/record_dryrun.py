"""Dry-run: open Streamlit in Playwright Firefox, type the earbuds query,
capture a screenshot. Used to verify the toolchain before the full demo.

Run: cd /home/rajasingh/retail-concierge && uv run python bench/record_dryrun.py
"""
from __future__ import annotations
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("/home/rajasingh/retail-concierge/docs/cues/raw")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = "a pair of wireless earbuds under 5k rupees"

with sync_playwright() as pw:
    browser = pw.firefox.launch(headless=False)
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    print("[1] navigate")
    page.goto("http://localhost:8501/", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_selector("textarea[placeholder='What are you looking for?']", timeout=30000)
    page.screenshot(path=str(OUT / "dryrun_01_load.png"))
    print("[2] type query")
    page.locator("textarea[placeholder='What are you looking for?']").fill(PROMPT)
    page.screenshot(path=str(OUT / "dryrun_02_typed.png"))
    page.keyboard.press("Enter")
    print("[3] wait for agent response (~5-15s)")
    try:
        page.wait_for_selector("text=Refine your search", timeout=60000)
        print("  - recommendation rendered (Refine chips present)")
    except Exception as e:
        print(f"  - TIMEOUT: agent didn't finish in 60s ({e})")
        page.screenshot(path=str(OUT / "dryrun_03_timeout.png"))
        raise
    page.screenshot(path=str(OUT / "dryrun_03_results.png"))
    print("[4] inspect page text")
    page_text = page.inner_text("body")
    print(f"  body length: {len(page_text)} chars")
    # Show a snippet with the agent reply
    import re
    for line in page_text.splitlines()[:60]:
        if line.strip():
            print(f"  | {line[:120]}")
    print("\n[5] close")
    context.close()
    browser.close()
    print("Done.")
