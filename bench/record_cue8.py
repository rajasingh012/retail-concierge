"""Record cue 8: GitHub repo page -> HF model page -> back to repo, one take.

Cue 8 SRT (32.77 s): "...reproducible from the repository: one script upgrades
vLLM... quantizes with Quark... deploys... application runs as a local console
agent. We also published the quantized model on Hugging Face for anyone to
use. Everything... is open source in the repo."

Flow (matches narration beats):
  0-12s  repo landing (README: badges, quick start, scripts)  "reproducible"
  12-26s Hugging Face model page (card: W8A8, Quark, 13 GB)   "published on HF"
  26-40s back to repo landing (file tree + README)             "open source"

We record ~40 s and trim the LAST CUE_LEN seconds per the skill (page-load
warmup at the start is dropped). Viewport 1920x860 (Linux compositor gotcha).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path("/home/rajasingh/retail-concierge")
RAW = REPO / "docs" / "cues" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
WEBM = RAW / "cue8_attempt1.webm"
STILL = RAW / "cue8_attempt1_ok.png"

REPO_URL = "https://github.com/rajasingh012/retail-concierge"
HF_URL = "https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8"

VIEW = {"width": 1920, "height": 860}

with sync_playwright() as p:
    browser = p.firefox.launch(headless=False)
    context = browser.new_context(
        viewport=VIEW,
        record_video_dir=str(RAW),
        record_video_size=VIEW,
    )
    page = context.new_page()

    def show(tag: str) -> None:
        print(f"[{time.time():.1f}] {tag}", flush=True)

    # 1. Repo landing — README top (badges + quick start)
    show("goto repo")
    page.goto(REPO_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector("article.markdown-body", timeout=30000)
    time.sleep(4.0)
    show("repo README loaded, scroll 1")
    page.mouse.wheel(0, 600)   # quick start section
    time.sleep(3.5)
    page.mouse.wheel(0, 700)   # scripts / perf section
    time.sleep(3.5)
    page.mouse.wheel(0, -900)  # back up to file tree + badges
    time.sleep(2.0)

    # 2. HF model page — model card
    show("goto HF")
    page.goto(HF_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".model-card-content", timeout=30000)
    time.sleep(4.0)
    show("HF card loaded, scroll")
    page.mouse.wheel(0, 500)   # model details table (W8A8 / Quark / 13 GB)
    time.sleep(4.0)
    page.mouse.wheel(0, 600)
    time.sleep(4.0)

    # 3. Back to repo — file tree + README
    show("back to repo")
    page.goto(REPO_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector("article.markdown-body", timeout=30000)
    time.sleep(4.0)
    page.mouse.wheel(0, 300)
    time.sleep(3.0)

    page.screenshot(path=str(STILL))
    show("done, closing")
    context.close()  # flushes webm
    browser.close()

# Playwright writes webm under RAW with a generated name — find it
cands = sorted(RAW.glob("*.webm"), key=lambda p: p.stat().st_mtime)
webm = cands[-1] if cands else None
if webm and webm != WEBM:
    webm.rename(WEBM)
print("WEBM:", WEBM, WEBM.exists())
