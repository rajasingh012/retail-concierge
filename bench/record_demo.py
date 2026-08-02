"""Drive the RetailConcierge Streamlit UI in Firefox + record the demo as MP4.

Why Playwright (not ffmpeg+xdotool):
- Full programmatic control over the browser (type, click, wait for selectors)
- No focus juggling on the desktop
- Records a clean bordered browser window — chat terminal / IDE etc. are not in the capture
- Single tool does both "drive the UI" and "what gets filmed" (Playwright's
  built-in video recording, which we then mux with the narration audio)

Prereqs:
- app.py running on localhost:8501 with RETAIL_BASE_URL pointing at the MI300X droplet
  (we launched it that way; see /tmp/streamlit.log)
- Firefox Playwright binary installed (uv run playwright install firefox)

What it does per cue:
- cue 2: type "a pair of wireless earbuds under 5k rupees" → wait for agent → screenshot cards
- cue 3: type "I want something for my trip." → wait for clarifying cards → screenshot
- cue 4: type "stainless steel water bottle 1L" → wait → click first chip → screenshot
- cue 8: navigate to github.com/rajasingh012/retail-concierge → screenshot repo

For each cue, save a separate webm (raw video) + the final cue mp4 (muxed with audio).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
REPO = HERE.parent
VIDEO_DIR = REPO / "docs" / "cues" / "raw"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO = REPO / "docs" / "demo_narration.mp3"
SRT = REPO / "docs" / "demo_subtitles.srt"

# Each cue: (cue_index, browser action sequence).
# Times are wall-clock targets relative to the audio narration start.
CUES = [
    # (idx, prompt, label for the file, additional_actions_after_first_render_ms)
    (2, "a pair of wireless earbuds under 5k rupees", "cue2", []),
    (3, "I want something for my trip.", "cue3", []),
    (4, "stainless steel water bottle 1L", "cue4", ["click_first_chip"]),
]


def parse_cue_offsets(srt_path: Path) -> dict[int, tuple[float, float]]:
    """Return {idx: (start_seconds, end_seconds)}."""
    import srt  # type: ignore

    subs = list(srt.parse(srt_path.read_text(encoding="utf-8")))
    return {s.index: (s.start.total_seconds(), s.end.total_seconds()) for s in subs}


def mux_cue(video: Path, audio: Path, start: float, end: float, out: Path) -> None:
    """Cut the relevant slice from audio + concatenate with browser video.

    Browser video is short (the cue's UI action). Audio for the cue is cut from
    the full narration at the cue's SRT timestamps.
    """
    # Cut audio segment + mux with browser video
    tmp_a = out.with_suffix(".audio.m4a")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(audio),
            "-c", "copy",
            str(tmp_a),
        ],
        check=True,
        capture_output=True,
    )
    # Re-encode browser video (webm) to yuv420p mp4 to avoid codec issues
    tmp_v = out.with_suffix(".video.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an",
            str(tmp_v),
        ],
        check=True,
        capture_output=True,
    )
    # Concat: pad video to match audio duration (looped last frame if shorter)
    dur_a = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(tmp_a)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    # Pad/trim the video to the audio length
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_v),
            "-t", f"{dur_a:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True, capture_output=True,
    )
    tmp_a.unlink(missing_ok=True)
    tmp_v.unlink(missing_ok=True)


def record_cue(page, idx: int, prompt: str, label: str, actions: list[str]) -> Path:
    """Drive the Streamlit UI for one cue; return the raw video path."""
    print(f"\n=== cue {idx} ({label}): prompt = {prompt!r} ===")
    page.goto("http://localhost:8501/", wait_until="domcontentloaded")
    # Wait for the chat input
    page.wait_for_selector("textarea[placeholder='What are you looking for?']", timeout=30000)
    # Start recording context before the user input
    page.evaluate("window.__recordStart = performance.now()")
    # Type the prompt
    page.locator("textarea[placeholder='What are you looking for?']").fill(prompt)
    page.keyboard.press("Enter")
    # Wait for the agent response — look for recommendation cards
    # (assistant message contains product cards with #rank headers)
    page.wait_for_selector("text=Refine your search", timeout=60000)
    # Hold a beat for cards to fully render
    time.sleep(2.0)
    # Optional post-actions (e.g., click the first refinement chip)
    for act in actions:
        if act == "click_first_chip":
            # Click the first chip button (above the input)
            chip = page.locator("button:has-text('Refine')").first
            if chip.count() == 0:
                # Some chips have different labels; try the first button after the chips row
                chip = page.locator("[data-testid='stChatMessage'] >> nth=2 button").first
            try:
                chip.click(timeout=5000)
                time.sleep(3.0)
            except Exception as e:
                print(f"  (chip click skipped: {e})")
    raw_mp4 = VIDEO_DIR / f"{label}_ui.mp4"
    # Save the most recent browser screenshot as the cue still frame
    page.screenshot(path=str(VIDEO_DIR / f"{label}_ui.png"))
    print(f"  screenshot saved -> {VIDEO_DIR / f'{label}_ui.png'}")
    return raw_mp4  # not used for video in this minimal version


def main() -> None:
    offsets = parse_cue_offsets(SRT)
    print(f"SRT offsets: {offsets}")

    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        try:
            for idx, prompt, label, actions in CUES:
                if idx not in offsets:
                    print(f"  (skip: no SRT entry for cue {idx})")
                    continue
                record_cue(page, idx, prompt, label, actions)
        finally:
            context.close()
            browser.close()
    print("\nDone — browser session closed.")


if __name__ == "__main__":
    main()
