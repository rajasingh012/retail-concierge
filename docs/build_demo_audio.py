#!/usr/bin/env python3
"""Build demo_narration.mp3 from demo_subtitles.srt.

The SRT is the source of truth for spoken narration. This script parses
the SRT, concatenates the cue text (one blank line between cues, giving
the TTS a natural pause), and prints the result to stdout.

Usage:
    python3 docs/build_demo_audio.py                # print concatenated text
    python3 docs/build_demo_audio.py --stats         # per-cue word count + total
    python3 docs/build_demo_audio.py --srt-only     # print without timestamps

The script does NOT call TTS itself — text-to-speech is provided by the
agent tool (text_to_speech) in the session the script runs in. The agent
reads this script's output, feeds it to text_to_speech, and writes the
MP3 back to docs/demo_narration.mp3. Run this script in any Python 3
environment; no third-party deps required.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).parent
SRT_PATH = HERE / "demo_subtitles.srt"

# SRT cue format: cue number, timestamp range, one-or-more text lines, blank line.
# This regex matches the timestamp line; everything between the timestamp and the
# next blank line (or EOF) is the cue text.
CUE_RE = re.compile(
    r"^\d+\s*$\n"          # cue number
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*$\n"
    r"((?:.*\n)+?)"        # one or more text lines (non-greedy)
    r"(?:\n|\Z)",          # terminated by blank line or EOF
    re.MULTILINE,
)


def parse_srt(text: str) -> list[dict]:
    matches = []
    for m in CUE_RE.finditer(text):
        start, end, body = m.group(1), m.group(2), m.group(3)
        text_only = " ".join(line.strip() for line in body.strip().splitlines())
        matches.append({
            "cue": len(matches) + 1,
            "start": start,
            "end": end,
            "text": text_only,
        })
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srt", type=Path, default=SRT_PATH,
                        help=f"Path to SRT (default: {SRT_PATH})")
    parser.add_argument("--stats", action="store_true",
                        help="print per-cue word count + total")
    parser.add_argument("--srt-only", action="store_true",
                        help="print cue text without timestamps")
    args = parser.parse_args()

    srt_text = args.srt.read_text(encoding="utf-8")
    cues = parse_srt(srt_text)
    if not cues:
        parser.error(f"No cues parsed from {args.srt}. Check the file format.")

    if args.stats:
        total_words = 0
        for c in cues:
            n = len(c["text"].split())
            total_words += n
            print(f"Cue {c['cue']}: {c['start']} -> {c['end']}  {n:>3} words")
        print(f"Total: {len(cues)} cues, {total_words} words")
        return

    if args.srt_only:
        for c in cues:
            print(f"[Cue {c['cue']}] {c['text']}\n")
        return

    # Default: concatenated text for TTS. One blank line between cues so the
    # TTS engine naturally pauses at the cue boundaries.
    print("\n\n".join(c["text"] for c in cues))


if __name__ == "__main__":
    main()
