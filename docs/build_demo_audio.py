#!/usr/bin/env python3
"""Build demo_narration.mp3 from demo_subtitles.srt.

The SRT is the source of truth for spoken narration. This script parses
the SRT with the `srt` library, concatenates the cue text (one blank line
between cues, giving the TTS a natural pause), and prints the result.

Usage:
    python3 docs/build_demo_audio.py                # print concatenated text
    python3 docs/build_demo_audio.py --stats         # per-cue word count + total
    python3 docs/build_demo_audio.py --srt-only     # print without timestamps

The script does NOT call TTS itself — text-to-speech is provided by the
agent tool (text_to_speech) in the session the script runs in. The agent
reads this script's output, feeds it to text_to_speech, and writes the
MP3 back to docs/demo_narration.mp3.

Dependency: `srt` (pip install srt) — the standard SRT parsing library.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import srt

HERE = Path(__file__).parent
SRT_PATH = HERE / "demo_subtitles.srt"


def parse_srt(text: str) -> list[dict]:
    subs = list(srt.parse(text))
    return [
        {
            "cue": sub.index,
            "start": sub.start,
            "end": sub.end,
            "text": " ".join(sub.content.split()),
        }
        for sub in subs
    ]


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
    try:
        cues = parse_srt(srt_text)
    except srt.SRTParseError as exc:
        parser.error(f"Failed to parse {args.srt}: {exc}")
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
