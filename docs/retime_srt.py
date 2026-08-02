#!/usr/bin/env python3
"""Retime demo_subtitles.srt so cue timestamps match the actual narration audio.

Problem: the SRT timestamps were hand-planned from the script's video timing
budget (acts at 0:25/1:40/2:50/...). The TTS engine reads faster than the
budget, so the audio ends at ~2:32 while the SRT claims 3:45. Subtitles
spliced onto the video would drift.

Fix: compute per-cue durations from the measured reading rate (s/word) of
the actual MP3, add a small inter-cue pause (TTS inserts a breath between
paragraphs), and rewrite the SRT timestamps so the last cue ends exactly at
the audio duration.

Usage:
    python3 docs/retime_srt.py --audio docs/demo_narration.mp3 \
        --pause 0.6 [--dry-run]

Dependencies: mutagen (pip install mutagen) for MP3 duration.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import srt
from mutagen.mp3 import MP3

HERE = Path(__file__).parent
SRT_PATH = HERE / "demo_subtitles.srt"
DEFAULT_AUDIO = HERE / "demo_narration.mp3"

PAUSE_BETWEEN_CUES_S = 0.6  # TTS inserts a short breath between paragraphs


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO,
                        help=f"MP3 whose duration anchors the retime (default: {DEFAULT_AUDIO})")
    parser.add_argument("--pause", type=float, default=PAUSE_BETWEEN_CUES_S,
                        help=f"inter-cue pause in seconds (default: {PAUSE_BETWEEN_CUES_S})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the new timeline without writing the file")
    args = parser.parse_args()

    # 1. Parse the current SRT (source of truth for cue text)
    subs = list(srt.parse(SRT_PATH.read_text(encoding="utf-8")))
    if not subs:
        parser.error(f"No cues in {SRT_PATH}")

    # 2. Measure actual audio duration
    audio = MP3(str(args.audio))
    total_audio = audio.info.length
    total_words = sum(len(s.content.split()) for s in subs)

    # 3. Effective reading rate: exclude the inter-cue pauses from the rate
    #    so the cumulative timeline lands exactly on the audio end.
    n_cues = len(subs)
    pause_total = (n_cues - 1) * args.pause
    rate = (total_audio - pause_total) / total_words

    # 4. Build the new cumulative timeline
    cursor = 0.0
    for i, sub in enumerate(subs):
        words = len(sub.content.split())
        dur = words * rate
        new_start = cursor
        new_end = new_start + dur
        sub.start = _to_timedelta(new_start)
        sub.end = _to_timedelta(new_end)
        cursor = new_end + args.pause

    # 5. Report
    print(f"Audio measured:        {total_audio:.1f}s")
    print(f"Words:                 {total_words}")
    print(f"Reading rate:          {rate:.3f} s/word (excl. {args.pause}s inter-cue pauses)")
    print(f"Retimed last cue end:  {fmt_ts(cursor - args.pause)}")
    print(f"Matches audio end:     {abs((cursor - args.pause) - total_audio) < 0.5} (within 0.5s)\n")

    if args.dry_run:
        for s in subs:
            print(f"Cue {s.index:2d}: {s.start} -> {s.end}  ({len(s.content.split())} words)")
        return

    SRT_PATH.write_text(srt.compose(subs), encoding="utf-8")
    print(f"Wrote {SRT_PATH}")
    for s in subs:
        print(f"Cue {s.index:2d}: {s.start} -> {s.end}  ({len(s.content.split())} words)")


def _to_timedelta(seconds: float):
    import datetime
    ms = int(round(seconds * 1000))
    return datetime.timedelta(milliseconds=ms)


if __name__ == "__main__":
    main()
