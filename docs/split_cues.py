#!/usr/bin/env python3
"""Split demo narration + video into per-cue segments (one per SRT cue).

Produces docs/cues/cueN.mp4 where N = SRT cue index:
- cues with Phase 1 footage (1, 5, 6, 7): video segment from phase1_demo.mp4
- cues pending Phase 2 (2, 3, 4, 8): black frame + narration audio (placeholder)

Dependencies: srt (parse), ffmpeg (segmentation).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import srt

HERE = Path(__file__).parent
SRT_PATH = HERE / "demo_subtitles.srt"
AUDIO_PATH = HERE / "demo_narration.mp3"
PHASE1_VIDEO = HERE / "phase1_demo.mp4"
OUT_DIR = HERE / "cues"

# Which cues have real footage so far (Phase 1 = GPU droplet evidence).
# Phase 2 (laptop agent demo) will fill cues 2, 3, 4, 8.
HAS_FOOTAGE = {1, 5, 6, 7}


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def run(cmd: list[str]) -> None:
    print("  " + " ".join(str(c) for c in cmd[:6]) + " ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    subs = list(srt.parse(SRT_PATH.read_text(encoding="utf-8")))
    print(f"{len(subs)} cues, output -> {OUT_DIR}")

    for s in subs:
        start = s.start.total_seconds()
        end = s.end.total_seconds()
        dur = end - start
        out = OUT_DIR / f"cue{s.index}.mp4"

        if s.index in HAS_FOOTAGE:
            # Video segment from Phase 1 footage, plus this cue's audio.
            # The Phase 1 video is ~2:15; cue times map by absolute offset.
            print(f"\ncue {s.index}: real footage, {dur:.1f}s "
                  f"({fmt_ts(start)} -> {fmt_ts(end)})")
            # Re-encode: cut video from phase1, add narration audio for the cue
            run([
                "ffmpeg", "-y",
                "-ss", fmt_ts(start), "-to", fmt_ts(end),
                "-i", str(PHASE1_VIDEO),
                "-ss", fmt_ts(start), "-to", fmt_ts(end),
                "-i", str(AUDIO_PATH),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac",
                "-shortest",
                str(out),
            ])
        else:
            # Placeholder: black 1080p frame + narration audio
            print(f"\ncue {s.index}: PLACEHOLDER (Phase 2 pending), {dur:.1f}s")
            run([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:r=30:d={dur:.3f}",
                "-ss", fmt_ts(start), "-to", fmt_ts(end),
                "-i", str(AUDIO_PATH),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac",
                "-shortest",
                str(out),
            ])
        print(f"    -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
