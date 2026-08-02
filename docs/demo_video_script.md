# Demo Video Script — RetailConcierge (AMD AI DevMaster 2026, Track 2: Agentic AI)

**Target:** 3:30 – 4:30 (hard max 5:00). English only (submission rule).
**Judging map:** functional completeness & value (60 pts) + AMD Radeon/ROCm local inference & speed optimization (40 pts).

Every second must serve one of the two judging buckets. The video is 60% "the agent works" and 40% "it runs on AMD, and here is the speed we engineered."

**Source of truth for narration:** `demo_subtitles.srt` (one cue per act). This script refers to cues by number — edit there, regenerate the audio.

---

## Filming rules (read first)

1. **Pre-warm everything before recording.** First torch.compile takes ~5 min and the model download is 23 GB. Boot vLLM + the app, run one warm-up query, THEN press record. Never show a loading spinner.
2. **Terminal font ≥ 16pt**, dark theme, no wrapping. Viewers on phones must read every command.
3. **Every command you type is visible and typed deliberately.** No mouse-only navigation for the core demo.
4. **Show the GPU.** `amd-smi` visible with utilization spiking during inference = the 40-point proof. This is non-negotiable.
5. **No dead air.** If something takes >2 s on screen, have narration covering it.
6. **Do NOT show** the multi-round escaped-quote corruption or any failed query. The tool-call reliability caveat is in the repo docs (DEPLOYMENT_JOURNAL.md) — the video shows the happy path only.
7. Record at 1080p. Screen recorder: OBS Studio (free) or `peek` on Linux. Audio: `demo_narration.mp3` (the AI voiceover, spliced onto the recording).

---

## Act 1 — Hook (0:00 – 0:25)

**On screen:** split view — left: `main.py` console booting; right: `amd-smi` showing MI300X.

**Narration:** SRT cue 1.

**Do:** one clean boot line visible — the app prints `145,615 products / 576 product types; model=...`. This is the "functional completeness" opener.

---

## Act 2 — The agent works (0:25 – 1:40)

**On screen:** console app, live. Type each query, narrate as it happens. Show the tool calls in the agent's trace if the app prints them (or overlay a small terminal with `--verbose`-style tool trace).

**Query 1 — full recommendation flow (0:25 – 0:55):**
```
you> a pair of wireless earbuds under 5k rupees
```
**Narration:** SRT cue 2. Let the 5 recommendations render. Point at one reason line: each recommendation carries "why it fits" and "trade-offs" — no invented prices, no fake availability.

**Query 2 — clarification, the agentic behavior (0:55 – 1:15):**
```
you> I want something for my trip.
```
**Narration:** SRT cue 3. The agent asks: what kind of items, budget, etc.

**Query 3 — refinement chips, multi-turn memory (1:15 – 1:40):**
```
you> stainless steel water bottle 1L
you> [2]   ← pick a refinement chip (e.g., "Vacuum Insulated")
```
**Narration:** SRT cue 4. (Optional if time is tight — skip to Act 3 if over 1:40.)

**Act 2 purpose:** prove 60 pts — tool use, reasoning, memory, task execution, practical value.

---

## Act 3 — It's real AMD: GPU + speed (1:40 – 3:10)

**On screen:** split — top: `amd-smi monitor` (live utilization + VRAM); bottom: a `curl` request to the vLLM endpoint or a second terminal running `benchmark_concurrency.sh`.

**3a. Live inference on GPU (1:40 – 2:10):**
Run one query; while it generates, point at `amd-smi`:
- GPU utilization jumping during decode
- VRAM usage (the INT8 model: ~12.5 GiB weights — say it out loud)
- **Narration:** SRT cue 5.

**3b. The speed numbers (2:10 – 2:50):**
Run `benchmark_concurrency.sh` (concurrency 1→2→4→8) or show a pre-generated table.
**Narration:** SRT cue 6.

**3c. The quantization optimization (2:50 – 3:10):**
Show the INT8 checkpoint + one command:
```
ssh root@<ip> "VLLM_FP8_MODEL=/models/gemma-4-12b-it-int8 bash deploy_droplet.sh"
```
**Narration:** SRT cue 7.

**Act 3 purpose:** prove the 40 pts — local inference on AMD, ROCm, and measurable speed optimization (quantization, prefix caching, chunked prefill, concurrency).

---

## Act 4 — Close (3:10 – 3:40)

**On screen:** the repo — `https://github.com/rajasingh012/retail-concierge`, README visible (architecture diagram + scripts).

**Narration:** SRT cue 8 (includes: "We also published the quantized model on Hugging Face for anyone to use.").

**Do:** end on the repo URL and the Track 2 badge line from the README. Freeze frame on the URL for the last 2 seconds. Optionally overlay the HF repo slug `rajasingh012/gemma-4-12b-it-quark-w8a8-int8` as a small caption next to the repo URL (text only — no upload screens, no browser navigation).

---

## Timing budget (hard cap 5:00)

The narration audio (`demo_narration.mp3`) is **2:32** — this anchors the video. The SRT timestamps are measured from the actual audio (run `python3 docs/retime_srt.py` after any narration edit to re-anchor). Plan the on-screen actions to match the audio cue positions:

| Cue | Narration | Audio position |
|---|---|---|
| 1 | Hook — app boot + amd-smi | 0:00 – 0:21 |
| 2 | Full recommendation flow | 0:22 – 0:41 |
| 3 | Clarification | 0:42 – 0:52 |
| 4 | Refinement chips | 0:52 – 1:01 |
| 5 | Live GPU inference | 1:02 – 1:14 |
| 6 | Speed numbers | 1:15 – 1:31 |
| 7 | Quantization (Quark) | 1:31 – 1:59 |
| 8 | Repo + HF mention + close | 1:59 – 2:32 |

If you run long, cut: refinement-chips query (cue 4) → then 3c quantization (cue 7) → keep cues 5-6 (that's the 40-point bucket).

---

## Pre-recording checklist

- [ ] `demo_narration.mp3` is the latest version (re-run the TTS step if you edited the SRT)
- [ ] vLLM up: `curl http://<ip>:8000/v1/models` returns the INT8 model
- [ ] One warm-up query run (torch.compile cache warm)
- [ ] `amd-smi monitor` in a dedicated terminal, dark theme
- [ ] `benchmark_concurrency.sh` table generated (or screenshot ready)
- [ ] Console font ≥ 16pt, no wrapping
- [ ] OBS/peek recording at 1080p
- [ ] Query texts copy-paste ready (typing errors waste time and look bad)
- [ ] Repo page open at README for the close shot
- [ ] Total run-through once WITHOUT recording to time it

---

## Splicing the audio onto the recording

Final assembly, one ffmpeg command on the laptop:

```bash
ffmpeg -i screen-recording.mkv -i demo_narration.mp3 \
  -c:v copy -map 0:v:0 -map 1:a:0 \
  -shortest final_demo.mp4
```

`-shortest` trims to the shorter of the two (audio or video) so a slow recording doesn't leave dead audio at the end.

---

## What NOT to do

- No cloud inference claims — everything must visibly run on the droplet (AMD MI300X).
- No waiting screens (model load, compile, download) — pre-warm.
- No failed queries or the escaped-quote corruption.
- No claim that 12B INT8 is "lossless" — say "1.7x smaller weights, ~1.7x less VRAM" (true) and verify accuracy via the tool-call benchmark before claiming equivalence.
- Don't exceed 5:00 — hard cap, the rules say recommended 3–5 min.
