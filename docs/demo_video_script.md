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

**Split into two recording sessions** (one per machine, cleaner evidence + easier editing):

- **Phase 1 — this session, GPU droplet only**: cues 1, 5, 7 (the AMD-native inference evidence)
- **Phase 2 — next session, laptop only**: cues 2, 3, 4, 6, 8 (the agent behavior)

The narration audio (`demo_narration.mp3`) is the master for both phases; the SRT anchors it. Final splice in the next session: take the video from Phase 1, lay it on top of the audio timeline at the cue positions (1/5/7) with black/blank space where the laptop footage will go (2/3/4/6/8). Phase 2 will fill those gaps and replace the blanks with the agent-on-laptop footage. **OR** simpler: record both as one continuous 2:32 video with the GPU terminal visible in the top half and the laptop agent terminal in the bottom half for the agent cues — judge reads "two things, one proof" instantly.

---

## Act 1 — Hook (0:00 – 0:25) [PHASE 1 — GPU droplet]

**On screen:** the SSH session in the GPU droplet. Boot banner + first real `rocm-smi` output + first chat completion to vLLM.

**Narration:** SRT cue 1.

**Do:** show the app/model on AMD hardware first. Commands (paste via xclip, then `xdotool key Return`):

```bash
# 1. Confirm we're on the GPU droplet
hostname; uname -a | head -1
# 2. AMD GPU info (the visual proof)
rocm-smi --showproductname
# 3. Model served on AMD
docker exec rocm curl -s http://localhost:8000/v1/models | python3 -m json.tool | head -15
# 4. Live chat completion to vLLM (proves inference works)
docker exec rocm curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/gemma-4-12b-it-int8","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("model:",d["model"]); print("reply:",d["choices"][0]["message"]["content"][:80])'
```

---

## Act 2 — The agent works (0:25 – 1:40) [PHASE 2 — laptop only, next session]

**On screen:** the console app on the laptop, hitting the droplet's vLLM.

**Queries to type in the laptop agent:**

```
you> a pair of wireless earbuds under 5k rupees
you> I want something for my trip.
you> stainless steel water bottle 1L
you> [2]   ← pick a refinement chip
```

**Narration:** SRT cues 2, 3, 4.

**Act 2 purpose:** prove 60 pts — tool use, reasoning, memory, task execution, practical value.

---

## Act 3 — It's real AMD: GPU + speed (1:40 – 3:10)

### 3a. Live inference on GPU (1:40 – 2:10) [PHASE 1 — GPU droplet]

**On screen:** while a chat completion is in flight, `rocm-smi --showuse` shows utilization spiking. Run the request, watch the GPU respond.

**Narration:** SRT cue 5.

**Commands:**

```bash
# 1. Start amd-smi polling in the foreground (shows GPU spiking during inference)
rocm-smi --showuse --csv
# 2. In another terminal / while this runs, fire a real chat completion
#    (this is the visual proof of "every token generated on AMD")
docker exec rocm curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/gemma-4-12b-it-int8","messages":[{"role":"user","content":"Recommend 3 wireless earbuds in one paragraph."}],"max_tokens":120,"temperature":0}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("usage:",d["usage"]); print("reply:",d["choices"][0]["message"]["content"][:200])'
# 3. Show VRAM (proves the INT8 size on GPU)
rocm-smi --showmeminfo vram --csv
```

### 3b. The speed numbers (2:10 – 2:50) [PHASE 2 — laptop, with droplet serving]

**On screen:** the concurrency bench output (concurrency 1/2/4/8) on the laptop, hitting the GPU droplet. The bench proves the AMD serving layer holds up.

**Narration:** SRT cue 6.

**Commands on the laptop:**

```bash
BASE_URL=http://129.212.185.54:8000/v1 MODEL=google/gemma-4-12b-it \
  bash scripts/benchmark_concurrency.sh
```

### 3c. The quantization optimization (2:50 – 3:10) [PHASE 1 — GPU droplet]

**On screen:** show the INT8 checkpoint on the droplet + the Quark quantization_config block.

**Narration:** SRT cue 7.

**Commands:**

```bash
# 1. The actual INT8 checkpoint on AMD
ls -la /models/gemma-4-12b-it-int8/
du -sh /models/gemma-4-12b-it-int8/
# 2. The Quark quantization config (proves what scheme we used)
python3 -c "import json; print(json.dumps(json.load(open('/models/gemma-4-12b-it-int8/config.json'))['quantization_config'], indent=2)[:600])"
# 3. The vLLM serve command that loads it (proves the AMD-native serving path)
docker exec rocm ps auxf | grep -E "vllm serve" | grep -v grep | head -1
# 4. The serving line proves Quark + ROCm 0.26
docker exec rocm bash -lc "vllm --version; python3 -c 'import quark; print(\"Quark\",quark.__version__)'"
```

**Act 3 purpose:** prove the 40 pts — local inference on AMD, ROCm, and measurable speed optimization (quantization, prefix caching, chunked prefill, concurrency).

---

## Act 4 — Close (3:10 – 3:40) [PHASE 2 — laptop, repo open in browser]

**On screen:** the GitHub repo — `https://github.com/rajasingh012/retail-concierge`, README visible (architecture diagram + scripts).

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

---

## PHASE 1 STATUS (2026-08-02, done)

- **Recorded + verified:** `docs/phase1_demo.mp4` (2:15, 1920x1080, H.264 + AAC narration)
- Content: RetailConcierge banner (145,615 products / model=gemma-4-12b-it-int8),
  rocm-smi (AMD Instinct MI300X VF), vLLM INT8 model list + runtime metrics,
  13G INT8 checkpoint, Quark quantization_config exclude block, vLLM 0.26+rocm723 /
  Quark 0.12.post1 / Torch 2.11.0 ROCm versions.
- All commands executed on the GPU droplet via `docker exec rocm` (model dir is
  INSIDE the container; host `/models/` does not exist — this cost one failed
  take, fixed by prefixing every `/models` command with `docker exec rocm`).
- No failed commands on camera (verified via OCR of 7 sampled frames).

## PHASE 2 PLAN (next session, laptop only — no SSH needed)

Record the agent behavior locally (app hits the droplet's public vLLM API):

1. `uv run python main.py` with RETAIL_PROVIDER=vllm, RETAIL_BASE_URL=http://129.212.185.54:8000/v1,
   RETAIL_MODEL=/models/gemma-4-12b-it-int8
2. Query 1 (cue 2, 0:22-0:41): "a pair of wireless earbuds under 5k rupees" → 5 recs
3. Query 2 (cue 3, 0:42-0:52): "I want something for my trip." → clarification
4. Query 3 (cue 4, 0:52-1:01): "stainless steel water bottle 1L" → recs + chip
5. Speed bench (cue 6, 1:15-1:31): `bash scripts/benchmark_concurrency.sh`
6. Repo close (cue 8, 1:59-2:32): browser/github.com/rajasingh012/retail-concierge

Final assembly: lay Phase 1 video at cue positions 1/5/7, Phase 2 video at 2/3/4/6/8,
over the single 2:32 narration track (audio is the master).
