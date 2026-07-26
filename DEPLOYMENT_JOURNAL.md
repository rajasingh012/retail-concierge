# AMD deployment journal — RetailConcierge

Track 2 submission notes from deploying Gemma 4 31B IT on AMD Radeon MI300X
via vLLM 0.23 on ROCm 7.2.

---

## July 26 — Initial deployment to Radeon Cloud MI300X

### Droplet spec
- GPU: AMD Instinct MI300X VF (gfx1100), 192 GB VRAM
- ROCm version: 7.2.3
- vLLM version: 0.23.0 (pre-installed in AMD 1-Click Docker image)
- Container name: `rocm` (not `vllm` as some docs suggest)
- Python 3.12 inside container

### Issue 1: Container auto-detection failed
**Symptom:** `deploy_droplet.sh` couldn't find the vLLM container. It searched for names `vllm`, `vllm-rocm`, `vllm_openai`, `amd-vllm`, `inference` — but the actual container is named `rocm`.

**Fix:** Added fallback detection paths:
1. Any running container with `vllm`/`inference` in its image name
2. Any container named exactly `rocm` or `amd`
3. If exactly one container is running, use it

**Status:** Fixed in `scripts/deploy_droplet.sh`.

### Issue 2: xargs inside docker exec can't call shell functions
**Symptom:** Line `docker inspect … | xargs -I{} log "  image: {}"` failed because `log()` is a bash function defined in the outer script, not available in the `xargs` child process.

**Fix:** Captured result into a variable first, then logged with `log`.
```bash
IMG=$(docker inspect --format '{{.Image}}' "$CTR_NAME" | head -1)
log "  image: ${IMG:-unknown}"
```

**Status:** Fixed in `scripts/deploy_droplet.sh`.

### Issue 3: Model download fails with LocalEntryNotFoundError
**Symptom:** `snapshot_download()` from `huggingface_hub` failed immediately with "LocalEntryNotFoundError: cannot find the requested files in the local cache and cannot connect to Hugging Face."

**Root cause:** The `allow_patterns` list included `*.safetensors` which requires authentication headers for some repositories. Also, the initial run attempted to download the entire model without `--max_workers`, which may have triggered HF rate limiting.

**Fix:** Re-ran after clearing the partial cache. The second attempt (without restrictive `allow_patterns` and with `max_workers=4`) succeeded. The model weighs ~59 GB on disk (2 safetensors shards: 49.8 GB + 12.8 GB).

**Takeaway:** Use plain `snapshot_download(repo_id=..., max_workers=4)` without filtering patterns for first download. Let HF decide which files to fetch.

**Status:** Working. Cache is preserved across container restarts.

### Issue 4: Speculative config flag fails in vLLM 0.23
**Symptom:** 
```
vllm serve: error: argument --speculative-config/-sc: Value method:ngram
cannot be converted to <function loads at 0x...>
```

**Root cause:** The `--speculative-config` flag was removed from vLLM 0.23 (the `spec_decode` module doesn't exist). The `ngram` speculative decoding feature was not included in the 0.23 build.

**Fix:** Removed the flag entirely. Performance is still good without it (prefix caching + chunked prefill + FP8 KV cache provide the majority of the optimization story).

**Status:** Fixed in `scripts/deploy_droplet.sh`.

### Issue 5: Wrong tool-call parser — used `hermes` instead of `gemma4`
**Symptom:** Model responded with raw Gemma 4 native tool-call format (`<|tool_call>call:search_catalog{...}<tool_call|>`) but the `hermes` parser didn't recognize it, resulting in empty `tool_calls` array and the tool output appearing as plain text content.

```
"content": "<|tool_call>call:search_catalog{query:<|\\"|>laptop backpack<|\\"|>}<tool_call|>"
"tool_calls": []
"finish_reason": "stop"
```

**Root cause:** vLLM 0.23 ships a `gemma4_tool_parser.py` that's purpose-built for Gemma 4's native chat template. The `hermes` parser expects a different format.

**Fix:** Changed `--tool-call-parser hermes` to `--tool-call-parser gemma4`.

```bash
vllm serve google/gemma-4-31B-it \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4
```

**Status:** Fixed. Verified with real curl request returning proper `tool_calls` array.

### Issue 6: GPU memory leak from killed vLLM process
**Symptom:** After stopping vLLM (via pkill), GPU memory stayed at 187 GB / 192 GB with no active process owning it. rocm-smi showed PID 9720 using 174.5 GB, but `ps` showed no such process.

```
VRAM Total Used Memory (B): 187672973312   ← ~175 GB leaked
```

**Fix:** Restarted the Docker container (`docker restart rocm`). This freed all GPU memory and reset the ROCm state. After restart: 286 MB used (just JupyterLab).

**Takeaway:** Never `pkill -9` vLLM. If it crashes or you need to restart it, do `docker restart rocm` instead.

**Status:** Documented. Script now requires manual "restart container" step before re-deploy.

### Issue 7: Startup banner detection regex
**Symptom:** Script polled vLLM log for "server is fired up and ready to roll" which doesn't exist in vLLM 0.23. The actual success banner is "Application startup complete".

**Fix:** Extended grep regex to match both patterns.

**Status:** Fixed in `scripts/deploy_droplet.sh`.

---

## Key learnings for the AMD stack

1. **Container name is `rocm`, not `vllm`** — don't guess, auto-detect.
2. **Use `--tool-call-parser gemma4`** — Gemma 4 has its own parser. `hermes` won't work.
3. **Restart the container, not the process** — `docker restart rocm` to free leaked GPU memory. `pkill -9 vllm` leaks VRAM.
4. **Model fits with headroom** — 59 GB weights, 108 GB KV cache (333K tokens), ~27× concurrent capacity at 12K context.
5. **FP8 KV cache works** — confirmed on ROCm 7.2.3 on MI300X. No issues.
6. **No speculative decoding** — removed from vLLM 0.23. Don't need it.
7. **HF download needs `max_workers=4`** — slower than ideal but works.
8. **vLLM REST port in container exposed as host port 8000** — no port mapping needed.
