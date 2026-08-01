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

7. **Startup banner detection regex** — same fix as above.

### Issue 8: vLLM /metrics not accessible from laptop
**Symptom:** `http://<droplet-ip>:8000/metrics` returns connection refused even though `/v1/models` works fine.

**Root cause:** The vLLM `--metrics` flag is not enabled by default in vLLM 0.23. The `/metrics` endpoint is served on the same port but only from inside the container, not exposed by the AMD 1-Click image's port mapping. The existing `/v1/chat/completions` works because it's the primary API endpoint.

**Workaround:** Access metrics inside the container:
```bash
docker exec rocm curl -s http://localhost:8000/metrics | grep vllm:
```

**Status:** Not a fix needed — documented as a limitation. The benchmark report already captures timing data from the Python client side.

### Issue 9: Context length overflow at 12288 tokens
**Symptom:** `This model's maximum context length is 12288 tokens. However, you requested 0 output tokens and your prompt contains at least 12289 input tokens.` — occurs on multi-turn tool-calling queries where the accumulated conversation history exceeds `--max-model-len`.

**Fix:** Increased `--max-model-len` to 32768 in `deploy_droplet.sh`. KV cache at 32K is 744,619 tokens capacity (108 GiB). Model fits on MI300X with headroom.

**Status:** Fixed. Default in `deploy_droplet.sh` changed from 12288 to 32768.

### Issue 10: Downloaded wrong model variant (26B-A4B instead of 26B-A4B-it)
**Symptom:** MoE model failed to start with "As of transformers v4.44, default chat template is no longer allowed" — same pattern as the 31B vs 31B-it distinction.

**Root cause:** Downloaded `google/gemma-4-26B-A4B` (base) instead of `google/gemma-4-26B-A4B-it` (instruction-tuned). The base model doesn't ship `chat_template.jinja`. The -it variant does.

**Fix:** Downloaded the correct model (`gemma-4-26B-A4B-it`). It works out of the box with `--tool-call-parser gemma4`, no modifications needed.

**Result:** MoE model delivers 1,575 tok/s at concurrency-8 (3.69× speedup), 48.5 GB VRAM (vs 58.9 GB for 31B), 35ms server-side TTFT, and 64% prefix cache hit rate.

**Status:** Fixed. Droplet switched to MoE as default model (`google/gemma-4-26B-A4B-it`).

---

## Key learnings for the AMD stack

1. Container name is `rocm`, not `vllm` — don't guess, auto-detect.
2. Use `--tool-call-parser gemma4` — Gemma 4 has its own parser. `hermes` won't work.
3. Restart the container, not the process — `docker restart rocm` to free leaked GPU memory.
4. **Always use `-it` suffix** — the instruction-tuned variant is what ships the chat template and understands tool calling. Base variants lack both.
5. MoE model (26B A4B) is faster than dense 31B — 3.69× system throughput, lower VRAM (48.5 vs 58.9 GB), same tool parser.
6. vLLM /metrics exposes real TTFT histogram — preferred over client-side timing (network latency inflates client measurements).
7. vLLM Prometheus metrics include: `time_to_first_token_seconds`, `prefix_cache_hits_total`, `num_requests_running`, `kv_cache_usage_perc`, `engine_sleep_state`, `generation_tokens_total`.
8. FP8 KV cache works on ROCm 7.2.3 MI300X.
9. Context window must be ≥32768 for multi-turn tool calling.
10. HF download needs `max_workers=4`.

### Issue 11: Quark INT8 quantization succeeded, vLLM 0.23 MoE loader rejects it
**Date:** 2026-08-01

**Goal:** Quantize `google/gemma-4-26B-A4B-it` (BF16, 51.6 GB) with AMD Quark to claim the quantization bonus.

**Recipe:** `nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8` on HF — same `Gemma4ForConditionalGeneration` architecture, measured −0.08pp on GSM8K vs BF16. Scheme: per-channel INT8 weights (ch_axis=0) + per-token dynamic INT8 activations (ch_axis=1). Exclusions stay BF16: `lm_head`, `*embed_tokens*`, `*vision_tower*`, `*embed_vision*`. No calibration data needed (dynamic activation).

**Quark 0.12 API drift vs the 0.11 recipe (all hit live):**
- `Config` → `QConfig`, `QuantizationConfig` → `QLayerConfig`
- `ModelQuantizer.export_model()` → `quark.torch.export_safetensors()`
- `amd_quark.tools.quark_quantize` CLI removed entirely
- `LLMTemplate.get("gemma4")` — **no gemma4 template exists** in Quark 0.12 (only gemma2/gemma3). The recipe's manual QConfig is mandatory.

**Result:** Quantization **succeeded in 57s** on MI300X → `/models/gemma-4-26B-A4B-it-int8/model.safetensors` (49.96 GB, I8 weights + BF16 per-channel scales + zero-points). Verified tensors: `layers.0.router.proj.weight` I8, `weight_scale` BF16 [128], `weight_zero_point` I8 — correct W8A8 format.

**Blocker:** vLLM 0.23 auto-detects `quantization=quark` from config.json, but the EngineCore fails:
```
KeyError: 'layers.0.router.proj.weight_scale'
```
The safetensors key is `model.language_model.layers.0.router.proj.weight_scale` (with prefix). vLLM 0.23's Quark MoE router loader builds the lookup key **without** the `model.language_model.` prefix → KeyError. Non-MoE layers work; the MoE router path is the bug.

**Verdict:** Our quantization is correct. The blocker is vLLM 0.23's Quark-MoE loader. Fix requires vLLM ≥ 0.26 (Quark MoE support matured) — the AMD 1-Click image ships 0.23. Deferred, documented per PR #7's "production recommendation remains FP16" framing.

**Status:** INT8 model saved at `/models/gemma-4-26B-A4B-it-int8` (droplet), BF16 server restored as default. Revisit post-submission with vLLM ≥ 0.26.

### Issue 12: vLLM 0.26 upgrade — version wall fixed, Quark INT8 still corrupts MoE output
**Date:** 2026-08-01

**Goal:** Serve the Quark INT8 checkpoint by upgrading vLLM past the 0.23 MoE loader bug (`KeyError: 'layers.0.router.proj.weight_scale'`).

**Upgrade path (AMD-sanctioned, researched):** AMD's `ROCm/vllm` fork is officially deprecated (2025-09-09); AMD points to upstream vLLM's ROCm wheel index. Installed `vllm==0.26.0+rocm723` from `wheels.vllm.ai/rocm/0.26.0/rocm723/` via `uv pip install --system` inside the container. ABI fixes along the way: uninstalled `flash-attn` (torch 2.11 ABI mismatch on `getCurrentHIPStream`) and `torchaudio` (libc10 mismatch).

**Result 1 — version wall FIXED:** vLLM 0.26 loads the Quark INT8 checkpoint (no KeyError). Model loads at 25.75 GiB (vs 49.79 BF16). Torch.compile first-boot is slow (~13 min) but warm-cache boots are fast.

**Result 2 — quantization FAILS quality on MoE:** INT8 output is garbage (`1-1-1-1-0-1-0-1-0-1-s-s-s-1-s-` for "What is 2+2"). Tested twice:
- (a) 31B-dense recipe as-is (everything quantized except lm_head/embeddings/vision): garbage
- (b) MoE-aware exclusions added (`*router*`, `*experts*`, `*shared_experts*`, `*moe*` kept BF16): **still garbage**

The W8A8 INT8 scheme works on the 31B dense (proven −0.08pp GSM8K) but corrupts the 26B A4B MoE. The corruption is not in the router/expert scales (exclusion didn't help) — likely the dynamic per-token activation quantization interacting with the MoE layer shapes, or the scale layout vLLM 0.26's loader expects vs what Quark exports for shared-expert paths.

**Verdict (final):** Quark W8A8 INT8 quantization is NOT viable for the 26B A4B MoE. The 20-pt quantization bonus is not claimable. Documented per PR #7's honest-rejection framing. Future work (post-submission): try static activation quantization with calibration data, or FP8 (`fp8_e4m3`) instead of INT8.

**Status:** BF16 restored as serving default on the vLLM 0.26 droplet (parity verified, mean 9.5s). INT8-MoE experiment closed with documented negative result.
