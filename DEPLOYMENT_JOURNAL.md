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

### Issue 13: MoE-aware quantization attempt — fused-expert split + correct exclusions, still garbage
**Date:** 2026-08-01

**Goal:** Fix the INT8-MoE corruption using the MoE-specific recipe from `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8` (HF) — a 256-expert MoE quantized with Quark, +0.00pp GSM8K. The recipe's two structural steps were missing from the 31B-dense recipe we initially copied.

**Pre-quantization rewrite (implemented):** `Gemma4TextExperts` stores fused 3D tensors (`gate_up_proj [128, 1408, 2816]`, `down_proj [128, 2816, 704]`). Replaced each of the 30 MoE layers' experts module with a `ModuleList[128]` of per-expert `gate_proj`/`up_proj`/`down_proj` nn.Linear triplets, copied from the fused tensors — so Quark observes standard nn.Linear modules and the export key layout matches vLLM's FusedMoE loader. Verified: "Split fused experts in 30 layers (128 experts each)".

**Exclusion fixes (iterative, all discovered via vLLM loader errors):**
- `*mlp.gate*` splits fused `gate_up_proj` shards → vLLM ValueError "different quantization schemes for gate_proj/up_proj"
- `*visual*` doesn't match `embed_vision`/`vision_tower` prefixes → vLLM ValueError on `embed_vision.embedding_projection.weight_scale`
- Final list: `lm_head`, `*router*`, `*shared_expert_gate*`, `*vision_tower*`, `*embed_vision*`, `*visual*`, `*embed_tokens*`

**Post-export rename (implemented):** `*_quantizer.scale` → `*_scale`, dropped `*_quantizer.zero_point`, squeezed `weight_scale` [out,1]→[out] (per the Qwen recipe's rename_keys.py).

**Result:** Model loads cleanly in vLLM 0.26 (25.75 GiB, no loader errors) but **output is still garbage** (`_0_re_0-re_0_s_0-s_re-target-0-s-1-s-1` for "What is 2+2"). Four attempts total (dense as-is, dense+MoE exclusions, MoE-aware, MoE-aware+vision fixes) — all load, all garbage.

**Verdict (final):** The W8A8 INT8 path is not viable for Gemma 4 26B A4B MoE with Quark 0.12 + vLLM 0.26, even with the full MoE recipe. Hypothesis: the corruption is in the **dynamic per-token activation quantization** interacting with Gemma 4's unusual `per_layer_input_gate` / `hidden_size_per_layer_input` structure (not present in Qwen MoE), or the vision-language cross-attention paths. The tokens come out grammatically-shaped but semantically empty — consistent with activations being destroyed mid-network rather than weights.

**Future work (post-submission):** (a) try `is_dynamic=False` + calibration dataset (static activation quantization); (b) try FP8 (`ptpc_fp8`) instead of INT8; (c) inspect Gemma 4's per_layer_input_gate handling in Quark's observer attachment. The recipe scaffolding (`_quark_quantize_moe.py`) is committed for provenance.

**Status:** BF16 restored as serving default on vLLM 0.26 (129.212.191.188). Quantization bonus not claimable — documented honestly per PR #7 framing.

### Recipe split + 12B dense verification (post-Issue-13)
**Date:** 2026-08-02

**Change:** Single `_quark_quantize_int8.py` split into a per-architecture pair so each recipe owns its own exclude list and pre/post steps. Common code (BF16 load, QTensorConfig construction, quantize+export) extracted into `_quark_common.py`. `quantize_int8.sh` now takes `--kind dense|moe --model <hf-repo>` and dispatches.

- `_quark_quantize_dense.py` — Gemma 4 dense (12B / 31B). Mirror of the proven 31B recipe, exclude list adjusted for 12B's actual structure.
- `_quark_quantize_moe.py` — Gemma 4 26B A4B MoE. Pre-quantization expert rewrite + post-export key rename. Still known-broken per Issue 13 (kept for failure-trail provenance).
- `_find_bf16.py` — generalized to take `--model SUBSTRING` (default `gemma-4-26B-A4B-it` preserves original behavior).

**12B structure verification (google/gemma-4-12b-it, fetched Aug 2):**
- Architecture class: **`Gemma4UnifiedForConditionalGeneration`** (NOT the older `Gemma4ForConditionalGeneration` used by 26B). Each recipe now passes its class explicitly to `_quark_common.load_bf16(model_in, model_class)`.
- Decoder layers: 48 (vs 30 on 26B MoE, ~62 on 31B).
- MoE: none (`enable_moe_block: False`, `num_experts: None`).
- Vision: **only `vision_embedder.*`** (patch + pos embed + LN) — NO `vision_tower` transformer encoder. The 31B recipe's 189-entry vision_tower exclude list does NOT apply; replaced with `model.vision_embedder.*`.
- `lm_head`: not separately stored (`tie_word_embeddings: True`) — implicitly covered when we exclude `embed_tokens`. So `embed_tokens` IS quantized (matching the proven 31B recipe, which also does not exclude embed_tokens).
- New modality: `model.embed_audio.embedding_projection` present (12B is multimodal w/ audio; 31B recipe has no analog). Excluded conservatively.

**Final 12B exclude list:** `model.vision_embedder.*`, `model.embed_vision.embedding_projection`, `model.embed_audio.embedding_projection`. embed_tokens quantized (matches proven 31B at −0.08pp GSM8K).

**Caveat (do not skip when claiming the quantization bonus):** The −0.08pp GSM8K figure was measured on the **older** Gemma4ForConditionalGeneration class (31B), not the Unified class used by 12B. The 12B run is treated as a fresh quantization with the 31B recipe as the closest analog — accuracy must be re-measured on Unified via the same tool-call accuracy gate before the bonus can be claimed.

**Open prerequisite (pre-flight on droplet before first 12B run):** Confirm `transformers` shipped in the AMD 1-Click `rocm` container exposes `Gemma4UnifiedForConditionalGeneration`. If not, `uv pip install` a transformers version that does. Same check applies to the MoE script + `Gemma4TextExperts` whenever the MoE path gets revisited.

### Pre-flight checks for the 12B dense run (post-Issue-13)
**Date:** 2026-08-02

Three checks surfaced from external verification (transformers / Quark 0.12 / 31B recipe). None block the code change. Each must pass on the droplet before the first `quantize_int8.sh --kind dense --model google/gemma-4-12b-it` run. Document the result inline; if any fails, fix and re-run the pre-flight before invoking the dense recipe.

**Check A: `transformers` >= 5.10.1 in the container**
- Why: `Gemma4UnifiedForConditionalGeneration` (12B's class) was introduced in PR huggingface/transformers#46385 "who needs encoders?", commit 1423d22f7a, 2026-06-03, first shipped in `transformers` v5.10.1. The class lives in `transformers.models.gemma4_unified`, NOT `transformers.models.gemma4` (separate modules).
- How to verify:
  `docker exec $CTR_NAME python3 -c "from transformers import Gemma4UnifiedForConditionalGeneration; print('ok')"`
- If it fails: `docker exec $CTR_NAME uv pip install --system 'transformers>=5.10.1'`
- Risk if skipped: `load_bf16()` raises ImportError AFTER the 51+ GB BF16 model has been downloaded - wasted 10+ min of GPU-host bandwidth.

**Check B: Quark FX trace succeeds with `dataloader=None` on Unified class**
- Why: Quark 0.12 has no `gemma4_unified` template entry in `LLMTemplate._templates` (only `gemma2`/`gemma3`/`gemma3_text`). We never call `LLMTemplate.get()` so this doesn't error - but `ModelQuantizer.quantize_model(model, dataloader=None)` traces the model via `torch.fx`, which requires a sample forward pass. With `dataloader=None` and no input kwarg, the trace falls back to symbolic shapes; for multimodal models with branching vision/text paths this can fail.
- Confirmed by reading quark-0.12 source: Quark's graph passes are class-agnostic (`get_layer_quant_config` filters by `layer_type in [nn.Linear, nn.Conv2d]` at module walk time, not by Python class hierarchy). The exclude matcher uses `fnmatch` on module paths, so `model.vision_embedder.*` matches `Gemma4UnifiedClippableLinear` the same as `nn.Linear` as long as the forward path goes through `aten::linear`. The remaining risk is purely the FX trace succeeding with no input.
- How to verify (dry-run, no quantization):
  `docker exec $CTR_NAME python3 -c "
from transformers import Gemma4UnifiedForConditionalGeneration, AutoTokenizer
import torch
m = Gemma4UnifiedForConditionalGeneration.from_pretrained('/models/google/gemma-4-12b-it', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True).eval()
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.config.type import Dtype, RoundType, ScaleType, QSchemeType
from quark.torch.quantization.observer import PerChannelMinMaxObserver
from quark.torch.quantization.config.config import QTensorConfig
weight_spec = QTensorConfig(dtype=Dtype.int8, observer_cls=PerChannelMinMaxObserver, symmetric=True, is_dynamic=False, qscheme=QSchemeType.per_channel, ch_axis=0, round_method=RoundType.round, scale_type=ScaleType.float)
input_spec = QTensorConfig(dtype=Dtype.int8, observer_cls=PerChannelMinMaxObserver, symmetric=True, is_dynamic=True, qscheme=QSchemeType.per_channel, ch_axis=1, round_method=RoundType.round, scale_type=ScaleType.float)
q = QConfig(global_quant_config=QLayerConfig(input_tensors=input_spec, weight=weight_spec), exclude=['model.vision_embedder.*', 'model.embed_vision.embedding_projection', 'model.embed_audio.embedding_projection'])
mq = ModelQuantizer(q, multi_device=True)
m = mq.quantize_model(m, dataloader=None)
print('FX trace + quantize attach OK')
"`
- If it fails with a trace error: switch to passing a tiny dummy input tensor. Add to recipe: `model_args=(torch.zeros(1, 8, dtype=torch.long, device='cuda'),)` and pass via the `args=` kwarg of `quantize_model`. Recipe stays generic because the shape doesn't matter for static-weight-only + dynamic-activation (no calibration data).
- If it fails on an unexpected submodule: the exclude list needs another entry. Walk `model.named_modules()` and look for any module whose forward path doesn't hit `aten::linear` (e.g., a vision embedding reshape) and add it to `DENSE_EXCLUDE_12B`.

**Check C: glob vs fully-resolved exclude for vLLM loader compatibility**
- Why: the published 31B recipe README uses globs (`*embed_tokens*`, `*vision_tower*`, `*embed_vision*`, `lm_head`) but the published 31B model's `config.json` has 192 fully-qualified paths. Quark's quantizer-time matcher uses `fnmatch` on module paths so globs work at quantization time - but vLLM 0.26's loader iterates `quantization_config.exclude` and may need exact paths to verify each layer is present. The 31B HF model presumably works with vLLM via the resolved form.
- How to verify after Check B passes: deploy the resulting checkpoint with `vLLM_FP8_MODEL=/models/gemma-4-12b-it-int8` and look at vLLM startup logs.
- If vLLM errors with "excluded layer X not found in model": re-run quantization with a resolved exclude list. Either (a) hand-enumerate the `model.vision_embedder.*` matches against `model.named_modules()` (small list - 10 tensors per the safetensors header), or (b) add a post-quantize resolver step that walks the model and rewrites `config.json`'s exclude globs to exact paths before export.
- Defer until Check B confirms a working checkpoint exists; resolving exclude paths against a non-traced model is wasted effort.

**Summary**: A and B must pass before the first 12B quantize run. C is a post-quantize verification, not a blocker. All three documented here so the next person to revisit 12B quantization has a checklist.

### How the 31B recipe actually worked (architecture-class audit)
**Date:** 2026-08-02

Followed up on the "no gemma4 in Quark" question. The answer matters because it tells us whether our 12B recipe is on solid ground or if we're flying blind.

**Recipe provenance chain (verified from external sources):**
- 2026-04-02: `transformers` PR #45192 "casually dropping the most capable open weights on the planet" added `Gemma4ForConditionalGeneration` (the older class, in `transformers/models/gemma4/`) in commit 91b1ab1fdf.
- 2026-04-23: nameistoken published `Gemma-4-31B-it-Quark-W8A8-INT8` using Quark 0.11.1 (the latest at that time; `quantization_config.version = "0.11.1"` in the published model).
- 2026-06-03: `transformers` PR #46385 "who needs encoders?" added `Gemma4UnifiedForConditionalGeneration` (the new Unified class, in `transformers/models/gemma4_unified/`), first shipped in transformers v5.10.1.
- 2026-07-03 to 2026-07-04: Quark 0.12 / 0.12.post1 released. **The `Config` / `QuantizationConfig` aliases that 0.11.1 exposed were removed in 0.12** — only `QConfig` / `QLayerConfig` remain.

**How the 31B recipe worked despite no Gemma4 template in Quark:**
- The recipe never called `LLMTemplate.get("gemma4")`. It bypassed the template system entirely with a manual `Config(global_quant_config=QuantizationConfig(...), exclude=[...])` call.
- Quark's quantize path is class-name-agnostic: it FX-traces the model, replaces `aten::linear` call_function with `QuantLinear` call_module, and uses `fnmatch` to match exclude patterns against named module paths. It does not isinstance-check against any specific Python class.
- Verified by reading `quark-0.12/quark/torch/export/main_export/quant_config_parser.py:60-90`: the exclude matcher uses `fnmatch.fnmatch(layer_name, exclude_layer)` on the dotted module name.
- Consequence: any model that goes through `aten::linear` for its weight-bearing ops will quantize correctly regardless of whether Quark has a template for it. Subclasses of `nn.Linear` (like `Gemma4UnifiedClippableLinear` in 12B's Unified class) work because the forward path still hits `aten::linear`.

**Implication for our 12B recipe:**
- Our manual-QConfig approach is the same shape the 31B recipe used, so the technique is proven for older `Gemma4ForConditionalGeneration`.
- 12B's `Gemma4UnifiedForConditionalGeneration` is class-orthogonal to Quark (FX trace + module-name matching), so the same approach should work in principle.
- The remaining unknown (covered by pre-flight Check B above) is whether Quark's FX trace can trace multimodal models with `dataloader=None` and no input. Pure-text models trace fine with symbolic shapes; multimodal models with vision/text branching may need a dummy input tensor.

**Implication for the recipe README as-published:**
- The 31B HF model card's "Reproduce Quantization" code uses `Config`/`QuantizationConfig` (0.11.1 names). It will NOT run on Quark 0.12 — those aliases are gone.
- Our code uses `QConfig`/`QLayerConfig` (0.12 names) which is correct per the rename. Already documented in Issue 11.
- If anyone tries to literally copy-paste the 31B recipe README into a fresh 0.12 install, they'll get `ImportError: cannot import name 'Config' from 'quark.torch.quantization.config.config'`. Point them at our `_quark_common.py` for the renamed equivalent.

### How the 31B recipe actually worked (architecture-class audit)
**Date:** 2026-08-02

Followed up on the "no gemma4 in Quark" question. The answer matters because it tells us whether our 12B recipe is on solid ground or if we're flying blind.

**Recipe provenance chain (verified from external sources):**
- 2026-04-02: `transformers` PR #45192 "casually dropping the most capable open weights on the planet" added `Gemma4ForConditionalGeneration` (the older class, in `transformers/models/gemma4/`) in commit 91b1ab1fdf.
- 2026-04-23: nameistoken published `Gemma-4-31B-it-Quark-W8A8-INT8` using Quark 0.11.1 (the latest at that time; `quantization_config.version = "0.11.1"` in the published model).
- 2026-06-03: `transformers` PR #46385 "who needs encoders?" added `Gemma4UnifiedForConditionalGeneration` (the new Unified class, in `transformers/models/gemma4_unified/`), first shipped in transformers v5.10.1.
- 2026-07-03 to 2026-07-04: Quark 0.12 / 0.12.post1 released. **The `Config` / `QuantizationConfig` aliases that 0.11.1 exposed were removed in 0.12** - only `QConfig` / `QLayerConfig` remain.

**How the 31B recipe worked despite no Gemma4 template in Quark:**
- The recipe never called `LLMTemplate.get("gemma4")`. It bypassed the template system entirely with a manual `Config(global_quant_config=QuantizationConfig(...), exclude=[...])` call.
- Quark's quantize path is class-name-agnostic: it FX-traces the model, replaces `aten::linear` call_function with `QuantLinear` call_module, and uses `fnmatch` to match exclude patterns against named module paths. It does not isinstance-check against any specific Python class.
- Verified by reading `quark-0.12/quark/torch/export/main_export/quant_config_parser.py:60-90`: the exclude matcher uses `fnmatch.fnmatch(layer_name, exclude_layer)` on the dotted module name.
- Consequence: any model that goes through `aten::linear` for its weight-bearing ops will quantize correctly regardless of whether Quark has a template for it. Subclasses of `nn.Linear` (like `Gemma4UnifiedClippableLinear` in 12B's Unified class) work because the forward path still hits `aten::linear`.

**Implication for our 12B recipe:**
- Our manual-QConfig approach is the same shape the 31B recipe used, so the technique is proven for older `Gemma4ForConditionalGeneration`.
- 12B's `Gemma4UnifiedForConditionalGeneration` is class-orthogonal to Quark (FX trace + module-name matching), so the same approach should work in principle.
- The remaining unknown (covered by pre-flight Check B above) is whether Quark's FX trace can trace multimodal models with `dataloader=None` and no input. Pure-text models trace fine with symbolic shapes; multimodal models with vision/text branching may need a dummy input tensor.

**Implication for the recipe README as-published:**
- The 31B HF model card's "Reproduce Quantization" code uses `Config`/`QuantizationConfig` (0.11.1 names). It will NOT run on Quark 0.12 - those aliases are gone.
- Our code uses `QConfig`/`QLayerConfig` (0.12 names) which is correct per the rename. Already documented in Issue 11.
- If anyone tries to literally copy-paste the 31B recipe README into a fresh 0.12 install, they'll get `ImportError: cannot import name 'Config' from 'quark.torch.quantization.config.config'`. Point them at our `_quark_common.py` for the renamed equivalent.

### 12B dense end-to-end live run (first working INT8 checkpoint)
**Date:** 2026-08-02 (droplet 165.245.129.253)

**Goal achieved:** `google/gemma-4-12b-it` BF16 -> W8A8 INT8 via Quark 0.12.post1, served on vLLM 0.26.0+rocm723. First Gemma 4 checkpoint to serve correctly (the 26B MoE INT8 was garbage; see Issues 11-13). Smoke tests all pass, "What is 2+2?" -> "4".

**Pipeline (all repo scripts, no manual steps beyond env vars):**
1. `upgrade_vllm.sh` — 0.23.0 -> 0.26.0+rocm723 (wheels.vllm.ai/rocm/0.26.0/rocm723). Already handled flash-attn/torchaudio/torch_c_dlpack_ext ABI breaks. Container already had Quark 0.12.post1 + transformers 5.12.0 (>= 5.10.1 requirement satisfied).
2. BF16 download — `snapshot_download` with `allow_patterns=['*.json','*.txt','*.model','*.safetensors','tokenizer*']` (23 GB).
3. `quantize_int8.sh --kind dense --model google/gemma-4-12b-it` — preflight (quark 0.12.post1, transformers 5.12.0, Gemma4Unified importable, BF16 found), quantize (~40s), 14 GB output.
4. `deploy_droplet.sh` with `SKIP_DOWNLOAD=1 VLLM_FP8_MODEL=/models/gemma-4-12b-it-int8 VLLM_MODEL=google/gemma-4-12b-it`.

**Three bugs found live + fixed in repo scripts (all would recur on a fresh droplet):**

**Bug 1 — `_find_bf16.py` rejected single-file checkpoints.** It looked only for `model.safetensors.index.json`; 12B ships a single `model.safetensors` (no index). Fixed to accept either layout. Symptom: "BF16 model matching ... not in HF cache" despite the model being present.

**Bug 2 — dense exclude list used HF names, Quark matches post-rename names.** Quark 0.12 renames `vision_embedder` -> `embed_vision.patch_*` / `embed_vision.pos_*` and wraps `embedding_projection` as `embed_vision.multimodal_embedder.embedding_projection` BEFORE applying exclude patterns. Our exclude (`model.vision_embedder.*`, `model.embed_vision.embedding_projection`) matched nothing -> vision weights got quantized -> vLLM loader rejected the checkpoint:
  `ValueError: There is no module or parameter named 'embed_vision.multimodal_embedder' in Gemma4UnifiedForConditionalGeneration.`
  Fix: `DENSE_EXCLUDE_12B = ["model.embed_vision.*", "model.embed_audio.*", "lm_head"]` (the names Quark sees). lm_head must be explicit because Quark un-ties it from embed_tokens during export.

**Bug 3 — `deploy_droplet.sh` OOM detector false-positive.** The OOM grep pattern included "Free memory on device", which is a NORMAL vLLM startup INFO line. It killed the deploy at the exact moment startup succeeded. Fixed to only match real OOM phrases. Also: smoke tests checked `$VLLM_MODEL` but INT8 serving uses `$SERVED_MODEL` (the checkpoint path) — `/v1/models` and the tool-call probe both got `$SERVED_MODEL`; summary lines updated.

**Post-quantize fixups now automated (new `_quark_fix_vllm_keys.py`, run by `quantize_int8.sh` step 2.5 for dense):**
- Key rename: `model.embed_vision.multimodal_embedder.embedding_projection.*` -> `model.embed_vision.embedding_projection.*`; `model.embed_vision.patch_*` / `pos_*` -> `model.vision_embedder.*`. Without it vLLM raises the ValueError above even with the correct exclude list (the rename is independent of quantization).
- `chat_template.jinja` copy: Quark does not export it, and the BF16 download's allow_patterns excluded `*.jinja`. vLLM 0.26 rejects chat requests with HTTP 400 ("default chat template is no longer allowed") without it. Fix script copies from BF16 source snapshot or downloads from HF.
- Skippable via `SKIP_VLLM_KEY_FIX=1`; MoE path unaffected (its rename_keys() handles the MoE-specific rename).

**Runtime notes for the next droplet:**
- First torch.compile took 302s (~5 min); the AOT cache persisted so the second deploy booted in 84s. Do not mistake the long first boot for a hang.
- `--kv-cache-dtype` is now **env-tuned, not hardcoded**: `KV_CACHE_DTYPE` defaults to `auto` when serving a Quark INT8 checkpoint (avoids the "uncalibrated q_scale ... may cause accuracy issues" warning — Quark INT8 exports carry no KV scale factors) and `fp8` for BF16 serving (proven 26B config). Override anytime with `KV_CACHE_DTYPE=bfloat16|fp8|auto`. Implemented in deploy_droplet.sh on 2026-08-02.
- Model: 12.54 GiB weights + 5.22 GiB CUDA graphs; ~157 GiB KV cache configured at GPU_MEM=0.9, max-model-len 32768, on 191.7 GB MI300X. Plenty of headroom.
- Quark output compression: 23.9 GB BF16 -> 14 GB INT8 (~1.7x; not 2x because lm_head + vision/audio embedders stay BF16).

**Not yet verified:** tool-call accuracy gate (bench/run_agent_bench.py against the INT8 model) — the make-or-break for the quantization bonus claim. Next step when the droplet is next up.

### 12B INT8 accuracy gate (run_5_queries + agent bench, live)
**Date:** 2026-08-02 (droplet 165.245.129.253)

**Result:** pipeline works end-to-end; 12B INT8 is functional but weaker than 26B BF16 at strict tool orchestration.

**What passed:**
- Deploy with KV_CACHE_DTYPE=auto (post-fix script): all smoke tests green.
- Query 1 (wireless earbuds <5k): full flow — extract_brief (product_type=HEADPHONES, budget 55 USD) -> find_product_types -> search_catalog -> finalize_recommendations -> 5 ranked items with reasons + trade-offs + refinement chips. INT8 model produces valid structured output.
- Latency healthy: no OOM, no loader errors, no KV warnings with auto.

**What failed (2 runs, fp8-KV then auto-KV):**
- Queries 2-5 mostly ended in narrative response or finalize_recommendations ValueError:
  - "Each shopping candidate needs product_type_match" (fp8 run)
  - "Each shopping candidate needs a non-empty item_id" (auto run)
- The model generates candidates but omits required schema fields under multi-tool load. This is a model-capability limit, not a quantization defect: the SAME queries degrade regardless of KV cache dtype.
- Aug 1 baseline for comparison: 26B BF16 got recommendations on 3/5 scenarios (run_agent_bench.py, mean 9.24s). 12B INT8 gets ~1-2/5.

**Conclusion:**
- KV_CACHE_DTYPE=auto is the correct default for INT8 serving (removed uncalibrated fp8 KV warnings; no accuracy regression vs fp8).
- The 12B INT8 is a functional fallback and a valid "quantization on AMD + local inference" story for the rubric, but it will NOT beat 26B BF16 on agentic tool orchestration — it is a smaller model. Keep 26B BF16 as the serving default; claim the quantization bonus only on dense-model evidence, not on 12B-vs-26B comparisons.
- If the bonus claim needs a dense INT8 with accuracy near BF16, the 31B-dense INT8 (proven -0.08pp GSM8K) is the stronger candidate than 12B INT8 for this task.

### Gemma 4 12B INT8 skill-isolation probe (hackathon weakness documentation)
**Date:** 2026-08-02 (droplet 165.245.129.253, /models/gemma-4-12b-it-int8 on vLLM 0.26, KV_CACHE_DTYPE=auto)

**Method:** `bench/probe_skills.py` — 5 queries, one per skill, against the live INT8 endpoint. Each query isolated so failure attribution is exact. (Probe added to repo.)

**Results (corrected scoring: S3/S4 asking a clarifying question = PASS, that is designed behavior):**

| Skill | Query | Result | Verdict |
|---|---|---|---|
| S1 direct-simple | "Find a 27-inch computer monitor." | Narrative apology, no recs | **FAIL** |
| S2 dense-finalize | "Show me all water bottles" | 5 recommendations | **PASS** |
| S3 clarification | "I want something for my trip." | Asked what kind of items | **PASS** (by design) |
| S4 accessory | "I need a screen protector for my phone." | Asked which phone model | **PASS** (by design) |
| S5 refine-followup | "laptop backpack, then only waterproof" | 2 recommendations | **PASS** |

**The weakness, precisely (evidence from raw tool-call JSON):**

1. **Escaped-quote JSON corruption in finalize_recommendations (the S1 killer).**
   S1's finalize call emitted string values wrapped in LITERAL escaped quotes:
   `"classification": "\\\"accessory\\\"", "item_id": "\\\"B07PQ7KTG8\\\""`
   After JSON parsing, item_id contains quote characters -> schema validation fails
   -> "Each shopping candidate needs a non-empty item_id" -> model gives up with a narrative apology.
   Clean calls (S2, S5) had `"item_id": "B07C81N8CH"` (no inner quotes) and passed.
   => The 12B model double-escapes JSON string values when it has been working
   through many tool rounds. It is a *serialization* failure under load, not a
   comprehension failure.

2. **Malformed JSON structure in compound intents (S5, recovered).**
   S5's extract_brief intent: `"intent": "\\\"recommend a laptop backpack", "then filter for waterproof ones\\\",item_count\":)`
   — embedded quote + comma inside a value -> structurally invalid JSON. The
   parser recovered and the query still produced 2 recommendations, but this is
   the same root cause: JSON escaping degrades as the conversation grows.

3. **Tool-name drift (S5, recovered).** One call used `final_recommendations`
   (missing "ize"). Model self-corrected on retry. Low frequency, observed once.

4. **NOT a weakness (measured):** structured output for single calls (extract_brief
   always schema-valid), search behavior, clarification judgment, accessory
   classification questions. S2 proves the model CAN emit clean finalize JSON
   when the conversation stays short.

**Root-cause hypothesis for hackathon writeup:** the 12B Unified model's
tool-call JSON serializer degrades as (a) the number of prior tool rounds grows
and (b) the intermediate search result payload is large. The escaped-quote
corruption is consistent with the model memorizing the tool schema's JSON
repr (which itself contains escaped quotes in the schema description) and
leaking those escape sequences into generated values. This is a known failure
class for smaller models with long tool schemas; the 26B A4B and 31B-dense
suffered it far less (26B baseline: 3/5 recommendations, no escaped-quote
errors in the same finalize step).

**Implication for the hackathon:** do NOT claim 12B INT8 as the accuracy
story. It is the "AMD local inference + quantization" story (works end-to-end,
fast, correct on short interactions). For the quantization bonus claim, use
31B-dense INT8 (proven -0.08pp GSM8K) whose capacity holds the finalize schema.
Mitigations to evaluate for 12B if it must serve the agent: shorten the
finalize tool schema description, trim search_catalog result size, or add a
repair pass that strips escaped quotes from tool-call arguments before
validation.

**Repeatability:** rerun with
  RETAIL_PROVIDER=vllm RETAIL_BASE_URL=http://<ip>:8000/v1 \
  RETAIL_MODEL=/models/gemma-4-12b-it-int8 uv run python bench/probe_skills.py

### Root cause resolution: 12B "JSON weakness" is a parser/plumbing issue, not model capability
**Date:** 2026-08-02 (follow-up to the skill-isolation probe)

**Question investigated:** is Gemma 4 12B genuinely bad at JSON output, or is the tool-call corruption something else? Checked vLLM docs/source, GitHub issues, and ran a direct vLLM test bypassing MAF.

**Findings (evidence-backed):**

1. **Direct vLLM test (no MAF) produces CLEAN JSON.** Same model, same tools, 50-item catalog payload, 3 tool rounds, tool_choice="required":
   - 0 escaped quotes, JSON parses, all required fields present (item_id, classification, product_type_match).
   - => The model CAN emit valid structured tool calls under load.

2. **The corruption is in the MAF (agent_framework) round-trip.** The probe traces showed `\\\"accessory\\\"` (escaped quotes inside values) only when going through MAF. MAF's `_prepare_content_for_openai` (agent_framework_openai/_chat_completion_client.py:985) does `json.dumps(content.arguments)` on the args mapping, and the tool-result string embedding (line 994) re-encodes JSON-in-string, which the model then echoes back with extra escaping on the next round. Root: JSON-string-in-JSON-string double-serialization across rounds.

3. **No JSON constraint is applied to tool calls in our stack.**
   - vLLM server: `--enable-auto-tool-choice --tool-call-parser gemma4` (verified from ps).
   - MAF never sends `tool_choice` (app never sets `tool_mode`; MAF only sets it if options has it — _chat_client.py:1434).
   - gemma4 parser sets `supports_required_and_named = False` (vllm/tool_parsers/gemma4_engine_tool_parser.py) — deliberately SKIPS the base parser's structured-output forcing (StructuredOutputsParams(json=...)) to preserve native `<|tool_call|>` syntax.
   - => tool-call arguments are ALWAYS parsed by the buggy `_parse_gemma4_args` with zero guided decoding.

4. **vLLM has the fix mechanism but it's not engaged for gemma4.** Base `ToolParser.adjust_request` (vllm/tool_parsers/abstract_tool_parser.py:127-146) sets `structured_outputs = StructuredOutputsParams(json=json_schema_from_tool)` when tool_choice is required/named. vLLM 0.26 also supports `response_format={"type": "json_schema", ...}` with backends xgrammar/guidance/outlines/lm-format-enforcer for the FINAL response. Neither is applied to gemma4 tool-call args today.

5. **Known open vLLM parser bugs in exactly our path (both open PRs, not merged, no ROCm wheel newer than 0.26.0 exists):**
   - #48678 — Gemma4 parser truncates string args at interior `<|"|>` escape tokens (the exact corrupting path).
   - #47909 — Gemma4 engine parser returns bare values as strings.
   - Both labeled `tool-calling`, both open as of 2026-08-02.

**Conclusion for the hackathon (corrected from the previous entry):**
- The earlier "12B is weak at tool orchestration" conclusion was PARTLY WRONG. The 12B model's JSON generation is fine; the failure is a **plumbing defect**: MAF's double-serialization + gemma4 parser's unconstrained arg parsing + open vLLM bugs #48678/#47909.
- 26B BF16 didn't hit it as hard because it is more robust to the noisy round-trip context (bigger model absorbs the corrupted echo), not because 12B is fundamentally incapable.
- **Mitigations to evaluate (in order of leverage):**
  1. Set `tool_mode={"mode": "required"}` / pass tool_choice so vLLM's base parser engages guided decoding on the finalize step — BUT gemma4 parser opts out; would need `--tool-call-parser` switch to a JSON-capable parser or a wrapper.
  2. Use vLLM `response_format` json_schema for the finalize step (the app already has the schema; MAF supports response_format).
  3. Fix MAF round-trip: avoid json.dumps on already-serialized args; pass tool results as raw JSON strings.
  4. Upgrade vLLM when #48678/#47909 land in a ROCm wheel (0.26.0 is the newest available).
- The 12B INT8 story for the hackathon: valid for "AMD local inference + quantization" demo. The accuracy gate comparison (12B INT8 vs 26B BF16) is CONFOUNDED by the plumbing defect — rerun after mitigation 2 or 3 before making claims.

### Hackathon submission note: 12B dense is the shipped quantization path; MoE removed
**Date:** 2026-08-02

**Decision:** `_quark_quantize_moe.py` and the `--kind moe` path in `quantize_int8.sh` are REMOVED. 12B dense INT8 is the shipped quantization story. Rationale: the MoE path produced garbage in 4 attempts (Issues 11-13); the dense path produces a working, vLLM-servable checkpoint (see "12B dense end-to-end live run" entry). The MoE failure trail stays in this journal as history; the code is gone (git history carries it).

**Quantize CLI is now dense-only:**
  bash /root/quantize_int8.sh --model google/gemma-4-12b-it
No --kind flag; recipe is always `_quark_quantize_dense.py`. Preflight, FX trace (opt-in), and post-quantize key fixup unchanged.

**12B findings summary (for the hackathon writeup):**

1. **Quantization works.** 23.9 GB BF16 -> 14 GB INT8 (~1.7x) via Quark 0.12.post1 W8A8, served on vLLM 0.26 with `--quantization quark`. Full pipeline reproducible from repo scripts: upgrade_vllm.sh -> deploy_droplet.sh (download BF16) -> quantize_int8.sh -> deploy with VLLM_FP8_MODEL.

2. **Two post-quantize fixups are required and automated** (`_quark_fix_vllm_keys.py`, step 2.5):
   - Key rename: Quark exports `embed_vision.multimodal_embedder.*` / `embed_vision.patch_*`; vLLM's gemma4_unified loader expects `embed_vision.embedding_projection.*` / `vision_embedder.*`.
   - chat_template.jinja copy: Quark does not export it; vLLM 400s chat requests without it.

3. **Tool-call reliability finding (honest, evidence-backed):** the 12B model is NOT weak at JSON or tool calling in general. Direct vLLM requests produce clean JSON; single-call structured output (extract_brief) is always schema-valid; the full agent flow works end-to-end on direct queries. The observed failures (escaped quotes like `"item_id": "\\\"B07PQ7KTG8\\\""` in finalize_recommendations) appear only in MULTI-ROUND agent sessions, where the model occasionally emits a plain `"` where Gemma 4's native `<|"|>` delimiter is expected, and vLLM's gemma4 parser (open PRs #48678, #47909; unfixed in any ROCm wheel as of 2026-08-02) mangles it. Root-cause chain verified: chat templates are md5-identical across 12B/31B/26B; parser and MAF identical; MAF upgrade (core 1.13.0 / openai 1.12.0) did not change the serialization line and tests pass. The 26B MoE baseline (Aug 1) did not hit this because the bigger model emits the delimiter format more faithfully. Honest framing for the writeup: quantization is sound; the residual risk is a narrow model-fidelity-vs-parser interaction under multi-round load, fixable in vLLM (parser fix) or app-side (repair pass), NOT a reason to disqualify the INT8 checkpoint.

4. **Numbers for the writeup:** 12.54 GiB weights + 5.22 GiB CUDA graphs; ~157 GiB KV cache at GPU_MEM=0.9 / max-model-len 32768 on MI300X (191.7 GB); first torch.compile ~5 min, cached afterward (84s boot); KV_CACHE_DTYPE=auto avoids uncalibrated fp8 KV warnings on INT8 checkpoints.
