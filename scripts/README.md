# scripts/

Scripts that manage the catalog and the AMD MI300X GPU droplet lifecycle. The application (app / main / MAF agent) and the SQLite catalog stay on the developer's laptop. The droplet only ever runs `vLLM` and, transiently, the Quark quantizer.

For the executable how-to (commands, ordering), see the per-script header comments. This README is the **what + when + where** map — source is the source of truth.

## Architecture boundary

| Host        | Lives here                          | Must NOT live here                |
|-------------|--------------------------------------|-----------------------------------|
| Laptop      | app, MAF agent, SQLite catalog, benchmarks | vLLM server, /models           |
| Droplet     | vLLM, HF model cache, Quark quantizer, /models | app code, catalog DB         |

`scp` moves data across the boundary; the repo is never cloned onto the droplet.

## Lifecycle order

1. `upgrade_vllm.sh` — one-time per droplet, upgrades vLLM 0.23 → 0.26 inside the `rocm` container. Required to serve Quark-quantized checkpoints. *(Droplet.)*
2. `deploy_droplet.sh` — preflight → download BF16 model → launch vLLM on :8000 with `--tool-call-parser gemma4` + chunked prefill + prefix caching + FP8 KV cache. Honors `VLLM_FP8_MODEL` for serving a quantized checkpoint. *(Droplet.)*
3. `quantize_int8.sh --kind dense|moe --model <hf-repo>` — AMD Quark W8A8 INT8 of the BF16 model. Dispatches to `_quark_quantize_dense.py` (production path: 12B / 31B) or `_quark_quantize_moe.py` (26B A4B MoE — known broken, kept for the failure trail). *(Droplet.)*
4. `benchmark_concurrency.sh` — concurrency-1/2/4/8 sweep against the live endpoint. *(Laptop, hits the droplet.)*

## Catalog scripts (laptop)

- `import_catalog.py` — ingest `abo-listings.tar.gz` shards into the SQLite catalog via `infrastructure.database.create_schema`.
- `probe_daily.py` — daily freshness probe; flips `url_active` 1→0 on hard dead signals, aborts on CAPTCHA / bot-wall. Cron-friendly exit codes (0/1/2).
- `audit_verify.py` — stdlib-only verifier for the tamper-evident audit JSONL chain (truncation, reorder, tampering). Works without the project venv.

## Helpers

- `_find_bf16.py --model <substring>` — prints the HF cache snapshot path for the BF16 model whose `repo_id` contains `<substring>`. Default substring `gemma-4-26B-A4B-it` preserves the original behavior. Used by the quantize dispatcher to locate the right snapshot for each `--kind`.
- `_quark_common.py` — shared Quark helpers (load BF16, build W8A8 specs, quantize+export). Imported by both recipe scripts; no executable logic of its own.
- `_quark_quantize_dense.py` — dense-model W8A8 INT8 recipe (per-channel weight + per-token dynamic activation). Exclude list adapted for Gemma 4 12B Unified — **must use Quark's post-rename names** (`model.embed_vision.*`, `model.embed_audio.*`, `lm_head`), not the HF names, because Quark renames `vision_embedder` → `embed_vision.patch_*` before matching (hit live on droplet 2026-08-02). embed_tokens is quantized to match the proven 31B recipe. **Caveat:** the 31B baseline accuracy (−0.08pp GSM8K) was measured on the older `Gemma4ForConditionalGeneration` class, not 12B's `Gemma4UnifiedForConditionalGeneration`. Treat 12B as a fresh run; verify via the tool-call accuracy gate before claiming the bonus.
- `_quark_quantize_moe.py` — MoE-aware W8A8 INT8 recipe: `split_fused_experts()` rewrite before quantization, MoE-tuned exclude list (router + shared-expert-gate), `rename_keys()` after export. Targets the older `Gemma4ForConditionalGeneration` class (26B A4B). **Known limitation:** loads on 26B A4B MoE but produces garbage; 4 documented attempts. The MoE ships as BF16.
- `_quark_fix_vllm_keys.py` — post-quantize fixup run automatically by `quantize_int8.sh` on the dense path. Renames Quark 0.12's `embed_vision.multimodal_embedder.*` / `embed_vision.patch_*` keys to the layout vLLM 0.26's `gemma4_unified` loader expects, and copies `chat_template.jinja` into the output (Quark doesn't export it; vLLM 400s without it). Idempotent, safe to re-run. Skippable via `SKIP_VLLM_KEY_FIX=1`.

## Known issue (current)

W8A8 INT8 is claimable only for **dense** models (31B proven, 12B planned). The 26B A4B MoE is served BF16 — see `DEPLOYMENT_JOURNAL.md` Issues 11–13.

See `../deploy.md` for the full command sequence and `../DEPLOYMENT_JOURNAL.md` for the live-issue trail.
