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
2. `deploy_droplet.sh` — preflight → download BF16 model → launch vLLM on :8000 with `--tool-call-parser gemma4` + chunked prefill + prefix caching. Honors `VLLM_FP8_MODEL` for serving a quantized checkpoint. KV cache dtype auto-tuned (`KV_CACHE_DTYPE`). *(Droplet.)*
3. `quantize_int8.sh --model <hf-repo>` — AMD Quark W8A8 INT8 of the BF16 dense model via `_quark_quantize_dense.py` (12B / 31B Unified). Includes preflight, optional FX trace, and the post-quantize vLLM key fixup. *(Droplet.)*
4. `benchmark_concurrency.sh` — concurrency-1/2/4/8 sweep against the live endpoint. *(Laptop, hits the droplet.)*

## Catalog scripts (laptop)

- `import_catalog.py` — ingest `abo-listings.tar.gz` shards into the SQLite catalog via `infrastructure.database.create_schema`.
- `audit_verify.py` — stdlib-only verifier for the tamper-evident audit JSONL chain (truncation, reorder, tampering). Works without the project venv.

## Helpers

- `_find_bf16.py --model <substring>` — prints the HF cache snapshot path for the BF16 model whose `repo_id` contains `<substring>`. Accepts both sharded (index.json) and single-file (model.safetensors) checkpoints. Used by the quantize preflight.
- `_quark_common.py` — shared Quark helpers (load BF16, build W8A8 specs, quantize+export). Imported by the dense recipe.
- `_quark_quantize_dense.py` — dense-model W8A8 INT8 recipe (per-channel weight + per-token dynamic activation). Exclude list adapted for Gemma 4 12B Unified — **must use Quark's post-rename names** (`model.embed_vision.*`, `model.embed_audio.*`, `lm_head`), not the HF names, because Quark renames `vision_embedder` → `embed_vision.patch_*` before matching (hit live on droplet 2026-08-02). embed_tokens is quantized to match the proven 31B recipe. **Caveat:** the 31B baseline accuracy (−0.08pp GSM8K) was measured on the older `Gemma4ForConditionalGeneration` class, not 12B's `Gemma4UnifiedForConditionalGeneration`. Treat 12B as a fresh run; verify via the tool-call accuracy gate before claiming the bonus.
- `_quark_fix_vllm_keys.py` — post-quantize fixup run automatically by `quantize_int8.sh`. Renames Quark 0.12's `embed_vision.multimodal_embedder.*` / `embed_vision.patch_*` keys to the layout vLLM 0.26's `gemma4_unified` loader expects, and copies `chat_template.jinja` into the output (Quark doesn't export it; vLLM 400s without it). Idempotent, safe to re-run. Skippable via `SKIP_VLLM_KEY_FIX=1`.

## Known issue (current)

The 26B A4B **MoE** quantization path was removed (2026-08-02): Quark W8A8 INT8 on the MoE produced garbage in 4 attempts (DEPLOYMENT_JOURNAL.md Issues 11–13). The shipped quantization path is **12B dense INT8** (works end-to-end on vLLM 0.26) — the resulting checkpoint is published publicly at **[`rajasingh012/gemma-4-12b-it-quark-w8a8-int8`](https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8)** (Hugging Face), the first AMD Quark W8A8 INT8 of Gemma 4 12B. Accuracy gate still pending (GSM8K −0.08pp was measured on the 31B class, not 12B Unified).

Tool-call reliability note: under multi-round agent sessions the 12B occasionally emits plain `"` where the Gemma 4 native `<|"|>` delimiter is expected, and vLLM's gemma4 parser (open bugs #48678/#47909) can corrupt the final `finalize_recommendations` arguments. This is model-generation fidelity under context pressure, not a JSON-capability failure — single-call structured output and direct vLLM requests are clean. See DEPLOYMENT_JOURNAL.md "Root cause resolution".

See `../deploy.md` for the full command sequence and `../DEPLOYMENT_JOURNAL.md` for the live-issue trail.
