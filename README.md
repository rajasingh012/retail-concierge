# RetailConcierge

One conversational retail agent built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. The agent keeps a multi-turn `AgentSession`, asks only blocking clarification questions, and uses schema-aware tools over an offline Amazon Berkeley Objects catalog.

SQLite FTS5 retrieves candidates from 145K products across 576 product types. The agent classifies exact products versus accessories and unrelated items; application code enforces that eligibility decision and deterministically ranks the remaining catalog evidence.

The system never claims current prices, availability, shipping, ratings, or specifications absent from the catalog, and it does not add items to a cart or make purchases.

![Track 2: Agentic AI](https://img.shields.io/badge/AMD-AI--DevMaster%202026-CC0000) ![GPU: MI300X](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300X-FF6B00) ![ROCm 7.2.3](https://img.shields.io/badge/ROCm-7.2.3-0086CB) ![vLLM 0.23.0](https://img.shields.io/badge/vLLM-0.23.0-7B68EE) ![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB) ![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue)

## Quick start

```bash
git clone https://github.com/rajasingh012/retail-concierge.git
cd retail-concierge
uv sync

# Build the ABO catalog (requires the abo-listings.tar.gz dataset)
uv run python scripts/import_catalog.py --shards data/abo/listings/

# Default: DeepSeek V4 Flash (set DEEPSEEK_API_KEY in your environment)
uv run python main.py

# AMD MI300X (vLLM with Gemma 4 26B A4B MoE)
export DROPLET="<your-droplet-ip>"
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL="http://$DROPLET:8000/v1" \
RETAIL_MODEL=google/gemma-4-26B-A4B-it \
  uv run python main.py

# Switch to the dense 31B variant (slower, stronger reasoning)
RETAIL_MODEL=google/gemma-4-31B-it uv run python main.py
```

## AMD Radeon performance (measured on MI300X, vLLM 0.23, ROCm 7.2.3)

| Metric | 31B Dense | **26B A4B MoE** |
|---|---|---|
| Single throughput | 150 tok/s | **427 tok/s** |
| Concurrency-8 throughput | 652 tok/s | **1,575 tok/s** |
| Median TTFT (server-side) | ~35ms | **~35ms** |
| Prefix cache hit rate | 64% | **64%** |
| Model VRAM | 58.9 GiB | **48.5 GiB** |

The MoE variant uses 3.8B active parameters per token and is **3.69× faster** than the dense 31B at concurrency-8 with the same tool-calling flags (`--tool-call-parser gemma4`, `--kv-cache-dtype fp8`, `--enable-prefix-caching`).

## GPU droplet workflow (scripts/)

Three orchestrator scripts manage the AMD MI300X droplet lifecycle. Run them ON the droplet (via `ssh root@<ip> 'bash -s' < script.sh` or scp + bash). The app + catalog DB stay on this machine.

```bash
# 1. Upgrade vLLM from the 1-Click image's 0.23 to 0.26 (one-time per droplet).
#    Required for serving Quark-quantized checkpoints (vLLM 0.23 MoE loader bug).
ssh root@$DROPLET 'bash -s' < scripts/upgrade_vllm.sh

# 2. Deploy: pull BF16 model into the container + start vLLM serving on :8000.
ssh root@$DROPLET 'bash -s' < scripts/deploy_droplet.sh
#    Set VLLM_FP8_MODEL=/models/<quark-output> to serve a quantized checkpoint instead.

# 3. Quantize the BF16 model with AMD Quark (W8A8 INT8).
#    Uses scripts/_quark_quantize_int8.py (dense recipe — for dense models
#    like Gemma 4 12B/31B). Output: /models/gemma-4-26B-A4B-it-int8/
ssh root@$DROPLET 'bash -s' < scripts/quantize_int8.sh

# 4. Benchmark: concurrency scaling 1→2→4→8 (run from THIS machine).
BASE_URL=http://$DROPLET:8000/v1 MODEL=google/gemma-4-26B-A4B-it \
  bash scripts/benchmark_concurrency.sh
```

- `upgrade_vllm.sh` — vLLM 0.23 → 0.26 via the AMD-sanctioned upstream wheel index (`wheels.vllm.ai/rocm/0.26.0/rocm723`). Fixes ABI breaks (flash-attn, torchaudio, torch_c_dlpack_ext) and verifies imports. See DEPLOYMENT_JOURNAL.md Issue 12.
- `deploy_droplet.sh` — preflight (GPU + container), model download (HF, `max_workers=4`), launch with AITER pinned + FP8 KV cache + chunked prefill + prefix caching + `--tool-call-parser gemma4`, startup polling, smoke tests.
- `quantize_int8.sh` — AMD Quark W8A8 INT8 quantization (orchestrator; the actual quantizer is `_quark_quantize_int8.py` for dense models, `_quark_quantize_moe.py` for MoE). Handles preflight, runs the quantizer in-container, verifies output, prints next steps.
- **Known limitation:** W8A8 INT8 works on dense models (31B proven −0.08pp GSM8K; 12B test planned) but **fails on the 26B A4B MoE** — 4 attempts, all load but produce garbage (DEPLOYMENT_JOURNAL.md Issues 11-13). The 26B ships as BF16; quantization bonus is claimable only via a dense model (e.g. 12B).

## Documentation

- [architecture.md](architecture.md) — current design and data flow
- [deploy.md](deploy.md) — catalog, inference, and benchmark commands
- [progress.md](progress.md) — forward-looking work
- [DEPLOYMENT_JOURNAL.md](DEPLOYMENT_JOURNAL.md) — real issues hit on AMD + fixes

## Hackathon: AMD AI DevMaster 2026

Submitted to **Track 2: Agentic AI** — reasoning, planning, tool use, memory, RAG, multi-agent systems. Local inference on AMD Radeon GPUs is a judging requirement.

- **Repo:** https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07
- **Deadline:** Aug 6, 2026 (PDT 8:59 AM, CEST 5:59 PM, UTC+8 11:59 PM)
- **Prize pool:** $30,000 USD (1st $5K / 2nd $3.5K / 3rd $1.5K)
- **Eligibility:** AMD AI Developer Program membership required before joining the event
- **Submit requirements:** local inference on Radeon GPUs, inference speed optimization, functional completeness, agentic UX
- **Discord:** https://discord.gg/zt9caur5B3
- **Contact:** ai_dev_contests@amd.com
