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

## Scripts

Droplet lifecycle + catalog utilities — see [scripts/README.md](scripts/README.md) for what each script does, where it runs (laptop vs droplet), and the ordering.

**Known limitation:** W8A8 INT8 works on dense models (31B proven −0.08pp GSM8K on the older `Gemma4ForConditionalGeneration` class; 12B target on `Gemma4UnifiedForConditionalGeneration` — accuracy not yet measured) but **fails on the 26B A4B MoE** — 4 attempts, all load but produce garbage (DEPLOYMENT_JOURNAL.md Issues 11-13). The 26B ships as BF16; quantization bonus is claimable only on a proven dense path. See [scripts/README.md](scripts/README.md) for recipe split + Unified-vs-older caveat.

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
