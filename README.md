# RetailConcierge

One conversational retail agent built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. The agent keeps a multi-turn `AgentSession`, asks only blocking clarification questions, and uses schema-aware tools over an offline Amazon Berkeley Objects catalog.

SQLite FTS5 retrieves candidates from 145K products across 576 product types. The agent classifies exact products versus accessories and unrelated items; application code enforces that eligibility decision and deterministically ranks the remaining catalog evidence.

The system never claims current prices, availability, shipping, ratings, or specifications absent from the catalog, and it does not add items to a cart or make purchases.

![Track 2: Agentic AI](https://img.shields.io/badge/AMD-AI--DevMaster%202026-CC0000) ![GPU: MI300X](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300X-FF6B00) ![ROCm 7.2.3](https://img.shields.io/badge/ROCm-7.2.3-0086CB) ![vLLM 0.26.0+rocm723](https://img.shields.io/badge/vLLM-0.26.0%2Brocm723-7B68EE) ![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB) ![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue)

## Live demo

**👉 [retail-concierge-9fz4fe3znfxcvqiqsncwxn.streamlit.app](https://retail-concierge-9fz4fe3znfxcvqiqsncwxn.streamlit.app/)** — Streamlit Community Cloud, served on a free-tier CPU.

The deployed demo runs against the **chair-only subset** of the catalog (2,173 listings across CHAIR + BEAN_BAG_CHAIR; built by [`scripts/build_chair_demo_db.py`](scripts/build_chair_demo_db.py)). The full catalog used in the AMD MI300X bench (145K listings, 576 product types) is too large to ship to Streamlit's free-tier disk quota, so the live demo is intentionally limited to the chairs subcategory. The agent code is unchanged — every query path, ranking weight, and provenance gate runs identically against the chair subset.

## Quick start

```bash
git clone https://github.com/rajasingh012/retail-concierge.git
cd retail-concierge
uv sync

# Build the ABO catalog (requires the abo-listings.tar.gz dataset)
uv run python scripts/import_catalog.py --shards data/abo/listings/

# Default: DeepSeek V4 Flash (set DEEPSEEK_API_KEY in your environment)
uv run python main.py

# AMD MI300X (vLLM with Gemma 4 12B W8A8 INT8, AMD Quark)
export DROPLET="<your-droplet-ip>"
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL="http://$DROPLET:8000/v1" \
RETAIL_MODEL=rajasingh012/gemma-4-12b-it-quark-w8a8-int8 \
  uv run python main.py
```

The 12B W8A8 INT8 checkpoint (published on [Hugging Face](https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8)) is served with vLLM 0.26's `--quantization quark` loader. Earlier BF16 paths (31B dense, 26B A4B MoE) were measured on vLLM 0.23; the MoE quantization path was removed — see [scripts/README.md](scripts/README.md).

## AMD Radeon performance (measured on MI300X, vLLM 0.26, ROCm 7.2.3)

W8A8 INT8 12B, `--tool-call-parser gemma4`, chunked prefill + prefix caching enabled:

| Metric | Value |
|---|---|
| Output throughput (single stream) | **49.8 tok/s** |
| Peak throughput | **51.0 tok/s** |
| Median TTFT | ~55 ms |
| TPOT | ~19.8 ms |
| Benchmark | 10,240 input → 1,280 generated tokens in 25.7 s |

## Scripts

Droplet lifecycle + catalog utilities — see [scripts/README.md](scripts/README.md) for what each script does, where it runs (laptop vs droplet), and the ordering.

**Published quantization:** the Gemma 4 12B W8A8 INT8 checkpoint (AMD Quark) is public on Hugging Face — **[`rajasingh012/gemma-4-12b-it-quark-w8a8-int8`](https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8)**. It is the first AMD Quark W8A8 INT8 quantization of Gemma 4 12B, produced by `scripts/quantize_int8.sh`, served with vLLM 0.26's `--quantization quark` loader (needs the Quark→vLLM key fixup + `chat_template.jinja` copy, both automated in `scripts/_quark_fix_vllm_keys.py`). Model card covers the W8A8 scheme, the 23.9 GB → 13 GB compression, and an honest accuracy caveat (GSM8K −0.08pp was measured on the 31B class, not 12B Unified).

**Known limitation:** W8A8 INT8 quantization ships on the **12B dense** path (works end-to-end on vLLM 0.26, served via `--quantization quark`). The 26B A4B MoE path was removed — Quark W8A8 INT8 on the MoE produced garbage in 4 attempts (DEPLOYMENT_JOURNAL.md Issues 11-13); the 26B ships as BF16. Accuracy caveat: the 31B-dense baseline (−0.08pp GSM8K) was measured on the older `Gemma4ForConditionalGeneration` class; 12B uses `Gemma4UnifiedForConditionalGeneration` — treat 12B as a fresh run and verify via the tool-call accuracy gate before claiming the quantization bonus. See [scripts/README.md](scripts/README.md).

## Documentation

- [architecture.md](architecture.md) — current design and data flow
- [deploy.md](deploy.md) — catalog, inference, and benchmark commands
- [progress.md](progress.md) — forward-looking work
- [DEPLOYMENT_JOURNAL.md](DEPLOYMENT_JOURNAL.md) — real issues hit on AMD + fixes
- [docs/hackathon_spec_document.md](docs/hackathon_spec_document.md) — Track 2 submission spec (AMD AI DevMaster 2026)
- [docs/hackathon_poster_checklist.md](docs/hackathon_poster_checklist.md) — poster slides + submission checklist
- [docs/demo_video_script.md](docs/demo_video_script.md) — demo video shot list

## Hackathon: AMD AI DevMaster 2026

Submitted to **Track 2: Agentic AI** — reasoning, planning, tool use, memory, RAG, multi-agent systems. Local inference on AMD Radeon GPUs is a judging requirement.

- **Repo:** https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07
- **Deadline:** Aug 6, 2026 (PDT 8:59 AM, CEST 5:59 PM, UTC+8 11:59 PM)
- **Prize pool:** $30,000 USD (1st $5K / 2nd $3.5K / 3rd $1.5K)
- **Eligibility:** AMD AI Developer Program membership required before joining the event
- **Submit requirements:** local inference on Radeon GPUs, inference speed optimization, functional completeness, agentic UX
- **Discord:** https://discord.gg/zt9caur5B3
- **Contact:** ai_dev_contests@amd.com
