# RetailConcierge

One conversational retail agent built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. The agent keeps a multi-turn `AgentSession`, asks only blocking clarification questions, and uses schema-aware tools over an offline Amazon Berkeley Objects catalog.

SQLite FTS5 retrieves candidates from 145K products across 576 product types. The agent classifies exact products versus accessories and unrelated items; application code enforces that eligibility decision and deterministically ranks the remaining catalog evidence.

The system never claims current prices, availability, shipping, ratings, or specifications absent from the catalog, and it does not add items to a cart or make purchases.

## Quick start

```bash
git clone https://github.com/rajasingh012/retail-concierge.git
cd retail-concierge
uv sync

# Build the ABO catalog (requires the abo-listings.tar.gz dataset)
uv run python scripts/import_catalog.py --shards data/abo/listings/

# Default: MiniMax-M3 (set MINIMAX_API_KEY in your environment)
uv run python main.py

# AMD MI300X (vLLM with Gemma 4)
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-4-31b-it \
  uv run python main.py

# Local development via DeepSeek
export DEEPSEEK_API_KEY=***
RETAIL_PROVIDER=deepseek RETAIL_MODEL=deepseek-chat uv run python main.py
```

## Documentation

- [architecture.md](architecture.md) — current design and data flow
- [deploy.md](deploy.md) — catalog, inference, and benchmark commands
- [progress.md](progress.md) — forward-looking work

## Hackathon: AMD AI DevMaster 2026

Submitted to **Track 2: Agentic AI** — reasoning, planning, tool use, memory, RAG, multi-agent systems. Local inference on AMD Radeon GPUs is a judging requirement.

- **Repo:** https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07
- **Deadline:** Aug 6, 2026 (PDT 8:59 AM, CEST 5:59 PM, UTC+8 11:59 PM)
- **Prize pool:** $30,000 USD (1st $5K / 2nd $3.5K / 3rd $1.5K)
- **Eligibility:** AMD AI Developer Program membership required before joining the event
- **Submit requirements:** local inference on Radeon GPUs, inference speed optimization, functional completeness, agentic UX
- **Discord:** https://discord.gg/zt9caur5B3
- **Contact:** ai_dev_contests@amd.com
