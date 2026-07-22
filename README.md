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

# AMD MI300X (vLLM)
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
  uv run python main.py

# Local development via DeepSeek
export DEEPSEEK_API_KEY=***
RETAIL_PROVIDER=deepseek RETAIL_MODEL=deepseek-chat uv run python main.py
```

## Documentation

- [architecture.md](architecture.md) — current design and data flow
- [deploy.md](deploy.md) — catalog, inference, and benchmark commands
- [progress.md](progress.md) — forward-looking work
