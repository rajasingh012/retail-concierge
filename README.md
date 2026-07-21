# RetailConcierge

Three collaborating agents built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. SQLite FTS5 retrieves candidates from the **Amazon Berkeley Objects** offline catalog (145K products, 576 product types). The Research agent excludes accessories and uncertain product types, deterministic catalog-signal ranking orders eligible products, and the Critic reviews with evidence-backed trade-offs.

The system never claims current prices, availability, shipping, or specifications absent from the catalog, and does not add items to a cart or make purchases.

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

- [progress.md](progress.md) — forward-looking work only
