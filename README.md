# RetailConcierge

Three collaborating agents built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. The Discovery agent asks a clarification only when a missing preference would materially affect the recommendation; read-only catalog lookups then run automatically.

The catalog is an offline snapshot of the Amazon product dataset. The system never claims current prices, availability, shipping, or product specifications absent from the catalog, and it does not add items to a cart or make purchases.

## Quick start

```bash
git clone https://github.com/rajasingh012/retail-concierge.git
cd retail-concierge
uv sync

# Build the local catalog from the external dataset
uv run python scripts/import_catalog.py \
  --products /home/rajasingh/Downloads/archive/amazon_products.csv \
  --categories /home/rajasingh/Downloads/archive/amazon_categories.csv

# Judge/demo backend: vLLM on AMD Developer Cloud
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
  uv run python main.py

# Local development fallback
RETAIL_PROVIDER=deepseek RETAIL_MODEL=deepseek-chat \
DEEPSEEK_API_KEY=*** uv run python main.py
```

## Documentation

- [architecture.md](architecture.md) — agents, tools, clarification loop, storage, and data flow
- [deploy.md](deploy.md) — catalog import and AMD MI300X deployment
- [progress.md](progress.md) — forward-looking work only
