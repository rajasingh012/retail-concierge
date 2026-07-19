# RetailConcierge

Three collaborating agents built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. The user is in the loop: every catalog query the agents want to make is shown to the user for approval, edit, or redirect before it runs.

The catalog is an offline snapshot of the Amazon product dataset. The system never claims current prices, availability, shipping, or product specifications absent from the catalog.

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

- [architecture.md](architecture.md) — agents, tools, approval gate, storage, and data flow
- [deploy.md](deploy.md) — catalog import and AMD MI300X deployment
- [progress.md](progress.md) — forward-looking work only
