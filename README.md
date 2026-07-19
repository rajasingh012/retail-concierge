# RetailConcierge

Collaborative retail recommendation agents built with Microsoft Agent Framework and served by vLLM on an AMD MI300X. Discovery clarifies the request, Catalog Research gathers evidence from an offline Amazon dataset, and a Critic independently checks and ranks the recommendations.

The application uses dataset snapshots, not live Amazon data. It does not claim current prices, availability, shipping, or product specifications absent from the catalog.

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

- [architecture.md](architecture.md) — agents, tools, storage, and data flow
- [deploy.md](deploy.md) — catalog import and AMD MI300X deployment
- [progress.md](progress.md) — forward-looking work only
