# RetailConcierge

Multi-agent retail assistant on AMD Radeon ROCm. Two MAF agents (Discovery → Synthesis) over a BM25 + SQLite catalog, with live product lookups via the Amazon scraper in our vendored E-Commerces-WebScraper submodule. vLLM runs on an AMD Developer Cloud MI300X droplet.

For this hackathon we ship with the **Amazon** scraper path. The submodule also includes AliExpress / Shein / Shopee / Mercado Livre scrapers — we don't disable them, just don't invoke them. Switching platforms is a one-line change in `ECommerceAdapter.fetch_product(url, platform="aliexpress")`.

## Quick start

```bash
git clone https://github.com/rajasingh012/retail-concierge.git
cd retail-concierge

# Install deps (uv ~10× faster than pip; uv.lock ensures reproducibility)
uv sync

# Install Playwright browsers
uv run playwright install chromium

# AMD Dev Cloud MI300X (judge demo)
./scripts/serve-vllm-rocm.sh        # in one terminal, after SSH'ing into the droplet
RETAIL_PROVIDER=vllm RETAIL_BASE_URL=http://<droplet>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
    uv run python main.py           # on your laptop

# Local-dev fallback (no GPU droplet needed)
RETAIL_PROVIDER=deepseek RETAIL_MODEL=deepseek-chat \
    DEEPSEEK_API_KEY=*** uv run python main.py
```

## Docs

- [architecture.md](architecture.md) — layers, agents, Protocol, data flow
- [deploy.md](deploy.md) — AMD Dev Cloud bring-up, firewall, spending limits
- [progress.md](progress.md) — what's next (P0/P1/P2)