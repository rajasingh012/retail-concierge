# RetailConcierge

Multi-agent retail assistant on AMD Radeon ROCm. Two MAF agents (Discovery → Synthesis) over a BM25 + SQLite catalog, talking to vLLM on an AMD Developer Cloud MI300X droplet.

## Run

```bash
pip install -r requirements.txt
playwright install chromium

# AMD Dev Cloud MI300X (judge demo)
./scripts/serve-vllm-rocm.sh        # in one terminal, after SSH'ing into the droplet
RETAIL_PROVIDER=vllm RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
    PYTHONPATH=. python main.py     # on your laptop

# Local-dev fallback (no GPU droplet needed)
RETAIL_PROVIDER=deepseek RETAIL_MODEL=deepseek-chat \
    DEEPSEEK_API_KEY=*** PYTHONPATH=. python main.py
```

## Docs

- [architecture.md](architecture.md) — layers, agents, Protocol, data flow
- [deploy.md](deploy.md) — AMD Dev Cloud bring-up, firewall, spending limits
- [progress.md](progress.md) — what's next (P0/P1/P2)