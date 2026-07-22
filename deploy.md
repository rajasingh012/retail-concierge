# Deployment

## Build the offline catalog

The source dataset stays outside Git. Import it on the machine running the application:

```bash
uv sync
mkdir -p data/abo
# Place abo-listings.tar.gz in data/abo/ and extract:
cd data/abo && tar xzf abo-listings.tar.gz && cd ../..
uv run python scripts/import_catalog.py --shards data/abo/listings/
```

The importer reads gzipped NDJSON shards, builds the listing, structured text-value, dimension, and FTS5 tables, and runs integrity checks. Add `--limit 50000` for a smaller catalog.

## AMD Developer Cloud

Use one MI300X GPU Droplet with the vLLM 1-Click image. The available credit covers GPU usage only; set a hard spending cap before creating the droplet and destroy it after each session.

1. Create a cloud firewall allowing SSH and port 8000 only from your public IP.
2. Create the MI300X droplet and add your SSH key.
3. Verify the GPU and vLLM container:

```bash
amd-smi --showproductname --showuse --showmeminfo vram
docker ps
```

4. Start vLLM:

```bash
./scripts/serve-vllm-rocm.sh
```

5. Run the catalog and agent on the laptop, pointing inference at the droplet:

```bash
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
  uv run python main.py
```

6. Run the benchmark:

```bash
VLLM_METRICS_URL=http://<droplet-ip>:8000/metrics \
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
  uv run python bench/run_agent_bench.py
```

## Local development

DeepSeek uses the same MAF Chat Completions client:

```bash
RETAIL_PROVIDER=deepseek \
RETAIL_MODEL=deepseek-chat \
DEEPSEEK_API_KEY=*** \
  uv run python main.py
```

## Demo constraints

- One MAF agent owns the conversation and reuses one `AgentSession`.
- All tools are read-only; the application cannot add items to a cart or purchase them.
- SQLite FTS5 retrieves candidates; the agent classifies product identity before deterministic application-owned ranking.
- `CatalogEvidenceTracker` enforces catalog provenance: the `finalize_recommendations` tool drops any candidate whose `item_id` was not returned by `search_catalog` in the current session.
- Non-exact products and unknown IDs cannot enter the displayed recommendation set.
- The offline catalog has no prices, ratings, reviews, popularity, or live availability.
- Only inference requires network access during the demo.
