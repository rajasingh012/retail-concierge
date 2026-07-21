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

The importer reads gzipped NDJSON shards (16 files, ~147K listings), builds
text-values and dimension tables alongside the main `listings` table, creates an
FTS5 index, and runs integrity checks. Add `--limit 50000` for a quick subset.

## AMD Developer Cloud

Use one MI300X GPU Droplet with the vLLM 1-Click image. The available credit
covers GPU usage only; set a hard spending cap before creating the droplet and
destroy it after each session.

1. Create a cloud firewall allowing SSH and port 8000 only from your current
   public IP.
2. Create the MI300X droplet and add your SSH key.
3. SSH in and verify the GPU and vLLM container:

```bash
amd-smi --showproductname --showuse --showmeminfo vram
docker ps
```

4. Start vLLM with the project launcher:

```bash
./scripts/serve-vllm-rocm.sh
```

5. Run the catalog and agents on the laptop, pointing only inference at the
   droplet:

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

DeepSeek uses the same MAF client shape and avoids GPU credit usage:

```bash
RETAIL_PROVIDER=deepseek \
RETAIL_MODEL=deepseek-chat \
DEEPSEEK_API_KEY=*** \
  uv run python main.py
```

## Demo constraints

- The catalog is offline, deterministic, and contains no prices, ratings,
  review counts, or popularity signals.
- The application does not claim live availability, current pricing, shipping,
  or specifications absent from the catalog.
- The application does not add items to a cart or make purchases.
- Only vLLM inference requires network access during the judge demo.
- Discovery asks only when ambiguity blocks a valid search; otherwise it shows
  products with explicit assumptions and contextual refinement chips. Selecting
  a chip reruns the read-only recommendation flow.
- SQLite FTS5 retrieves candidates; Research excludes accessories and uncertain
  product types before deterministic ranking. No vector database or Qdrant
  service is required.
