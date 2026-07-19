# Deployment

## Build the offline catalog

The source dataset stays outside Git. Import it on the machine running the application:

```bash
uv sync
uv run python scripts/import_catalog.py \
  --products /home/rajasingh/Downloads/archive/amazon_products.csv \
  --categories /home/rajasingh/Downloads/archive/amazon_categories.csv \
  --database retail_catalog.db
```

The importer validates the CSV columns and category foreign keys, builds the database beside the target, runs SQLite integrity checks, and atomically installs the completed catalog. For a quick development catalog, add `--limit 50000`.

## AMD Developer Cloud

Use one MI300X GPU Droplet with the vLLM 1-Click image. The available credit covers GPU usage only; set a hard spending cap before creating the droplet and destroy it after each session.

1. Create a cloud firewall allowing SSH and port 8000 only from your current public IP.
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

5. Run the catalog and agents on the laptop, pointing only inference at the droplet:

```bash
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
RETAIL_DB=./retail_catalog.db \
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

- The catalog is offline and deterministic.
- Product prices, ratings, review counts, and popularity are dataset snapshots.
- The application does not claim live availability, current pricing, shipping, or specifications absent from product titles.
- Only vLLM inference requires network access during the judge demo.
- Every catalog query and every final recommendation is shown to the user for approval before it runs or ships. The benchmark uses an auto-approving middleware so the headless run does not stall.
