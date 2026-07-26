# Deployment

## Active deployment (current droplet)

- **Droplet IP:** `129.212.178.184` (MI300X, Radeon Cloud)
- **Local SSH key fingerprint:** `SHA256:t8bki1FeGBrV5ykHchsQ2D96IHxEevFpxw5j/vJgl+M`
- **Local SSH public key:**
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMH48B5iHqh9P8Eb4EjhXsccN6KLoh31+Q9iL6a4vVcS amd-radeon-droplet-202607
  ```
- **SSH from laptop:** `ssh root@129.212.178.184`
- **Run deploy script:** `ssh root@129.212.178.184 'bash -s' < scripts/deploy_droplet.sh`

The public key is safe to commit (it's designed to be public). The private key lives at `~/.ssh/id_ed25519` with mode 600 and never enters git.

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

The 1-Click image ships vLLM 0.23.0 on ROCm 7.2.4 (Ubuntu 24.04) inside a Docker container. It does **not** come with Gemma 4 31B pre-installed — the model is downloaded from Hugging Face on first launch (~5-10 minutes on the first run, cached afterwards).

1. Create a cloud firewall allowing SSH and port 8000 only from your public IP.
2. Create the MI300X droplet and add your SSH key (see "Active deployment" above for the public key).
3. SSH in and verify the GPU and Docker container:

```bash
amd-smi --showproductname --showuse --showmeminfo vram
docker ps
```

Read the MOTD — it prints the JupyterLab URL/token, the vLLM container name, and the `docker exec` command for an interactive shell.

4. Upload and run the project's deployment script:

```bash
scp scripts/deploy_droplet.sh root@129.212.178.184:/root/
ssh root@129.212.178.184 bash /root/deploy_droplet.sh
```

Or in one shot (no scp):

```bash
ssh root@129.212.178.184 'bash -s' < scripts/deploy_droplet.sh
```

Set `VLLM_MODEL` to download a different model (default is `google/gemma-4-31B-it`):

```bash
VLLM_MODEL=meta-llama/llama-3.3-70b-instruct \
  ssh root@129.212.178.184 'bash -s' < scripts/deploy_droplet.sh
```

The script stops the default vLLM process, then re-launches it inside the container with prefix caching, chunked prefill, fp8 KV cache, speculative decoding, and tool-calling support (`--enable-auto-tool-choice --tool-call-parser hermes`).

On any failure the script writes a diagnostic bundle to `/root/retailconcierge_report.txt` (full vLLM log tail, step timing, exit-code table) and exits with a distinct code (3=download, 4=startup, 5/6/7=smoke). See the script header for the full exit-code table and rerun procedure.

5. Run the catalog and agent on the laptop, pointing inference at the droplet:

```bash
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://129.212.178.184:8000/v1 \
RETAIL_MODEL=google/gemma-4-31B-it \
  uv run python main.py
```

6. Run the benchmark:

```bash
VLLM_METRICS_URL=http://129.212.178.184:8000/metrics \
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://129.212.178.184:8000/v1 \
RETAIL_MODEL=google/gemma-4-31B-it \
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
