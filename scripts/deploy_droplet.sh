#!/usr/bin/env bash
# scripts/deploy_droplet.sh
#
# One-shot deployment for RetailConcierge on an AMD Radeon Cloud MI300X droplet.
# Run from your laptop via `ssh root@<droplet-ip> 'bash -s' < scripts/deploy_droplet.sh`
# or scp + run on the droplet.
#
# What this does:
#   1. Verifies GPU + ROCm
#   2. Locates the vLLM container (assumes AMD 1-Click Docker image)
#   3. Stops the default vLLM inside the container
#   4. Pre-downloads the model to the HF cache (saves minutes on first serve)
#   5. Records the model SHA-256 to /root/retailconcierge_fingerprint.txt
#   6. Re-launches vLLM with rubric-winning flags:
#        --enable-prefix-caching (multi-turn bonus)
#        --enable-chunked-prefill (latency)
#        --kv-cache-dtype fp8 (VRAM headroom)
#        --speculative-config ngram (inter-token latency)
#        --enable-auto-tool-choice --tool-call-parser hermes (MAF tool calls)
#   7. Waits for "server is fired up" or fails fast
#   8. Smoke-tests: /health, /v1/models, and a real tool-call request
#   9. Records GPU snapshot (amd-smi / rocm-smi) to /root/retailconcierge_gpu.txt
#
# Overridable env vars (with defaults):
#   VLLM_MODEL          = google/gemma-4-31B-it
#   MAX_LEN             = 12288
#   GPU_MEM             = 0.90
#   CTR_NAME            = vllm  (auto-detected if that name isn't found)
#   SKIP_DOWNLOAD       = 0     (set to 1 to skip pre-download; useful for re-runs)
#   SMOKE_TEST_QUERIES  = 1     (set to 0 to skip smoke tests)
#
# Exit codes:
#   0  = success
#   1  = pre-flight failure (GPU/Docker)
#   2  = container not found
#   3  = vLLM did not come up within timeout
#   4  = smoke test failed (vLLM up but tool calls broken)

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
MAX_LEN="${MAX_LEN:-12288}"
GPU_MEM="${GPU_MEM:-0.90}"
CTR_NAME="${CTR_NAME:-vllm}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SMOKE_TEST_QUERIES="${SMOKE_TEST_QUERIES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"   # 10 min for model load

FP_FILE="/root/retailconcierge_fingerprint.txt"
GPU_FILE="/root/retailconcierge_gpu.txt"
LOG_FILE="/root/retailconcierge_vllm.log"

# ─── helper: timestamped log ────────────────────────────────────────────────
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

# ─── [1/5] pre-flight ───────────────────────────────────────────────────────
log "[1/5] Pre-flight: GPU + Docker"

if command -v amd-smi >/dev/null 2>&1; then
    GPU_CMD="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_CMD="rocm-smi"
else
    die "amd-smi / rocm-smi not found — is ROCm installed?"
fi
log "GPU tool: $GPU_CMD"
$GPU_CMD --showproductname 2>/dev/null | head -5 || true
$GPU_CMD --showmeminfo vram 2>/dev/null | head -8 || true

command -v docker >/dev/null 2>&1 || die "docker not found on host"
docker version --format '{{.Server.Version}}' >/dev/null || die "docker daemon unreachable"

# ─── [2/5] locate the vLLM container ────────────────────────────────────────
log "[2/5] Locating vLLM container"

# Try to auto-detect by name first, then by image.
DETECTED_CTR=""
for candidate in "$CTR_NAME" vllm-rocm vllm_openai amd-vllm inference; do
    if docker ps --format '{{.Names}}' | grep -qx "$candidate"; then
        DETECTED_CTR="$candidate"
        break
    fi
done
if [ -z "$DETECTED_CTR" ]; then
    # Fallback: any running container with vLLM in its image
    DETECTED_CTR=$(docker ps --format '{{.Image}}\t{{.Names}}' | grep -iE 'vllm' | head -1 | cut -f2 || true)
fi
if [ -z "$DETECTED_CTR" ]; then
    log "No vLLM container found. Running containers:"
    docker ps -a
    die "could not find vLLM container — set CTR_NAME or launch the 1-Click image first" 2
fi
CTR_NAME="$DETECTED_CTR"
log "Using container: $CTR_NAME"
docker inspect --format '{{.Image}}' "$CTR_NAME" | head -1 | xargs -I{} log "  image: {}"

# ─── [3/5] stop default vLLM, pre-download model ───────────────────────────
log "[3/5] Stopping default vLLM inside $CTR_NAME"
docker exec "$CTR_NAME" bash -c "pkill -f 'vllm serve' 2>/dev/null || true; pkill -f 'vllm.entrypoints' 2>/dev/null || true; sleep 3; pgrep -af vllm || echo '  vllm stopped'"

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "    Pre-downloading $VLLM_MODEL to HF cache (skippable via SKIP_DOWNLOAD=1)"
    docker exec "$CTR_NAME" bash -c "
        export HF_ENDPOINT=\${HF_ENDPOINT:-https://hf-mirror.com}
        python3 -c \"
from huggingface_hub import snapshot_download
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
p = snapshot_download(repo_id='$VLLM_MODEL', allow_patterns=['*.json','*.txt','*.model','*.safetensors','tokenizer*'])
print('Cached at:', p)
\" 2>&1 | tail -5
    "
else
    log "    SKIP_DOWNLOAD=1 — using existing cache"
fi

# Record model fingerprint for the spec PDF
log "    Recording model fingerprint"
docker exec "$CTR_NAME" bash -c "
    python3 -c \"
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('$VLLM_MODEL', files_metadata=True)
print('Model:', '$VLLM_MODEL')
print('SHA:', info.sha)
print('Last modified:', info.last_modified)
print('Pipeline:', info.pipeline_tag)
print('Library:', info.library_name)
\"
" > "$FP_FILE" 2>&1
log "    Fingerprint saved to $FP_FILE"
cat "$FP_FILE"

# ─── [4/5] launch vLLM with rubric-winning flags ────────────────────────────
log "[4/5] Launching vLLM inside $CTR_NAME with rubric-winning flags"
log "    --enable-prefix-caching (multi-turn smoothness bonus)"
log "    --enable-chunked-prefill (latency under concurrency)"
log "    --kv-cache-dtype fp8 (VRAM headroom on MI300X)"
log "    --speculative-config ngram (free inter-token-latency win)"
log "    --enable-auto-tool-choice --tool-call-parser hermes (MAF tool calls)"

# shellcheck disable=SC2086
docker exec -d "$CTR_NAME" bash -c "
    cd /workspace 2>/dev/null || cd /root
    nohup vllm serve '$VLLM_MODEL' \
        --host 0.0.0.0 --port $VLLM_PORT \
        --max-model-len $MAX_LEN \
        --gpu-memory-utilization $GPU_MEM \
        --tensor-parallel-size 1 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --kv-cache-dtype fp8 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":5}' \
        > '$LOG_FILE' 2>&1 &
    echo 'vLLM PID:' \$!
"

# Wait for startup banner. Pattern matches vLLM 0.23+ success message.
log "    Waiting for vLLM startup (timeout: ${STARTUP_TIMEOUT}s)"
ELAPSED=0
INTERVAL=10
STARTED=0
while [ "$ELAPSED" -lt "$STARTUP_TIMEOUT" ]; do
    if docker exec "$CTR_NAME" grep -q "server is fired up and ready to roll" "$LOG_FILE" 2>/dev/null; then
        STARTED=1
        break
    fi
    if docker exec "$CTR_NAME" grep -qiE "error|exception|traceback" "$LOG_FILE" 2>/dev/null; then
        log "    vLLM logged an error. Tail:"
        docker exec "$CTR_NAME" tail -40 "$LOG_FILE" || true
        die "vLLM startup failed — check $LOG_FILE inside $CTR_NAME" 3
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
    log "    ...${ELAPSED}s elapsed"
done

if [ "$STARTED" -ne 1 ]; then
    log "    Timeout. Tail of log:"
    docker exec "$CTR_NAME" tail -50 "$LOG_FILE" || true
    die "vLLM did not start within ${STARTUP_TIMEOUT}s" 3
fi
log "    vLLM is ready."

# ─── [5/5] smoke tests + GPU snapshot ───────────────────────────────────────
log "[5/5] Smoke tests + GPU snapshot"

# GPU snapshot — capture to file, prints go to GPU_FILE
log "    Recording GPU snapshot to $GPU_FILE"
{
    echo "=== GPU snapshot at $(ts) ==="
    echo
    echo "--- amd-smi / rocm-smi ---"
    $GPU_CMD 2>&1 | head -30 || true
    echo
    echo "--- vLLM process memory (via container exec) ---"
    docker exec "$CTR_NAME" bash -c "
        python3 -c \"
import os, subprocess
# rocmlite.SMI or just rocm-smi
out = subprocess.run(['rocm-smi','--showmemuse'], capture_output=True, text=True).stdout
print(out)
\"
    " 2>&1 | head -15 || true
} > "$GPU_FILE" 2>&1
log "    Saved."

if [ "$SMOKE_TEST_QUERIES" != "1" ]; then
    log "    SMOKE_TEST_QUERIES=0 — skipping tool-call probe"
    log "Done. vLLM is serving on port $VLLM_PORT. Verify with:"
    log "  curl http://localhost:$VLLM_PORT/v1/models"
    exit 0
fi

# 1. /health
log "    [smoke] /health"
HEALTH=$(docker exec "$CTR_NAME" curl -fsS "http://localhost:$VLLM_PORT/health" 2>&1 || true)
echo "$HEALTH" | head -3
echo "$HEALTH" | grep -q '"status":"ok"' || die "vLLM /health did not return ok" 4

# 2. /v1/models — confirm model is loaded
log "    [smoke] /v1/models"
MODELS_JSON=$(docker exec "$CTR_NAME" curl -fsS "http://localhost:$VLLM_PORT/v1/models" 2>&1 || true)
echo "$MODELS_JSON" | head -3
if ! echo "$MODELS_JSON" | grep -q "$VLLM_MODEL"; then
    die "/v1/models does not list $VLLM_MODEL" 4
fi

# 3. Tool-call probe — this is the make-or-break test for MAF
log "    [smoke] tool-call probe (the critical test)"
TOOL_CALL_RESP=$(docker exec "$CTR_NAME" curl -fsS "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$VLLM_MODEL\",
        \"messages\": [
            {\"role\":\"user\",\"content\":\"List two coffee makers from the catalog.\"}
        ],
        \"tools\": [{
            \"type\": \"function\",
            \"function\": {
                \"name\": \"search_catalog\",
                \"description\": \"Search the product catalog.\",
                \"parameters\": {
                    \"type\": \"object\",
                    \"properties\": {\"query\": {\"type\": \"string\"}},
                    \"required\": [\"query\"]
                }
            }
        }]
    }" 2>&1 || true)
echo "$TOOL_CALL_RESP" | head -3

if echo "$TOOL_CALL_RESP" | grep -q '"tool_calls"'; then
    log "    ✓ tool_calls present — vLLM is producing structured output"
elif echo "$TOOL_CALL_RESP" | grep -q '"finish_reason":"tool_calls"'; then
    log "    ✓ finish_reason=tool_calls — vLLM is producing structured output"
else
    log "    Tool-call response (last 500 chars):"
    echo "$TOOL_CALL_RESP" | tail -c 500
    die "vLLM did not return tool_calls — check --tool-call-parser flag" 4
fi

log ""
log "=== DEPLOYMENT COMPLETE ==="
log "  Model:       $VLLM_MODEL"
log "  Endpoint:    http://localhost:$VLLM_PORT/v1"
log "  vLLM log:    $LOG_FILE  (inside container $CTR_NAME)"
log "  Fingerprint: $FP_FILE"
log "  GPU info:    $GPU_FILE"
log ""
log "From your laptop, verify with:"
log "  curl http://<droplet-ip>:$VLLM_PORT/v1/models"
log ""
log "Then run the agent bench:"
log "  RETAIL_PROVIDER=vllm \\"
log "  RETAIL_BASE_URL=http://<droplet-ip>:$VLLM_PORT/v1 \\"
log "  RETAIL_MODEL=$VLLM_MODEL \\"
log "    uv run python bench/run_5_queries.py"