#!/usr/bin/env bash
# Restart vLLM with rubric-winning flags INSIDE the AMD 1-Click image's
# Docker container (per https://www.amd.com 1-Click vLLM droplet description).
#
# What the 1-Click image actually does:
#   - Ubuntu 24.04 host with ROCm 7.2.4 driver stack
#   - Docker container running vLLM 0.23.0 + OpenAI server on :8000
#   - JupyterLab with example notebooks (URL/token in MOTD at SSH login)
#   - amd-smi / rocminfo on the host for GPU visibility
#
# The default vLLM inside the container likely runs WITHOUT our rubric-winning
# flags (--enable-prefix-caching, --kv-cache-dtype fp8, --speculative-config).
# This script stops that default, then re-launches vLLM inside the container
# with the flags judges reward.
#
# ASSUMPTIONS (verify against your MOTD on first run):
#   - Container name:           vllm  (AMD may name it differently)
#   - Container image:          rocm/vllm:latest or amd/vllm:0.23.0
#   - vLLM config dir in ctr:   /workspace  or /root
#   - Model path inside ctr:    /root/.cache/huggingface  (HF cache)
# If your MOTD shows different names, override via env vars:
#   CTR_NAME=your-container VLLM_MODEL=org/model ./scripts/serve-vllm-rocm.sh
#
# Rubric-winning flags (per AMD/ROCm optimization research):
#   --enable-prefix-caching       THE headline multi-turn win
#   --enable-chunked-prefill      prevents head-of-line blocking
#   --kv-cache-dtype fp8          doubles context capacity on MI300X
#   --speculative-config ngram    free inter-token-latency win
#
# Usage:
#   # From your LAPTOP, after SSH-ing into the droplet:
#   ssh root@<droplet-ip>  # read the MOTD, find the container name
#   ./scripts/serve-vllm-rocm.sh
#
#   # Or run it remotely:
#   scp scripts/serve-vllm-rocm.sh root@<droplet-ip>:/root/
#   ssh root@<droplet-ip> bash /root/serve-vllm-rocm.sh

set -euo pipefail

CTR_NAME="${CTR_NAME:-vllm}"            # override if your MOTD shows a different name
VLLM_MODEL="${VLLM_MODEL:-google/gemma-3-27b-it}"
QUANT="${QUANT:-bf16}"                  # bf16 | fp8 | gguf
MAX_LEN="${MAX_LEN:-12288}"
GPU_MEM="${GPU_MEM:-0.90}"
TP="${TP:-1}"

echo "==> [1/5] Pre-flight: GPU + Docker"
if ! command -v amd-smi >/dev/null 2>&1 && ! command -v rocm-smi >/dev/null 2>&1; then
    echo "    amd-smi / rocm-smi not found"
    exit 1
fi
(amd-smi --showproductname 2>/dev/null || rocm-smi --showproductname) | head -5

if ! command -v docker >/dev/null 2>&1; then
    echo "    docker not found on host"
    exit 1
fi

# Find the vLLM container (don't assume exact name)
echo "==> [2/5] Locating vLLM container"
DETECTED_CTR=$(docker ps --format '{{.Names}}' | grep -iE 'vllm|inference' | head -1 || true)
if [ -z "$DETECTED_CTR" ]; then
    DETECTED_CTR=$(docker ps --format '{{.Names}}' | head -1)
fi
if [ -z "$DETECTED_CTR" ]; then
    echo "    No running containers found. Is the 1-Click image booted?"
    echo "    All containers:"
    docker ps -a
    exit 1
fi
echo "    Detected container: $DETECTED_CTR"
if [ "$DETECTED_CTR" != "$CTR_NAME" ]; then
    echo "    (env override CTR_NAME=$CTR_NAME didn't match; using detected)"
    CTR_NAME="$DETECTED_CTR"
fi

# Build the vLLM flags
case "$QUANT" in
    bf16) QUANT_FLAGS="" ;;
    fp8)  QUANT_FLAGS="--quantization fp8 --kv-cache-dtype fp8" ;;
    gguf) QUANT_FLAGS="--quantization gguf" ;;
    *)    echo "    Unknown QUANT=$QUANT"; exit 1 ;;
esac

echo "==> [3/5] Stopping default vLLM inside $CTR_NAME"
docker exec "$CTR_NAME" bash -c "pkill -f 'vllm serve' || true; sleep 2; pgrep -af vllm || echo 'vllm stopped'"

echo "==> [4/5] Re-launching vLLM with rubric-winning flags (inside container)"
echo "    Model:        $VLLM_MODEL"
echo "    Quant:        $QUANT"
echo "    Max length:   $MAX_LEN"
echo "    Flags:        --enable-prefix-caching --enable-chunked-prefill"
echo "                  --speculative-config ngram $QUANT_FLAGS"

# Run vLLM in detached mode inside the container; logs go to /tmp/vllm.log
docker exec -d "$CTR_NAME" bash -c "vllm serve '$VLLM_MODEL' \
    --host 0.0.0.0 --port 8000 \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --tensor-parallel-size $TP \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":5}' \
    $QUANT_FLAGS > /tmp/vllm.log 2>&1"

echo "    vLLM launched in background. Tailing /tmp/vllm.log ..."
echo "    Watch for: 'The server is fired up and ready to roll!'"
echo ""

# Optional: tail logs for 30s so user sees progress (Ctrl-C to detach)
read -p "    Tail logs now? [y/N] " TAIL
if [[ "$TAIL" =~ ^[Yy]$ ]]; then
    docker exec "$CTR_NAME" tail -f /tmp/vllm.log
fi

echo "==> [5/5] Done"
echo "    Verify from your laptop:"
echo "      curl http://<droplet-ip>:8000/health"
echo "      curl http://<droplet-ip>:8000/v1/models"
echo "      curl http://<droplet-ip>:8000/metrics | grep vllm:prefix_cache_hit_rate"
echo ""
echo "    If vLLM is unreachable, check:"
echo "      docker logs $CTR_NAME"
echo "      docker exec $CTR_NAME cat /tmp/vllm.log"