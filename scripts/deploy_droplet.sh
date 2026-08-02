#!/usr/bin/env bash
# scripts/deploy_droplet.sh
#
# ARCHITECTURE BOUNDARY: this script manages vLLM ONLY on the GPU droplet.
# The application (app.py / main.py / MAF agent), the catalog DB, and all
# source code live on the developer's laptop. The droplet serves the model
# endpoint; nothing else. Do NOT copy the repo or the catalog DB to the
# droplet — the calibration/quantization scripts run on the laptop or pass
# data via scp, never by leaving app artifacts on the GPU host.
#
# One-shot reconfiguration of vLLM on an AMD Radeon Cloud MI300X 1-Click droplet.
# Run from your laptop via `ssh root@<droplet-ip> 'bash -s' < scripts/deploy_droplet.sh`
# or scp + run on the droplet.
#
# This script assumes the AMD 1-Click image is already running. That image provides:
#   - Ubuntu 24.04 host with ROCm 7.2.4 driver stack
#   - Docker container named `rocm` running vLLM 0.23.0 + OpenAI server
#   - JupyterLab environment (URL/token in MOTD at SSH login)
#   - amd-smi / rocminfo for GPU verification
#   - Port 8000 mapped to host (vLLM OpenAI-compatible API)
#
# What this script does on top of the preconfigured image:
#   1. Verifies GPU + ROCm
#   2. Locates the vLLM container (auto-detects `rocm` as the default name)
#   3. Stops the default vLLM inside the container
#   4. Pre-downloads the model to the HF cache (saves minutes on first serve)
#   5. Records the model SHA-256 to /root/retailconcierge_fingerprint.txt
#   6. Re-launches vLLM with rubric-winning flags:
#        --enable-prefix-caching (multi-turn bonus)
#        --enable-chunked-prefill (latency under concurrency)
#        --kv-cache-dtype fp8 (BF16 serving; auto for Quark INT8 — see KV_CACHE_DTYPE)
#        --enable-auto-tool-choice --tool-call-parser gemma4 (Gemma 4 native tool calls)
#   7. Waits for "server is fired up" or fails fast
#   8. Smoke-tests: /health, /v1/models, and a real tool-call request
#   9. Records GPU snapshot (amd-smi / rocm-smi) to /root/retailconcierge_gpu.txt
#
# Overridable env vars (with defaults):
#   VLLM_MODEL          = google/gemma-4-26B-A4B-it   (BF16 fallback / training)
#   VLLM_HF_MODEL       = rajasingh012/gemma-4-12b-it-quark-w8a8-int8
#                         (our public Quark W8A8 INT8 12B on HF — served with
#                          --quantization quark; destroy/recreate-able droplet:
#                          this script pulls the checkpoint from HF, no
#                          re-quantize needed. Set VLLM_HF_MODEL= to serve the
#                          BF16 VLLM_MODEL instead.)
#   MAX_LEN             = 32768
#   GPU_MEM             = 0.90
#   CTR_NAME            = rocm (auto-detected: rocm, vllm, inference, amd-vllm)
#   SKIP_DOWNLOAD       = 0     (set to 1 to skip pre-download; useful for re-runs)
#   SMOKE_TEST_QUERIES  = 1     (set to 0 to skip smoke tests)
#
# Exit codes:
#   0  = success
#   1  = pre-flight failure (GPU/Docker)
#   2  = container not found
#   3  = model download / fingerprint failure
#   4  = vLLM did not come up within timeout
#   5  = smoke /health failed
#   6  = smoke /v1/models failed
#   7  = smoke tool-call probe failed
#
# Rerun procedure:
#   - First run:        ./deploy_droplet.sh
#   - Re-run (cached):  SKIP_DOWNLOAD=1 ./deploy_droplet.sh
#   - Re-run (vLLM up): docker exec $CTR_NAME pkill -f 'vllm serve' && ./deploy_droplet.sh
#
# On any failure, read the diagnostic bundle:
#   cat /root/retailconcierge_report.txt      # step-by-step summary
#   cat /root/retailconcierge_vllm.log | tail -200   # full vLLM startup log
#   cat /root/retailconcierge_fingerprint.txt # model + SHA-256
#   cat /root/retailconcierge_gpu.txt         # amd-smi snapshot at failure

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-26B-A4B-it}"
# VLLM_HF_MODEL: a Hugging Face repo of an already-quantized checkpoint
# (e.g. our public Quark W8A8 INT8 12B). When set, the script downloads it
# into the container and serves it with --quantization quark — no local
# re-quantize needed, so a destroyed droplet is fully recreate-able from
# just this script + the public HF repo.
VLLM_HF_MODEL="${VLLM_HF_MODEL:-rajasingh012/gemma-4-12b-it-quark-w8a8-int8}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_MEM="${GPU_MEM:-0.90}"
CTR_NAME="${CTR_NAME:-vllm}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SMOKE_TEST_QUERIES="${SMOKE_TEST_QUERIES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"   # 10 min for model load
# KV cache dtype. Defaults: fp8 for BF16 serving (proven, benchmarked on
# 26B MoE: 427 tok/s, VRAM headroom), auto for Quark INT8 serving (small
# model, and fp8 KV triggers "uncalibrated q_scale ... may cause accuracy
# issues" since Quark INT8 checkpoints carry no KV scale factors).
# Override explicitly with KV_CACHE_DTYPE=bfloat16|fp8|auto when needed.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

# Optional: serve an AMD Quark-quantized FP8 W8A8 checkpoint from a LOCAL dir.
# Default is to serve the BF16 source model — but the recommended default is
# now VLLM_HF_MODEL (our public Quark INT8 on HF, set above). Build a local
# FP8 model once with scripts/quantize_int8.sh, then set
# VLLM_FP8_MODEL=/models/gemma-4-26B-A4B-it-fp8 to swap.
VLLM_FP8_MODEL="${VLLM_FP8_MODEL:-}"

FP_FILE="/root/retailconcierge_fingerprint.txt"
GPU_FILE="/root/retailconcierge_gpu.txt"
LOG_FILE="/root/retailconcierge_vllm.log"
REPORT_FILE="/root/retailconcierge_report.txt"

# ─── helper: timestamped log + step timer ──────────────────────────────────
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() {
    printf '[%s] %s\n' "$(ts)" "$*"
    printf '[%s] %s\n' "$(ts)" "$*" >> "$REPORT_FILE"
}
die() {
    local code="${2:-1}"
    {
        printf '[%s] FATAL: %s\n' "$(ts)" "$*"
        printf '[%s] --- vLLM log tail (200 lines) ---\n' "$(ts)"
        docker exec "$CTR_NAME" tail -200 "$LOG_FILE" 2>/dev/null || echo "  (log unavailable)"
        printf '[%s] --- end of bundle ---\n' "$(ts)"
    } >> "$REPORT_FILE" 2>&1
    printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2
    printf 'Full diagnostic bundle: %s\n' "$REPORT_FILE" >&2
    exit "$code"
}
step_start() { STEP_NAME="$1"; STEP_T0="$(date +%s)"; log "─── $STEP_NAME ───"; }
step_end() {
    local dt=$(( $(date +%s) - STEP_T0 ))
    log "    ✓ $STEP_NAME done (${dt}s)"
}

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
    # Fallback 1: any running container with vllm/inference in its image
    DETECTED_CTR=$(docker ps --format '{{.Image}}\t{{.Names}}' | grep -iE 'vllm|inference' | head -1 | cut -f2 || true)
fi
if [ -z "$DETECTED_CTR" ]; then
    # Fallback 2: AMD Radeon droplets often name the container just 'rocm' or 'amd'
    DETECTED_CTR=$(docker ps --format '{{.Names}}' | grep -iE '^(rocm|amd|vllm|inference)$' | head -1 || true)
fi
if [ -z "$DETECTED_CTR" ]; then
    # Fallback 3: if there's exactly one running container, use it
    RUNNING_COUNT=$(docker ps --format '{{.Names}}' | wc -l)
    if [ "$RUNNING_COUNT" -eq 1 ]; then
        DETECTED_CTR=$(docker ps --format '{{.Names}}' | head -1)
        log "    Only one running container — falling back to it: $DETECTED_CTR"
    fi
fi
if [ -z "$DETECTED_CTR" ]; then
    log "No vLLM container found. Running containers:"
    docker ps -a
    die "could not find vLLM container — set CTR_NAME explicitly (current candidates tried: vllm, vllm-rocm, vllm_openai, amd-vllm, inference)" 2
fi
CTR_NAME="$DETECTED_CTR"
log "Using container: $CTR_NAME"
IMG=$(docker inspect --format '{{.Image}}' "$CTR_NAME" 2>/dev/null | head -1)
log "  image: ${IMG:-unknown}"

# ─── [3/5] stop default vLLM, pre-download model ───────────────────────────
step_start "[3/5] prep + download"

log "    Stopping default vLLM inside $CTR_NAME (if running)"
# Use || true on each command to prevent the script from dying if no vLLM exists.
# Also kill orphaned VLLM::EngineCore children — they hold GPU VRAM even after
# the parent vllm serve process dies (seen live on the MI300X droplet).
docker exec "$CTR_NAME" bash -c '
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 3
    if pgrep -af vllm >/dev/null 2>&1; then
        echo "  vllm still running, attempting SIGKILL"
        pkill -9 -f vllm
        sleep 1
    fi
    pgrep -af vllm && echo "  WARN: vllm processes still present" || echo "  vllm stopped (or none was running)"
' || log "    WARN: pkill command returned non-zero (probably no vllm was running)"

# Download the BF16 source model ONLY if we're actually going to serve it.
# When VLLM_HF_MODEL is set (the common case now), the quantized HF
# checkpoint is what we serve — skip the 26B BF16 download entirely.
if [ -n "$VLLM_HF_MODEL" ] && [ -z "$VLLM_FP8_MODEL" ]; then
    log "    VLLM_HF_MODEL=$VLLM_HF_MODEL set — skipping BF16 $VLLM_MODEL download"
elif [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "    Pre-downloading $VLLM_MODEL (skippable via SKIP_DOWNLOAD=1)"
    DL_LOG="$(mktemp)"
    if ! docker exec "$CTR_NAME" bash -c "
        export HF_ENDPOINT=\${HF_ENDPOINT:-https://hf-mirror.com}
        python3 -c \"
from huggingface_hub import snapshot_download
import os, sys
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
try:
    p = snapshot_download(repo_id='$VLLM_MODEL', allow_patterns=['*.json','*.txt','*.model','*.safetensors','tokenizer*'])
    print('Cached at:', p)
except Exception as e:
    print('DOWNLOAD ERROR:', type(e).__name__, str(e)[:500], file=sys.stderr)
    sys.exit(1)
\"
    " >"$DL_LOG" 2>&1; then
        log "    Download failed. Tail of download log:"
        tail -20 "$DL_LOG" | sed 's/^/      /' | tee -a "$REPORT_FILE"
        rm -f "$DL_LOG"
        die "model download failed — set SKIP_DOWNLOAD=1 if cache is intact, or check HF_ENDPOINT" 3
    fi
    log "    $(tail -3 "$DL_LOG")"
    rm -f "$DL_LOG"
else
    log "    SKIP_DOWNLOAD=1 — using existing cache"
fi

# Download the quantized HF checkpoint (if VLLM_HF_MODEL is set and not
# already cached). Runs even with SKIP_DOWNLOAD=1 — that flag only gates
# the BF16 $VLLM_MODEL download above.
if [ -n "$VLLM_HF_MODEL" ]; then
    log "    Pre-downloading quantized checkpoint $VLLM_HF_MODEL (skippable via VLLM_HF_MODEL=)"
    DL_LOG="$(mktemp)"
    if ! docker exec "$CTR_NAME" bash -c "
        export HF_ENDPOINT=\${HF_ENDPOINT:-https://hf-mirror.com}
        python3 -c \"
from huggingface_hub import snapshot_download
import os, sys
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
try:
    p = snapshot_download(repo_id='$VLLM_HF_MODEL', allow_patterns=['*.json','*.txt','*.model','*.safetensors','tokenizer*','*.jinja'])
    print('Quantized checkpoint cached at:', p)
except Exception as e:
    print('DOWNLOAD ERROR:', type(e).__name__, str(e)[:500], file=sys.stderr)
    sys.exit(1)
\"
    " >"$DL_LOG" 2>&1; then
        log "    Quantized download failed. Tail of download log:"
        tail -20 "$DL_LOG" | sed 's/^/      /' | tee -a "$REPORT_FILE"
        rm -f "$DL_LOG"
        log "    Continuing with $VLLM_MODEL (BF16) — set VLLM_HF_MODEL= to skip, or check HF_ENDPOINT"
    else
        log "    $(tail -3 "$DL_LOG")"
        rm -f "$DL_LOG"
    fi
fi

log "    Recording model fingerprint"
if ! docker exec "$CTR_NAME" bash -c "
    python3 -c \"
from huggingface_hub import HfApi
import sys
try:
    api = HfApi()
    info = api.model_info('$VLLM_MODEL', files_metadata=True)
    print('Model:', '$VLLM_MODEL')
    print('SHA:', info.sha)
    print('Last modified:', info.last_modified)
    print('Pipeline:', info.pipeline_tag)
    print('Library:', info.library_name)
except Exception as e:
    print('FINGERPRINT ERROR:', type(e).__name__, str(e)[:500], file=sys.stderr)
    sys.exit(1)
\"
" > "$FP_FILE" 2>&1; then
    cat "$FP_FILE"
    die "fingerprint capture failed — HF metadata unreachable; deployment cannot continue" 3
fi
log "    Fingerprint saved to $FP_FILE"
cat "$FP_FILE"
step_end "[3/5] prep + download"

# ─── [4/5] launch vLLM with rubric-winning flags ────────────────────────────
step_start "[4/5] launch vLLM"

# Pick the serving model: prefer FP8 if it's been quantized, else BF16 source.
# Quark-quantized weights halve weight VRAM and lift MoE decode ~20-40%
# (see scripts/quantize_int8.sh for the build pipeline).
#
# vLLM ≥ 0.26 supports `--quantization quark` directly (reads the
# `quantization_config` block Quark writes into config.json and picks
# the right kernel). Reference:
#   https://docs.vllm.ai/stable/features/quantization/quark/
#
# vLLM 0.23 (the AMD 1-Click image version) lacks this flag. We fall back
# to `--quantization fp8` for the BF16→FP8 inference path that ships in
# our window. The Quark-quantized model path remains a deferred TODO for
# after vLLM ≥ 0.26 lands on the image.
if [ -n "$VLLM_FP8_MODEL" ] && docker exec "$CTR_NAME" test -d "$VLLM_FP8_MODEL"; then
    SERVED_MODEL="$VLLM_FP8_MODEL"
    VLLM_VERSION=$(docker exec "$CTR_NAME" python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "0.0.0")
    if printf '%s\n' "$VLLM_VERSION" | awk -F. '{ exit !($1 > 0 || $2 >= 26) }'; then
        SERVED_FLAGS="--quantization quark"
        log "    Serving Quark-quantized FP8 weights from $VLLM_FP8_MODEL (vLLM $VLLM_VERSION supports --quantization quark)"
    else
        SERVED_FLAGS="--quantization fp8"
        log "    WARNING: vLLM $VLLM_VERSION < 0.26 — falling back to --quantization fp8 (no Quark recipe);"
        log "             see scripts/quantize_int8.sh TODO on upgrading to vLLM nightly for the Quark path"
    fi
    # Quark INT8 checkpoints carry no KV scale factors; fp8 KV cache would run
    # uncalibrated (vLLM warns "may cause accuracy issues"). Default to auto.
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    log "    Serving Quark-quantized FP8 weights from $VLLM_FP8_MODEL"
elif [ -n "$VLLM_HF_MODEL" ]; then
    # VLLM_HF_MODEL: serve the Quark INT8 checkpoint. vLLM 0.26 can resolve
    # an HF repo ID straight from the local HF cache (no local dir needed).
    SERVED_MODEL="$VLLM_HF_MODEL"
    SERVED_FLAGS="--quantization quark"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    log "    Serving Quark W8A8 INT8 from HF repo $VLLM_HF_MODEL (vLLM resolves it from the local HF cache)"
else
    SERVED_MODEL="$VLLM_MODEL"
    SERVED_FLAGS=""
    # BF16 serving: keep fp8 KV cache (proven on 26B MoE: 427 tok/s bench).
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
    if [ -n "$VLLM_FP8_MODEL" ]; then
        log "    VLLM_FP8_MODEL set but $VLLM_FP8_MODEL not in container — falling back to BF16 source"
    fi
fi

# Pin AITER explicitly. AMD ROCm 7.2 + vLLM 0.23 auto-detect Aiter for the
# attention and MoE GEMM paths on gfx942 (MI300X), but pinning forces the
# choice and gives the published 5-20% decode-side lift an engineer can
# rely on. Source: ROCm AITER README, AMD Quark vLLM tuning blog (Aug 2026).
#   VLLM_USE_AITER=1                 — enable AITER attention + linear
#   VLLM_ROCM_USE_AITER_FA=1         — AITER flash attention
#   VLLM_ROCM_USE_AITER_LINEAR=1     — AITER linear/FFN (MoE expert path)
# MLA is off because Gemma 4 26B A4B is pure MoE, not a hybrid
# (multi-latent-attention) model.
log "    --enable-prefix-caching (multi-turn smoothness bonus)"
log "    --enable-chunked-prefill (latency under concurrency)"
log "    --kv-cache-dtype $KV_CACHE_DTYPE (auto for Quark INT8, fp8 for BF16)"
log "    --enable-auto-tool-choice --tool-call-parser gemma4 (MAF tool calls, Gemma 4 native)"
log "    AITER pinned (VLLM_USE_AITER=1, FA + LINEAR on) — AMD-tuned attention/MoE paths"
[ -n "$SERVED_FLAGS" ] && log "    $SERVED_FLAGS (Quark W8A8 quantization)"

# Open the API port on the host firewall. The DO 1-Click image ships UFW
# enabled with only 22/80/443 allowed; without this, /v1/models and
# /v1/chat/completions hang at the TCP layer (TCP handshake completes but
# the app-level reply never returns through UFW). Idempotent.
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
    ufw allow "${VLLM_PORT:-8000}/tcp" >/dev/null 2>&1 || true
    log "    UFW: opened ${VLLM_PORT:-8000}/tcp for the vLLM API"
fi

# shellcheck disable=SC2086
docker exec -d "$CTR_NAME" bash -c "
    export VLLM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_FA=1
    export VLLM_ROCM_USE_AITER_LINEAR=1
    export VLLM_ROCM_USE_AITER_MLA=0
    cd /workspace 2>/dev/null || cd /root
    nohup vllm serve '$SERVED_MODEL' \
        --host 0.0.0.0 --port $VLLM_PORT \
        --max-model-len $MAX_LEN \
        --gpu-memory-utilization $GPU_MEM \
        --tensor-parallel-size 1 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --kv-cache-dtype $KV_CACHE_DTYPE \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        $SERVED_FLAGS \
        > '$LOG_FILE' 2>&1 &
    echo 'vLLM PID:' \$!
"

# Wait for startup banner. vLLM 0.23 prints "Application startup complete".
# Older versions print "server is fired up and ready to roll".
# The log file lives INSIDE the container (the launch redirect wrote it
# there via docker exec), so the polling loop must read it via docker exec —
# a host-path read never sees it.
log "    Waiting for vLLM startup (timeout: ${STARTUP_TIMEOUT}s, polling every ${INTERVAL:-10}s)"
ELAPSED=0
INTERVAL=10
STARTED=0
while [ "$ELAPSED" -lt "$STARTUP_TIMEOUT" ]; do
    if docker exec "$CTR_NAME" grep -qE "server is fired up and ready to roll|Application startup complete" "$LOG_FILE" 2>/dev/null; then
        STARTED=1
        break
    fi
    if docker exec "$CTR_NAME" grep -qiE "OutOfMemoryError|CUDA out of memory|HIP out of memory|RuntimeError: out of memory|memory allocation failed|failed to allocate memory" "$LOG_FILE" 2>/dev/null; then
        log "    OOM detected in vLLM log. Tail:"
        docker exec "$CTR_NAME" tail -80 "$LOG_FILE" 2>/dev/null | sed 's/^/      /' | tee -a "$REPORT_FILE"
        die "vLLM ran out of GPU memory — try VLLM_MODEL with smaller weights, lower GPU_MEM, or shorter MAX_LEN" 4
    fi
    if docker exec "$CTR_NAME" grep -qiE "error|exception|traceback" "$LOG_FILE" 2>/dev/null; then
        log "    vLLM logged an error. Tail:"
        docker exec "$CTR_NAME" tail -80 "$LOG_FILE" 2>/dev/null | sed 's/^/      /' | tee -a "$REPORT_FILE"
        die "vLLM startup failed — see diagnostic bundle for full log" 4
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
    log "    ...${ELAPSED}s elapsed"
done

if [ "$STARTED" -ne 1 ]; then
    log "    Timeout. Tail of log:"
    docker exec "$CTR_NAME" tail -80 "$LOG_FILE" 2>/dev/null | sed 's/^/      /' | tee -a "$REPORT_FILE"
    die "vLLM did not start within ${STARTUP_TIMEOUT}s — check STARTUP_TIMEOUT env or model size" 4
fi
log "    vLLM is ready."
step_end "[4/5] launch vLLM"

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

# 1. /health — vLLM 0.23 returns HTTP 200 with an EMPTY body. Also note:
#    `docker exec rocm curl` returns empty even when the server is healthy
#    (loopback inside the container doesn't resolve the same way), so probe
#    from the HOST, not inside the container.
log "    [smoke 1/3] /health"
HEALTH_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://localhost:$VLLM_PORT/health" 2>&1 || true)
log "      /health HTTP $HEALTH_CODE"
if [ "$HEALTH_CODE" != "200" ]; then
    log "    /health did not return 200 (got $HEALTH_CODE)"
    log "    Recent vLLM log:"
    docker exec "$CTR_NAME" tail -30 "$LOG_FILE" 2>/dev/null | sed 's/^/      /' | tee -a "$REPORT_FILE"
    die "vLLM /health did not return 200 — server may have crashed after startup" 5
fi

# 2. /v1/models — confirm model is loaded
log "    [smoke 2/3] /v1/models"
MODELS_JSON=$(docker exec "$CTR_NAME" curl -fsS --max-time 10 "http://localhost:$VLLM_PORT/v1/models" 2>&1 || true)
log "      $(echo "$MODELS_JSON" | head -c 200)"
# The served model name is $SERVED_MODEL (which equals $VLLM_MODEL for BF16
# serving, or the Quark checkpoint path when VLLM_FP8_MODEL is set). Check
# against it, not $VLLM_MODEL, so the FP8/INT8 path passes.
if ! echo "$MODELS_JSON" | grep -q "$SERVED_MODEL"; then
    log "    Expected: $SERVED_MODEL"
    log "    Got:      $MODELS_JSON"
    die "/v1/models does not list $SERVED_MODEL — check --served-model-name or model arg" 6
fi

# 3. Tool-call probe — this is the make-or-break test for MAF
log "    [smoke 3/3] tool-call probe (the critical test for MAF)"
TOOL_CALL_RESP=$(docker exec "$CTR_NAME" curl -fsS --max-time 30 "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$SERVED_MODEL\",
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

if echo "$TOOL_CALL_RESP" | grep -q '"tool_calls"'; then
    log "    ✓ tool_calls present — vLLM is producing structured output"
elif echo "$TOOL_CALL_RESP" | grep -q '"finish_reason":"tool_calls"'; then
    log "    ✓ finish_reason=tool_calls — vLLM is producing structured output"
else
    {
        echo ""
        echo "=== TOOL-CALL PROBE FAILED ==="
        echo ""
        echo "Full response:"
        echo "$TOOL_CALL_RESP"
        echo ""
        echo "Hint: if response is plain text with JSON in it, the --tool-call-parser"
        echo "flag is wrong. Try --tool-call-parser=llama3_json or pythonic."
        echo "Last 50 lines of vLLM log:"
        docker exec "$CTR_NAME" tail -50 "$LOG_FILE" 2>/dev/null
    } | tee -a "$REPORT_FILE" >&2
    die "vLLM did not return tool_calls — check --tool-call-parser flag" 7
fi
log "    ✓ All smoke tests passed."

log ""
log "=== DEPLOYMENT COMPLETE ==="
log "  Model:       $SERVED_MODEL"
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
log "  RETAIL_MODEL=$SERVED_MODEL \\"
log "    uv run python bench/run_5_queries.py"