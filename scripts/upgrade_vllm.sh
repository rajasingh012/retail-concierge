#!/usr/bin/env bash
# scripts/upgrade_vllm.sh
#
# ARCHITECTURE BOUNDARY: this script runs ON the GPU droplet (host shell).
# It upgrades vLLM inside the `rocm` container from the AMD 1-Click image's
# bundled 0.23.0 to the latest AMD-compatible release from the upstream
# wheel index. The app + catalog DB stay on the developer's laptop.
#
# WHY UPGRADE: vLLM 0.23's Quark-MoE loader has a prefix bug
# (KeyError: 'layers.0.router.proj.weight_scale') that prevents serving
# Quark-quantized MoE checkpoints. vLLM 0.26 fixes it. The AMD fork
# (ROCm/vllm) is officially deprecated (2025-09-09); AMD points to the
# upstream vLLM ROCm wheel index. See DEPLOYMENT_JOURNAL.md Issue 12.
#
# Research (verified 2026-08-01):
#   - Upstream ROCm wheel index: https://wheels.vllm.ai/rocm/0.26.0/rocm723/
#   - vllm-0.26.0+rocm723-cp312-cp312-manylinux_2_34_x86_64.whl
#   - Built for ROCm 7.2.3 + Python 3.12 (the 1-Click image ships both)
#   - Install with `uv pip` per AMD's pip doc warning (uv refuses system
#     installs without --system)
#
# ABI fixes discovered live (torch 2.11 mismatches):
#   - flash-attn 2.8.3 .so links c10::hip::getCurrentHIPStream with a
#     signature mismatch -> undefined symbol at import. UNINSTALL it;
#     vLLM falls back to Triton/AITER attention.
#   - torchaudio 2.9.0 links against a different libc10 -> undefined
#     symbol _ZN3c1013MessageLoggerC1EPKciib. UNINSTALL it (not needed
#     for server inference).
#   - torch_c_dlpack_ext 0.1.5 same class of mismatch. UNINSTALL it.
#
# Usage (run on the droplet, not the laptop):
#   ssh root@<droplet-ip> 'bash -s' < scripts/upgrade_vllm.sh
#   or: scp scripts/upgrade_vllm.sh root@<droplet-ip>:/root/ && \
#       ssh root@<droplet-ip> 'bash /root/upgrade_vllm.sh'
#
# Env overrides:
#   VLLM_VERSION = 0.26.0+rocm723   (default; adjust for newer releases)
#   CTR_NAME     = rocm

set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.26.0+rocm723}"
CTR_NAME="${CTR_NAME:-rocm}"
WHEELS_URL="https://wheels.vllm.ai/rocm/0.26.0/rocm723"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

# ─── [1/4] preflight: container + current version ────────────────────────────
log "[1/4] Preflight"
docker ps --format '{{.Names}}' | grep -qx "$CTR_NAME" || die "container '$CTR_NAME' not running"
log "Current vLLM: $(docker exec "$CTR_NAME" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"
log "Python: $(docker exec "$CTR_NAME" python3 --version)"
docker exec "$CTR_NAME" python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "need Python 3.12 for the upstream wheel"; print("Python 3.12 OK")' \
    || die "wheel requires Python 3.12 (image may have moved to 3.14)"

# ─── [2/4] stop vLLM + remove ABI-broken packages ────────────────────────────
log "[2/4] Stop vLLM, remove ABI-broken packages (flash-attn, torchaudio, torch_c_dlpack_ext)"
docker exec "$CTR_NAME" bash -c 'ps -ef | grep -E "vllm serve|EngineCore" | grep -v grep | awk "{print \$2}" | xargs -r kill -9; sleep 2' || true
docker exec "$CTR_NAME" pip uninstall -y flash-attn torchaudio torch_c_dlpack_ext 2>&1 | grep -E "Uninstalling|not installed" | head -5 || true

# ─── [3/4] install vLLM from upstream ROCm wheel index ───────────────────────
log "[3/4] Install vllm==${VLLM_VERSION} from ${WHEELS_URL}"
docker exec "$CTR_NAME" bash -lc "uv pip install --system --no-cache-dir \"vllm==${VLLM_VERSION}\" --extra-index-url ${WHEELS_URL}" 2>&1 | tail -8

# Re-install amd-aiter from the same tree (ABI-matched wheel).
log "    amd-aiter from ${WHEELS_URL}"
docker exec "$CTR_NAME" bash -lc "uv pip install --system --no-cache-dir amd-aiter --extra-index-url ${WHEELS_URL}" 2>&1 | tail -4

# ─── [4/4] verify imports ────────────────────────────────────────────────────
log "[4/4] Verify imports"
docker exec "$CTR_NAME" bash -lc "
python3 -c \"import vllm; print('vllm', vllm.__version__)\"
python3 -c \"import torch; print('torch', torch.__version__, 'hip', getattr(torch.version, 'hip', '?'))\"
python3 -c \"import quark; print('quark', quark.__file__)\"
python3 -c \"from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkMoEMethod; print('quark_moe OK')\"
python3 -c \"import aiter; print('aiter OK')\"
" || die "import verification failed"

cat <<EOF

Upgrade complete.

Next: serve with deploy_droplet.sh (BF16) or point VLLM_FP8_MODEL at a
Quark-quantized checkpoint. Note: first torch.compile of any model on
vLLM 0.26 takes ~10-13 min (AOT cache persists for later boots).

Quark checkpoint serving:
  VLLM_FP8_MODEL=/models/<quark-output> bash scripts/deploy_droplet.sh
EOF
