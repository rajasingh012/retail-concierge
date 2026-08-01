#!/usr/bin/env bash
# scripts/quantize_int8.sh
#
# ARCHITECTURE BOUNDARY: the catalog DB and app stay on the developer's
# laptop. This script runs on the GPU droplet and quantizes the BF16 model
# that lives in the container's HF cache. It needs NO catalog DB — W8A8 INT8
# uses dynamic activation quantization, so no calibration data is required.
#
# One-shot AMD Quark W8A8 INT8 quantization of the Gemma 4 26B A4B-it BF16
# checkpoint. Produces /models/gemma-4-26B-A4B-it-int8/ which
# scripts/deploy_droplet.sh picks up when VLLM_FP8_MODEL is set.
#
# Recipe provenance (proven, not invented):
#   nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8 on HF — same
#   Gemma4ForConditionalGeneration architecture, measured −0.08pp on
#   GSM8K vs BF16 (essentially lossless). Scheme: per-channel INT8
#   weights (ch_axis=0, symmetric, static) + per-token INT8 activations
#   (ch_axis=1, symmetric, dynamic). Exclusions stay BF16: lm_head,
#   *embed_tokens*, *vision_tower*, *embed_vision*.
#   https://huggingface.co/nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8
#
# Quark 0.12 API drift (vs the 0.11 recipe in that HF repo):
#   - amd_quark.tools.quark_quantize CLI        -> removed, use Python API
#   - quark.torch.quantization.config.Config      -> QConfig
#   - QuantizationConfig                          -> QLayerConfig
#   - ModelQuantizer.export_model()               -> quark.torch.export_safetensors()
#
# Tool-call accuracy gate (the make-or-break check):
#   - After quantizing, deploy with VLLM_FP8_MODEL=/models/gemma-4-26B-A4B-it-int8
#     and re-run bench/run_agent_bench.py from the laptop.
#   - Pass criteria: same finalize_recommendations pass rate as BF16; no
#     provenance_blocked non-empty where baseline was empty.
#   - If gate fails: revert to BF16, document the rejection per PR #7's
#     "Production recommendation remains FP16" pattern.
#
# Run on the droplet (the script must live there; app/DB stay on laptop):
#   ssh root@<droplet-ip>
#   docker exec -it rocm bash
#   bash /root/quantize_int8.sh
#
# Env overrides:
#   QUARK_OUT  = /models/gemma-4-26B-A4B-it-int8

set -euo pipefail

CTR_NAME="${CTR_NAME:-rocm}"
QUARK_OUT="${QUARK_OUT:-/models/gemma-4-26B-A4B-it-int8}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── [1/4] preflight: Quark + BF16 source in container ───────────────────────
log "[1/4] Preflight: Quark + BF16 source inside $CTR_NAME"

# Copy the BF16 locator helper into the container. Standalone .py avoids
# the multi-layer shell-quoting hell of inline python3 -c inside docker
# exec inside bash -c.
docker cp "$SCRIPT_DIR/_find_bf16.py" "$CTR_NAME":/tmp/_find_bf16.py 2>/dev/null || \
    docker exec "$CTR_NAME" sh -c "cat > /tmp/_find_bf16.py" < "$SCRIPT_DIR/_find_bf16.py" 2>/dev/null || \
    die "could not copy _find_bf16.py into container $CTR_NAME"

docker exec "$CTR_NAME" bash -c '
    python3 -c "import quark; print(\"quark\", quark.__version__)" 2>/dev/null \
        || { echo "quark not installed — run: pip install amd-quark"; exit 1; }
    python3 /tmp/_find_bf16.py > /tmp/bf16_path.txt
    if [ ! -s /tmp/bf16_path.txt ]; then
        echo "BF16 Gemma 4 26B A4B-it not in HF cache — run deploy_droplet.sh first" >&2
        exit 1
    fi
    echo "BF16 source: $(cat /tmp/bf16_path.txt)"
    echo "Files: $(ls $(cat /tmp/bf16_path.txt)/*.safetensors | wc -l) safetensors shards"
' || die "preflight failed — Quark not installed or BF16 model missing"

# ─── [2/4] run W8A8 INT8 quantization ────────────────────────────────────────
log "[2/4] Quark W8A8 INT8 quantization → $QUARK_OUT"

docker cp "$SCRIPT_DIR/_quark_quantize_int8.py" "$CTR_NAME":/tmp/_quark_quantize_int8.py 2>/dev/null || \
    docker exec "$CTR_NAME" sh -c "cat > /tmp/_quark_quantize_int8.py" < "$SCRIPT_DIR/_quark_quantize_int8.py" 2>/dev/null || \
    die "could not copy _quark_quantize_int8.py into container $CTR_NAME"

docker exec "$CTR_NAME" env \
    QUARK_OUT="$QUARK_OUT" \
    python3 /tmp/_quark_quantize_int8.py \
    || die "Quark quantization failed — see error above"

# ─── [3/4] verify output ─────────────────────────────────────────────────────
log "[3/4] Verifying output"
docker exec "$CTR_NAME" bash -c "
    if [ ! -d '$QUARK_OUT' ]; then echo 'output dir missing' >&2; exit 1; fi
    echo 'Output dir: $QUARK_OUT'
    ls -la '$QUARK_OUT' | head -20
    echo
    echo 'Safetensors shards:'
    ls '$QUARK_OUT'/*.safetensors 2>/dev/null | wc -l
    echo 'Total size:'
    du -sh '$QUARK_OUT'
    echo
    echo 'quantization_config in config.json:'
    python3 -c \"
import json
cfg = json.load(open('$QUARK_OUT/config.json'))
qc = cfg.get('quantization_config')
print(json.dumps(qc, indent=2)[:800] if qc else 'MISSING — not vLLM-loadable!')
\"
" || die "output verification failed"

# ─── [4/4] next steps ────────────────────────────────────────────────────────
cat <<EOF

Quantization complete.

Next steps (manual):

  1. Deploy serving the INT8 model (from laptop):
       ssh root@<droplet-ip> "VLLM_FP8_MODEL=$QUARK_OUT bash /root/deploy_droplet.sh"

  2. Run the tool-call accuracy gate (from laptop):
       RETAIL_PROVIDER=vllm \\
         RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \\
         RETAIL_MODEL=$QUARK_OUT \\
         uv run python bench/run_agent_bench.py

  3. Compare against the BF16 baseline (mean 9.6s / median 6.2s, Aug 1).
     Either:
       - Pass: ship INT8, claim the quantization bonus in PR body
       - Fail: revert to BF16, document rejection per PR #7 framing

Note: vLLM 0.23 loads Quark output via --quantization fp8? NO — Quark INT8
output uses --quantization quark (needs vLLM >= 0.26) OR vLLM may auto-detect
the quantization_config block. Verify the serve command picks it up.
EOF
