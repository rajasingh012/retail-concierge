#!/usr/bin/env bash
# scripts/quantize_int8.sh
#
# ARCHITECTURE BOUNDARY: the catalog DB and app stay on the developer's
# laptop. This script runs on the GPU droplet and quantizes the BF16 model
# that lives in the container's HF cache. It needs NO catalog DB - W8A8 INT8
# uses dynamic activation quantization, so no calibration data is required.
#
# One-shot AMD Quark W8A8 INT8 quantization. Dispatches to the correct
# recipe script based on --kind:
#   dense -> _quark_quantize_dense.py  (Gemma 4 12B / 31B; production path)
#   moe   -> _quark_quantize_moe.py    (Gemma 4 26B A4B; known broken - see below)
#
# Recipe provenance (proven, not invented):
#   dense: nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8 on HF - same
#     Gemma4ForConditionalGeneration architecture, measured -0.08pp on
#     GSM8K vs BF16 (essentially lossless). Scheme: per-channel INT8
#     weights (ch_axis=0, symmetric, static) + per-token INT8 activations
#     (ch_axis=1, symmetric, dynamic). Exclusions stay BF16: lm_head,
#     *embed_tokens*, *vision_tower*, *embed_vision*.
#     https://huggingface.co/nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8
#   moe:   nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 (HF) - MoE recipe
#     with pre-quantization expert rewrite + post-export key rename.
#     https://huggingface.co/nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8
#
# Quark 0.12 API drift (vs the 0.11 recipes):
#   - amd_quark.tools.quark_quantize CLI        -> removed, use Python API
#   - quark.torch.quantization.config.Config      -> QConfig
#   - QuantizationConfig                          -> QLayerConfig
#   - ModelQuantizer.export_model()               -> quark.torch.export_safetensors()
#
# Tool-call accuracy gate (the make-or-break check):
#   - After quantizing, deploy with VLLM_FP8_MODEL=$QUARK_OUT
#     and re-run bench/run_agent_bench.py from the laptop.
#   - Pass criteria: same finalize_recommendations pass rate as BF16; no
#     provenance_blocked non-empty where baseline was empty.
#   - If gate fails: revert to BF16, document the rejection per PR #7's
#     "Production recommendation remains FP16" pattern.
#
# Usage (on the droplet):
#   docker exec -it rocm bash
#   bash /root/quantize_int8.sh --kind dense --model google/gemma-4-12b-it
#   bash /root/quantize_int8.sh --kind moe   --model google/gemma-4-26B-A4B-it
#
# Env overrides:
#   CTR_NAME             = rocm
#   QUARK_OUT            = (defaults set by --kind + --model)
#   QUARK_PREFLIGHT_TRACE = 0   set to 1 for the optional FX trace dry-run
#                              (loads BF16 into VRAM, ~5-10 min). Off by default.
#   SKIP_VLLM_KEY_FIX    = 0   set to 1 to skip the post-quantize Quark->vLLM
#                              key rename + chat_template.jinja copy (dense only).

set -euo pipefail

# ─── Args ────────────────────────────────────────────────────────────────────
KIND=""
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind)
            KIND="${2:-}"; shift 2 ;;
        --model)
            MODEL="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$KIND" || -z "$MODEL" ]]; then
    echo "Usage: $0 --kind dense|moe --model <hf-repo-id>" >&2
    exit 2
fi
if [[ "$KIND" != "dense" && "$KIND" != "moe" ]]; then
    echo "FATAL: --kind must be 'dense' or 'moe' (got '$KIND')" >&2; exit 2
fi

# Map (kind, model) -> recipe script + default QUARK_OUT.
case "$KIND:$MODEL" in
    dense:google/gemma-4-12b-it)
        RECIPE="_quark_quantize_dense.py"
        QUARK_OUT="${QUARK_OUT:-/models/gemma-4-12b-it-int8}" ;;
    dense:google/gemma-4-31b-it)
        RECIPE="_quark_quantize_dense.py"
        QUARK_OUT="${QUARK_OUT:-/models/gemma-4-31b-it-int8}" ;;
    moe:google/gemma-4-26B-A4B-it)
        RECIPE="_quark_quantize_moe.py"
        QUARK_OUT="${QUARK_OUT:-/models/gemma-4-26B-A4B-it-int8-moe2}" ;;
    *)
        # Unknown combo but a valid kind - use the matching recipe script
        # and let QUARK_OUT default to a model-derived path.
        RECIPE="_quark_quantize_${KIND}.py"
        local_tag=$(echo "$MODEL" | tr '/: ' '___')
        QUARK_OUT="${QUARK_OUT:-/models/${local_tag}-int8}"
        echo "Note: unrecognized ($KIND, $MODEL) combo. Using recipe=$RECIPE, output=$QUARK_OUT." >&2 ;;
esac

CTR_NAME="${CTR_NAME:-rocm}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── [1/4] preflight: Quark + BF16 source in container ───────────────────────
log "[1/4] Preflight: Quark + BF16 source inside $CTR_NAME (kind=$KIND model=$MODEL)"

# Copy the helpers into the container. Standalone .py avoids the multi-
# layer shell-quoting hell of inline python3 -c inside docker exec inside
# bash -c.
copy_into() {
    local src="$1" dst="$2"
    docker cp "$src" "$CTR_NAME:$dst" 2>/dev/null \
        || docker exec "$CTR_NAME" sh -c "cat > $dst" < "$src" 2>/dev/null \
        || die "could not copy $(basename "$src") into container $CTR_NAME"
}

copy_into "$SCRIPT_DIR/_quark_common.py"   /tmp/_quark_common.py
copy_into "$SCRIPT_DIR/$RECIPE"            /tmp/$RECIPE
copy_into "$SCRIPT_DIR/_find_bf16.py"      /tmp/_find_bf16.py
copy_into "$SCRIPT_DIR/_quark_fix_vllm_keys.py" /tmp/_quark_fix_vllm_keys.py

docker exec "$CTR_NAME" bash -c "
    # 1a. Quark version (must be importable; specific version not enforced -
    #     Quark's 0.11 -> 0.12 rename is already handled by _quark_common.py).
    python3 -c 'import quark; print(\"    quark       :\", quark.__version__)' \
        || { echo 'quark not installed - run: uv pip install --system amd-quark' >&2; exit 1; }

    # 1b. transformers version + class import for the requested kind.
    #     dense -> Gemma4UnifiedForConditionalGeneration (12B Unified; needs
    #              transformers >= 5.10.1, see DEPLOYMENT_JOURNAL.md pre-flight).
    #     moe   -> Gemma4ForConditionalGeneration (26B A4B; older class).
    case '$KIND' in
        dense)
            python3 -c '
import transformers
print(\"    transformers:\", transformers.__version__)
from transformers import Gemma4UnifiedForConditionalGeneration
print(\"    Gemma4UnifiedForConditionalGeneration: importable\")
' || { echo '' >&2; echo 'transformers is missing Gemma4UnifiedForConditionalGeneration.' >&2; echo 'This class landed in transformers v5.10.1 (2026-06-03).' >&2; echo 'Fix inside the container:' >&2; echo '  uv pip install --system \"transformers>=5.10.1\"' >&2; exit 1; }
            ;;
        moe)
            python3 -c '
import transformers
print(\"    transformers:\", transformers.__version__)
from transformers import Gemma4ForConditionalGeneration
print(\"    Gemma4ForConditionalGeneration: importable\")
' || { echo '' >&2; echo 'transformers is missing Gemma4ForConditionalGeneration.' >&2; echo 'The MoE recipe uses the older class. If upgrading transformers broke this,' >&2; echo 'pin transformers<5.10 or update the recipe to the Unified class.' >&2; exit 1; }
            ;;
    esac

    # 1c. BF16 model snapshot exists in HF cache. Must run BEFORE 1d
    #     (FX trace dry-run needs the BF16 path on disk).
    python3 /tmp/_find_bf16.py --model '$MODEL' --quiet > /tmp/bf16_path.txt
    if [ ! -s /tmp/bf16_path.txt ]; then
        echo 'BF16 model matching $MODEL not in HF cache - run deploy_droplet.sh first' >&2
        exit 1
    fi
    echo \"    BF16 source : \$(cat /tmp/bf16_path.txt)\"
    echo \"    Files       : \$(ls \$(cat /tmp/bf16_path.txt)/*.safetensors | wc -l) safetensors shards\"

    # 1d. (Optional, opt-in) FX trace dry-run. Verifies Quark can trace the
    #     model with dataloader=None before we commit to a full quantize.
    #     Costs ~5-10 min (loads BF16 into VRAM, no calibration). Enable with
    #     QUARK_PREFLIGHT_TRACE=1 on the host when something looks off.
    if [ \"\${QUARK_PREFLIGHT_TRACE:-0}\" = \"1\" ]; then
        echo '    [optional] FX trace dry-run (QUARK_PREFLIGHT_TRACE=1)...'
        python3 -c '
import sys
sys.path.insert(0, \"/tmp\")
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    QConfig, QLayerConfig, QTensorConfig, Dtype,
)
from quark.torch.quantization.config.type import (
    RoundType, ScaleType, QSchemeType,
)
from quark.torch.quantization.observer import PerChannelMinMaxObserver
from _quark_common import load_bf16

# Conservative exclude - we only care that trace + observer attach succeed,
# not whether the exact exclude set matches the recipe.
weight_spec = QTensorConfig(dtype=Dtype.int8, observer_cls=PerChannelMinMaxObserver,
    symmetric=True, is_dynamic=False, qscheme=QSchemeType.per_channel, ch_axis=0,
    round_method=RoundType.round, scale_type=ScaleType.float)
input_spec = QTensorConfig(dtype=Dtype.int8, observer_cls=PerChannelMinMaxObserver,
    symmetric=True, is_dynamic=True, qscheme=QSchemeType.per_channel, ch_axis=1,
    round_method=RoundType.round, scale_type=ScaleType.float)
q_cfg = QConfig(global_quant_config=QLayerConfig(input_tensors=input_spec, weight=weight_spec),
                exclude=[\"lm_head\", \"*embed_tokens*\", \"*vision_tower*\", \"*embed_vision*\"])

model_in = open(\"/tmp/bf16_path.txt\").read().strip()
if \"$KIND\" == \"dense\":
    from transformers import Gemma4UnifiedForConditionalGeneration as Cls
else:
    from transformers import Gemma4ForConditionalGeneration as Cls
model, _ = load_bf16(model_in, Cls)
mq = ModelQuantizer(q_cfg, multi_device=True)
model = mq.quantize_model(model, dataloader=None)
print(\"    FX trace + quantize attach: OK\")
' || { echo 'FX trace failed. Likely cause: Quark needs a dummy input tensor for' >&2; echo 'multimodal models with dataloader=None. See DEPLOYMENT_JOURNAL.md Check B.' >&2; exit 1; }
    fi
" || die "preflight failed"

# ─── [2/4] run W8A8 INT8 quantization ────────────────────────────────────────
log "[2/4] Quark W8A8 INT8 ($KIND) -> $QUARK_OUT"

docker exec "$CTR_NAME" env \
    QUARK_OUT="$QUARK_OUT" \
    python3 "/tmp/$RECIPE" \
    || die "Quark quantization failed - see error above"

# ─── [2.5/4] post-quantize vLLM key fixup (dense only) ──────────────────────
# Quark 0.12 exports Gemma4Unified weights with names vLLM's gemma4_unified
# loader does not map (embed_vision.multimodal_embedder.*, embed_vision.patch_*).
# The dense path must run the rename fixup + chat_template.jinja copy before
# the output is vLLM-loadable. The MoE recipe handles its own key rename
# (rename_keys) inside _quark_quantize_moe.py, so it is skipped here.
# See scripts/_quark_fix_vllm_keys.py docstring + DEPLOYMENT_JOURNAL.md.
if [ "$KIND" = "dense" ] && [ "${SKIP_VLLM_KEY_FIX:-0}" != "1" ]; then
    log "[2.5/4] Quark->vLLM key fixup + chat template -> $QUARK_OUT"
    docker exec "$CTR_NAME" python3 /tmp/_quark_fix_vllm_keys.py "$QUARK_OUT" \
        || die "Quark->vLLM key fixup failed"
fi

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
print(json.dumps(qc, indent=2)[:800] if qc else 'MISSING - not vLLM-loadable!')
\"
" || die "output verification failed"

# ─── [4/4] next steps ────────────────────────────────────────────────────────
cat <<EOF

Quantization complete ($KIND).

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

Note: vLLM 0.23 loads Quark output via --quantization fp8? NO - Quark INT8
output uses --quantization quark (needs vLLM >= 0.26) OR vLLM may auto-detect
the quantization_config block. Verify the serve command picks it up.
EOF
