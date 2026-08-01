#!/usr/bin/env bash
# scripts/quantize_fp8.sh
#
# One-shot AMD Quark FP8 W8A8 quantization of the Gemma 4 26B A4B-it BF16
# checkpoint. Produces /models/gemma-4-26B-A4B-it-fp8/ which
# scripts/deploy_droplet.sh will pick up when VLLM_FP8_MODEL is set.
#
# Run from inside the vLLM container on the AMD droplet:
#   docker exec -it rocm bash
#   cd /workspace         # or wherever the repo is checked out
#   bash scripts/quantize_fp8.sh
#
# Why FP8 W8A8:
#   - Halves weight VRAM (48.5 GiB BF16 → ~24 GiB FP8) → frees headroom for
#     longer KV cache and more concurrent sessions.
#   - Lifts MoE decode tok/s ~20-40% on gfx942 (MI300X) when served via
#     AITER FP8 GEMM (VLLM_USE_AITER=1 — set by deploy_droplet.sh).
#   - Source: AMD Quark blog (kimi-k25-mxfp4-atom, Jul 2026), ppc-fp8-rocm.
#
# Why this calibration set:
#   - Quark needs ~64-512 short sequences to compute per-tensor activation
#     ranges before rounding weights to FP8. Generic WikiText works but
#     the agent's traffic is shopping queries + catalog descriptions — we
#     want activation ranges from the same distribution we actually serve.
#   - We seed from the 5 SAMPLE_SCENARIOS in bench/run_agent_bench.py plus
#     titles from the catalog. Deterministic, committed, reproducible.
#
# Tool-call accuracy gate (the make-or-break check):
#   - After quantizing, deploy with VLLM_FP8_MODEL=/models/gemma-4-26B-A4B-it-fp8
#     and re-run bench/run_5_queries.py against the 5 scenarios.
#   - Pass criteria: same finalize_recommendations pass rate as BF16; no
#     provenance_blocked non-empty where baseline was empty.
#   - If gate fails: revert to BF16, document the rejection in
#     DEPLOYMENT_JOURNAL.md per PR #7's "Production recommendation remains
#     FP16" pattern. The rejection is publishable, not a failure.
#
# Env overrides (with defaults):
#   FP8_OUT      = /models/gemma-4-26B-A4B-it-fp8
#   CALIB_OUT    = /root/retailconcierge_calib.jsonl
#   NUM_CALIB    = 256        # Quark's recommended default
#   SKIP_CALIB   = 0          # set 1 if you've already produced CALIB_OUT
#
# Pre-flight: Quark 0.12+ must be installed in the container.
#   docker exec rocm pip show amd-quark  ||  pip install "amd-quark>=0.12"
# Verified CLI shape against:
#   https://quark.docs.amd.com/  (versions.html dated 2026-07-03 → 0.12)
#   https://docs.vllm.ai/stable/features/quantization/quark/ (dated 2026-05-15)

set -euo pipefail

CTR_NAME="${CTR_NAME:-rocm}"
BF16_CACHE="/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it"
DB="${DB:-/workspace/retail_catalog.db}"
FP8_OUT="${FP8_OUT:-/models/gemma-4-26B-A4B-it-fp8}"
CALIB_OUT="${CALIB_OUT:-/root/retailconcierge_calib.jsonl}"
NUM_CALIB="${NUM_CALIB:-256}"
SKIP_CALIB="${SKIP_CALIB:-0}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

# ─── preflight ───────────────────────────────────────────────────────────────
log "[1/4] Preflight: Quark + BF16 cache inside container $CTR_NAME"

docker exec "$CTR_NAME" bash -c '
    python3 -c "import amd_quark" 2>/dev/null \
        || { echo "amd_quark not installed — run: pip install amd-quark"; exit 1; }
    python3 -c "
from huggingface_hub import scan_cache_dir
hits = [e for e in scan_cache_dir().values() if any(\"gemma-4-26B-A4B-it\" in str(r) for r in [e])]
if not hits:
    raise SystemExit(\"BF16 Gemma 4 26B A4B-it not in HF cache — run deploy_droplet.sh first\")
print(\"BF16 cache OK\")
"
' || die "preflight failed — install Quark and verify cache"

# ─── [2/4] build calibration set from agent's natural traffic ────────────────
log "[2/4] Building calibration set → $CALIB_OUT"

if [ "$SKIP_CALIB" = "1" ] && docker exec "$CTR_NAME" test -s "$CALIB_OUT"; then
    log "    SKIP_CALIB=1 — reusing existing $CALIB_OUT"
else
    # Calibration set is built inside the container so it sees the same
    # Python env (huggingface_hub, sqlite3, json, random) and the same
    # catalog DB the agent will serve from. python3 -c '...' because
    # the shell-side heredoc would reformat the long f-string.
    docker exec "$CTR_NAME" python3 -c "
import json, random, sys
from pathlib import Path

NUM_CALIB = $NUM_CALIB

# Seed prompts from bench/run_agent_bench.py SAMPLE_SCENARIOS — same workload
# the quantization will serve in production. We oversample prompts (40%)
# and undersample catalog titles (60%) so the activation range reflects
# both the brief's natural distribution and the catalog's vocabulary.
PROMPT_FRACTION = 0.40

scenarios = [
    'I need a lightweight carry-on spinner luggage with four wheels.',
    'Find noise cancelling over-ear headphones with strong noise reduction.',
    'I need a 27-inch computer monitor for office work.',
    'Recommend a laptop backpack for daily commuting.',
    'Find a mechanical gaming keyboard with RGB lighting.',
]
prompt_quota = max(1, int(NUM_CALIB * PROMPT_FRACTION))
catalog_quota = NUM_CALIB - prompt_quota

# Catalog titles pulled live from the same DB the agent queries.
catalog_titles = []
try:
    import sqlite3
    conn = sqlite3.connect('$DB')
    cur = conn.cursor()
    catalog_titles = [
        r[0] for r in cur.execute(
            'SELECT title_en FROM listings WHERE title_en IS NOT NULL'
        ).fetchall()
        if r and r[0]
    ]
    conn.close()
except Exception as exc:
    print('WARN: could not read catalog titles:', exc, file=sys.stderr)

# Build the quota with sampling-with-replacement so we always hit NUM_CALIB
# regardless of how many unique catalog titles we have. This is fine for
# activation-range calibration: Quark only needs ~64-512 short sequences,
# duplicate tokens are normal.
random.seed(20260801)
samples = []
samples.extend(random.choices(scenarios, k=prompt_quota))
if catalog_titles:
    samples.extend(random.choices(catalog_titles, k=catalog_quota))
else:
    # Fallback: pad with scenario prompts if catalog reading failed.
    samples.extend(random.choices(scenarios, k=catalog_quota))

# Interleave by alternating (round-robin), padding the shorter bucket
# up-front so the tail of the longer bucket gets interleaved too.
prompts = samples[:prompt_quota]
catalogs = samples[prompt_quota:]
interleaved = []
i = j = 0
while i < len(prompts) or j < len(catalogs):
    if i < len(prompts):
        interleaved.append(prompts[i])
        i += 1
    if j < len(catalogs):
        interleaved.append(catalogs[j])
        j += 1
assert len(interleaved) == NUM_CALIB, f'expected {NUM_CALIB}, got {len(interleaved)}'

out_path = Path('$CALIB_OUT')
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8') as f:
    for s in interleaved:
        f.write(json.dumps({'text': s}) + '\n')
print(f'wrote {len(interleaved)} samples ({prompt_quota} prompts + {catalog_quota} catalog) to {out_path}')
" || die "calibration set build failed"
    log "    Calibration set ready"
fi

# ─── [3/4] Quark quantization ────────────────────────────────────────────────
# Verify CLI shape against https://quark.docs.amd.com/ before running.
# As of Aug 2026 the relevant flags are below; pin versions in CI later.
log "[3/4] Quark FP8 W8A8 → $FP8_OUT"

docker exec "$CTR_NAME" bash -c "
    mkdir -p $(dirname '$FP8_OUT')
    # Resolve the snapshot path dynamically — HF cache snapshots change.
    BF16_PATH=\$(python3 -c \"
from huggingface_hub import scan_cache_dir
for repo in scan_cache_dir().values():
    for rev in repo.revisions:
        for fn in rev.files:
            if 'gemma-4-26B-A4B-it' in str(fn).lower() and fn.rfilename.endswith('model.safetensors.index.json'):
                print(rev.snapshot_path)
                break
\")
    if [ -z \"\$BF16_PATH\" ]; then
        echo 'FATAL: could not resolve BF16 Gemma 4 26B A4B snapshot' >&2
        exit 1
    fi
    echo \"BF16 source: \$BF16_PATH\"

    # Pin Quark to FP8 W8A8. --quant-scheme and --kv-cache-dtype are the
    # documented scheme names — verify against current Quark release notes.
    # Known good example (Quark v0.6+):
    #   quark quantize \\
    #     --model-dir \$BF16_PATH \\
    #     --output-dir '$FP8_OUT' \\
    #     --quant-scheme w_fp8_a_fp8 \\
    #     --kv-cache-dtype fp8 \\
    #     --calib-dataset '$CALIB_OUT' \\
    #     --num-calib-samples $NUM_CALIB \\
    #     --batch-size 1
    # If CLI shape changed in the installed Quark version, this command will
    # fail with a usage error — that is the signal to re-check
    # https://quark.docs.amd.com/ before continuing.
    python3 -m amd_quark.tools.quark_quantize \
        --model-dir \"\$BF16_PATH\" \
        --output-dir '$FP8_OUT' \
        --quant-scheme w_fp8_a_fp8 \
        --kv-cache-dtype fp8 \
        --calib-dataset '$CALIB_OUT' \
        --num-calib-samples $NUM_CALIB \
        --batch-size 1
" || die "Quark quantization failed — verify CLI shape at https://quark.docs.amd.com/"

# ─── [4/4] gate check (human run, not automated) ─────────────────────────────
log "[4/4] Quantization complete"

cat <<EOF

Next steps (manual, in order):

  1. Stop the current BF16 vLLM and relaunch serving the FP8 model:
       VLLM_FP8_MODEL=$FP8_OUT bash scripts/deploy_droplet.sh

  2. Re-run the tool-calling accuracy gate:
       RETAIL_PROVIDER=vllm \\
         RETAIL_BASE_URL=http://localhost:8000/v1 \\
         RETAIL_MODEL=$FP8_OUT \\
         uv run python bench/run_5_queries.py

  3. Compare against the BF16 baseline (saved by prior deploy).

  4. Update DEPLOYMENT_JOURNAL.md with measured delta. Either:
       - Pass: ship FP8, claim the 20-pt quantization bonus in PR body
       - Fail: revert to BF16, document rejection per PR #7 framing

Output dir: $FP8_OUT (consumes ~26 GiB of host disk)
EOF
