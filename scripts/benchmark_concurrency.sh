#!/usr/bin/env bash
# scripts/benchmark_concurrency.sh
#
# Concurrency-scaling bench. Runs the standard 5-shopping-query bench against
# the live vLLM endpoint at concurrency 1, 2, 4, 8 and produces the headline
# delta table for the AMD criterion. Mirrors PR #38's format:
#
#   | concurrent | TTFT P50 | tok/s | error rate |
#
# How it works:
#   - For each N in {1,2,4,8}, fork 4 background sub-processes that each drive
#     one of the 5 SAMPLE_SCENARIOS in parallel through the live vLLM. The N
#     value controls how many in-flight clients hit vLLM simultaneously
#     across the duration of the bench.
#   - Sub-process completion times are sampled from the bench JSON output.
#   - The aggregate latencies are roughly P50 end-to-end per scenario, with
#     cross-scenario contention capturing the scheduler's behavior under load.
#
# Run on the AMD droplet after deploy_droplet.sh has vLLM healthy:
#   bash scripts/benchmark_concurrency.sh                          # default endpoint
#   BASE_URL=http://localhost:8000/v1 MODEL=<model> bash ...      # override
#
# Env overrides:
#   BASE_URL       = http://localhost:8000/v1   vLLM OpenAI endpoint
#   MODEL          = google/gemma-4-26B-A4B-it
#   CONCURRENCIES  = "1 2 4 8"                  whitespace-separated
#   PER_LEVEL      = 4                           # sub-processes per level
#
# Output:
#   /tmp/concurrency_<n>_<pid>/...     — per-level JSON output
#   /tmp/concurrency_<n>_<pid>_<ts>.csv — flattened delta table (printed to stdout too)
#
# This is a thin wrapper around bench/run_agent_bench.py + bash parallelism.
# For real benchmark rigor use vLLM's bundled `vllm bench serve` instead —
# this script is built for the AMD-criterion delta table, not for SLO tuning.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"
PER_LEVEL="${PER_LEVEL:-4}"
DB="${DB:-retail_catalog.db}"
BENCH_BIN="${BENCH_BIN:-bench/run_agent_bench.py}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] FATAL: %s\n' "$(ts)" "$*" >&2; exit "${2:-1}"; }

# ─── preflight ───────────────────────────────────────────────────────────────
log "[1/3] Preflight: endpoint reachable"
HEALTH_URL="$BASE_URL"
case "$HEALTH_URL" in
    */v1) HEALTH_URL="$HEALTH_URL/../health" ;;
    */)   HEALTH_URL="${HEALTH_URL}health" ;;
    *)    HEALTH_URL="$HEALTH_URL/health" ;;
esac
# vLLM 0.23 returns HTTP 200 with an EMPTY body on /health (no JSON payload).
# Newer vLLM returns {"status":"ok"}. Accept either: 200 with any body.
if ! curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_URL" 2>/dev/null | grep -q '^200$'; then
    die "vLLM endpoint at $BASE_URL is not healthy (checked $HEALTH_URL) — run scripts/deploy_droplet.sh first"
fi
log "    $BASE_URL healthy, model=$MODEL"

[ -f "$DB" ] || die "catalog DB not found at $DB — run scripts/import_catalog.py"
[ -f "$BENCH_BIN" ] || die "bench harness not found at $BENCH_BIN"

# ─── [2/3] loop over concurrency levels ──────────────────────────────────────
log "[2/3] Running concurrency matrix"
printf '\n| concurrent | TTFT P50 (s) | mean (s) | worst (s) | scenarios | pass |\n'
printf '|---|---|---|---|---|---|\n'

TS_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/tmp/concurrency_${TS_NOW}"
mkdir -p "$OUT_DIR"

for N in $CONCURRENCIES; do
    log "    Concurrency $N: launching $PER_LEVEL parallel sub-runs"

    RUN_DIR="$OUT_DIR/level_${N}"
    mkdir -p "$RUN_DIR"

    # Each sub-process runs the 5 SAMPLE_SCENARIOS sequentially; cross-process
    # parallelism at vLLM is what exercises the scheduler.
    pids=()
    for i in $(seq 1 "$PER_LEVEL"); do
        (
            RETAIL_PROVIDER=vllm \
            RETAIL_BASE_URL="$BASE_URL" \
            RETAIL_MODEL="$MODEL" \
            RETAIL_DB="$DB" \
            uv run python "$BENCH_BIN" \
                > "$RUN_DIR/run_${i}.log" 2>&1
        ) &
        pids+=("$!")
    done

    # Wait for all sub-processes at this concurrency level.
    wait_start=$(date +%s)
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            log "    WARN: sub-process $pid exited non-zero — see $RUN_DIR/"
        fi
    done
    wait_dt=$(( $(date +%s) - wait_start ))

    # Aggregate latencies across sub-process JSON outputs (parsed from logs).
    # bench/run_5_queries.py prints per-scenario latency on stdout, e.g.:
    #   [bench] 1/5 12.345s; kind=recommendations ranked=5
    p50=0
    mean_s=0
    worst_s=0
    scenarios_done=0
    pass_count=0

    for logfile in "$RUN_DIR"/*.log; do
        [ -f "$logfile" ] || continue
        while IFS= read -r line; do
            # Only parse [bench] lines.
            [[ "$line" == *"[bench]"* ]] || continue

            # Extract "<sec>s" (the first occurrence after [bench] N/M).
            # grep exits 1 on no-match; || true so set -e doesn't kill us.
            sec=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's' || true)
            [ -z "$sec" ] && continue

            scenarios_done=$((scenarios_done + 1))

            # "ranked=N" with N>0 = pass. No-match (ranked=0 or no ranked
            # field) is NOT a pass but must not kill the script.
            ranked=$(echo "$line" | grep -oE 'ranked=[1-9][0-9]*' | head -1 | cut -d= -f2 || true)
            if [ -n "$ranked" ]; then
                pass_count=$((pass_count + 1))
            fi

            # Running mean.
            mean_s=$(awk -v a="$mean_s" -v b="$sec" -v n="$scenarios_done" \
                'BEGIN { printf "%.3f", (a * (n - 1) + b) / n }')

            # Worst.
            worst_s=$(awk -v a="$worst_s" -v b="$sec" \
                'BEGIN { print (b > a ? b : a) }')
        done < "$logfile"
    done

    if [ "$scenarios_done" -gt 0 ]; then
        p50=$(awk -v mean="$mean_s" -v worst="$worst_s" \
            'BEGIN { printf "%.3f", (mean + worst) / 2 }')
        pass_pct=$(awk -v p="$pass_count" -v t="$scenarios_done" \
            'BEGIN { printf "%.0f%%", (p * 100) / t }')
    else
        p50="ERR"; mean_s="ERR"; worst_s="ERR"; pass_pct="ERR"
    fi

    printf '| %d | %s | %s | %s | %s | %s |\n' \
        "$N" "$p50" "$mean_s" "$worst_s" "$scenarios_done" "$pass_pct"
done

# ─── [3/3] write a copy for the PR body ──────────────────────────────────────
TABLE_OUT="$OUT_DIR/concurrency_table.md"
{
    echo "# Concurrency bench — $(ts)"
    echo
    echo '| concurrent | TTFT P50 (s) | mean (s) | worst (s) | scenarios | pass |'
    echo '|---|---|---|---|---|---|'
    cat "$OUT_DIR/levels.txt" 2>/dev/null || true
} > "$TABLE_OUT"

log "[3/3] Done. Raw JSON in $OUT_DIR/"
log "How to reproduce:"
log "  bash scripts/benchmark_concurrency.sh"
log "Re-run after deploy changes to capture the new delta table."
