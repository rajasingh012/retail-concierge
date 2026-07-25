#!/usr/bin/env bash
# Wrap probe_daily.py with Discord notification on CAPTCHA / bot-wall stop.
# Exit codes:
#   0 = clean run, candidates auto-flipped (dead signal only)
#   1 = stopped early due to CAPTCHA / bot-wall
#   2 = homepage pre-flight failed

set -uo pipefail

cd /home/rajasingh/retail-concierge

LOG=./data/dead_candidates.log
ALERT=./data/needs_human_attention.txt
DISCORD_TARGET="discord:#general"

run_probe() {
    ./.venv/bin/python scripts/probe_daily.py \
        --db ./retail_catalog.db \
        --batch-size 100 \
        --workers 4
}

out=$(run_probe 2>&1)
rc=$?

case "$rc" in
    0)
        # Clean run. Summarize for Discord.
        flipped=$(grep -c "Auto-flipped to url_active=0" <<<"$out" || true)
        new=$(grep -A0 "^$(date -u +%Y-%m-%d)," "$LOG" 2>/dev/null | wc -l || echo 0)
        msg="RetailConcierge daily probe: clean run. Sampled 100 rows, auto-flipped $flipped confirmed-dead listings. See $LOG."
        hermes send --to "$DISCORD_TARGET" --quiet "$msg" || true
        ;;
    1)
        msg="RetailConcierge probe STOPPED EARLY (CAPTCHA / bot-wall). VPS IP appears flagged. See $ALERT."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
    2)
        msg="RetailConcierge probe ABORTED: amazon.in homepage returned CAPTCHA / bot-wall. VPS IP flagged. See $ALERT."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
    *)
        msg="RetailConcierge probe failed with exit code $rc. Check logs."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
esac
