#!/usr/bin/env bash
# Wrap probe_daily.py with Discord notification on CAPTCHA / bot-wall stop.
# Exit codes:
#   0 = clean run, candidates logged
#   1 = stopped early due to CAPTCHA / bot-wall (alert sent)
#   2 = homepage pre-flight failed (alert sent)

set -uo pipefail

cd /home/rajasingh/retail-concierge

LOG=./data/dead_candidates.log
ALERT=./data/needs_human_attention.txt
DISCORD_TARGET="discord:#general"

date_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run_probe() {
    ./.venv/bin/python scripts/probe_daily.py \
        --db ./retail_catalog.db \
        --product-types CHAIR SHOES BOOT SANDAL HOME GROCERY \
                        HOME_BED_AND_BATH CELLULAR_PHONE_CASE \
        --workers 8
}

out=$(run_probe 2>&1)
rc=$?

case "$rc" in
    0)
        # Clean run. Summarize for Discord.
        new=$(grep -c "^$(date -u +%Y-%m-%d)," "$LOG" 2>/dev/null || echo 0)
        msg="RetailConcierge daily probe: clean run. New 404 candidates: $new. See $LOG."
        hermes send --to "$DISCORD_TARGET" --quiet "$msg" || true
        ;;
    1)
        # CAPTCHA / bot-wall stop
        first=$(grep -m1 "STOPPED EARLY" <<<"$out" || true)
        msg="RetailConcierge probe STOPPED EARLY (CAPTCHA / bot-wall). Our VPS IP appears flagged. $first See $ALERT."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
    2)
        # Homepage pre-flight failed
        msg="RetailConcierge probe ABORTED: amazon.in homepage returned CAPTCHA / bot-wall. VPS IP flagged. Waiting 24h before next attempt. See $ALERT."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
    *)
        msg="RetailConcierge probe failed with exit code $rc. Check logs."
        hermes send --to "$DISCORD_TARGET" "$msg" || true
        ;;
esac
