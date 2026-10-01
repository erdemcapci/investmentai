#!/bin/zsh
# Scheduled Investment AI run: one capture per completed trading day.
# Builds the point-in-time prediction history that --validation-report scores.
set -u
ROOT="${0:A:h:h}"
LOG_DIR="$ROOT/investment_ai_logs"
LOCK="$LOG_DIR/daily_run.lock"
mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

# Skip if a previous run is still going (mkdir is atomic).
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) skipped: previous run still active" >> "$LOG_DIR/daily_run.log"
    exit 0
fi
trap 'rmdir "$LOCK"' EXIT

LOG="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ).log"
"$ROOT/.venv/bin/python" main.py > "$LOG" 2>&1
STATUS=$?
echo "$(date -u +%FT%TZ) exit=$STATUS log=$LOG" >> "$LOG_DIR/daily_run.log"
# Keep the last 60 run logs.
ls -1t "$LOG_DIR"/run-*.log 2>/dev/null | tail -n +61 | xargs rm -f
exit $STATUS
