#!/bin/zsh
# Install a launchd job that runs scripts/daily_run.sh at 03:30 local time,
# Tuesday to Saturday (after the US close has become a completed UTC day).
# A run missed while the Mac sleeps starts when it wakes.
# Remove with: scripts/uninstall_daily_run.sh
set -eu
ROOT="${0:A:h:h}"
LABEL="com.investmentai.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/investment_ai_logs"
chmod +x "$ROOT/scripts/daily_run.sh"
{
  cat <<HEAD
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$ROOT/scripts/daily_run.sh</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StandardOutPath</key><string>$ROOT/investment_ai_logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$ROOT/investment_ai_logs/launchd.err</string>
  <key>StartCalendarInterval</key>
  <array>
HEAD
  for day in 2 3 4 5 6; do
    echo "    <dict><key>Weekday</key><integer>$day</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>30</integer></dict>"
  done
  cat <<TAIL
  </array>
</dict>
</plist>
TAIL
} > "$PLIST"
plutil -lint "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL -> $PLIST"
