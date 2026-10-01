#!/bin/zsh
# Remove the scheduled Investment AI run installed by install_daily_run.sh.
set -u
LABEL="com.investmentai.daily"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Removed $LABEL"
