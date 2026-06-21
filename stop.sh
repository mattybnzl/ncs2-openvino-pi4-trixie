#!/bin/bash
# Cleanly stop the pan-tilt app + NCS2 inference service started by start.sh.
# Order matters: stop the app FIRST (releases the camera + Unix socket), then the
# inference service (releases the MYRIAD stick so the next start can re-acquire it).
# Graceful SIGTERM first, SIGKILL only if a process refuses to exit.
#
# Override the app dir via environment variable if running from elsewhere:
APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")" && pwd)}"

stop_proc() {
  local pattern="$1" name="$2"
  local pids
  pids=$(pgrep -f "$pattern")
  if [ -z "$pids" ]; then
    echo "  $name: not running"
    return
  fi
  echo "  $name: stopping (PIDs: $pids)"
  pkill -TERM -f "$pattern" 2>/dev/null
  # Wait up to 5s for a clean exit
  for _ in 1 2 3 4 5; do
    pgrep -f "$pattern" >/dev/null || { echo "  $name: stopped"; return; }
    sleep 1
  done
  echo "  $name: did not exit, forcing"
  pkill -KILL -f "$pattern" 2>/dev/null
  sleep 1
  pgrep -f "$pattern" >/dev/null && echo "  $name: STILL RUNNING - check manually" || echo "  $name: killed"
}

echo "Stopping pan-tilt stack..."
stop_proc "$APP_DIR/app.py"       "pan-tilt app"
stop_proc "$APP_DIR/ncs_infer.py" "NCS2 service"
echo "Done."
