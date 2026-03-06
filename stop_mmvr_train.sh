#!/usr/bin/env bash
set -u

# Gracefully stop mmVR nohup training launcher:
# 1) find outer "bash -lc ... train_kpt.py ..." process
# 2) send SIGTERM and wait
# 3) escalate to SIGKILL if needed
# 4) clean possible immediate shutdown command

WAIT_SECONDS="${WAIT_SECONDS:-15}"

usage() {
  cat <<'EOF'
Usage:
  ./stop_mmvr_train.sh               # auto-find launcher PID(s) and stop
  ./stop_mmvr_train.sh <PID>         # stop a specific launcher PID
  WAIT_SECONDS=30 ./stop_mmvr_train.sh
EOF
}

is_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

find_launcher_pids() {
  ps -eo pid=,cmd= | awk '
    /bash -lc/ &&
    /python train_kpt.py/ &&
    /python train_cls.py/ &&
    !/awk/ {
      print $1
    }'
}

terminate_pid() {
  local pid="$1"
  local waited=0

  if ! is_alive "$pid"; then
    echo "[skip] PID $pid not running"
    return 0
  fi

  echo "[term] sending SIGTERM to PID $pid"
  kill -TERM "$pid" 2>/dev/null || true

  while is_alive "$pid" && [ "$waited" -lt "$WAIT_SECONDS" ]; do
    sleep 1
    waited=$((waited + 1))
  done

  if is_alive "$pid"; then
    echo "[kill] PID $pid still alive after ${WAIT_SECONDS}s, sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
  else
    echo "[ok] PID $pid exited gracefully"
  fi
}

cleanup_shutdown() {
  local sd_pids
  sd_pids="$(pgrep -f '/sbin/shutdown -h now' || true)"
  if [ -n "$sd_pids" ]; then
    echo "$sd_pids" | while read -r spid; do
      [ -n "$spid" ] || continue
      echo "[term] killing shutdown PID $spid"
      kill -TERM "$spid" 2>/dev/null || true
    done
  fi

  # Cancel any scheduled shutdown if present.
  shutdown -c >/dev/null 2>&1 || true
}

main() {
  local pids

  if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
  fi

  if [ $# -gt 1 ]; then
    usage
    exit 1
  fi

  if [ $# -eq 1 ]; then
    pids="$1"
  else
    pids="$(find_launcher_pids)"
  fi

  if [ -z "$pids" ]; then
    echo "[info] no matching mmVR launcher process found"
  else
    echo "[info] target PID(s): $pids"
    for pid in $pids; do
      terminate_pid "$pid"
    done
  fi

  cleanup_shutdown

  echo "[check] remaining related processes:"
  pgrep -af 'python train_kpt.py|python train_cls.py|bash -lc .*train_kpt.py|/sbin/shutdown -h now' || echo "none"
}

main "$@"
