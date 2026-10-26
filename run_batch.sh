#!/usr/bin/env bash
# Background batch audit over the GPU server (single-pass repair, resumable).
#
#   bash run_batch.sh start         start (or resume) in the background; it keeps going until stopped
#   bash run_batch.sh continue      same as start; only works on an existing batch
#   bash run_batch.sh stop          finish the current contract, then exit
#   bash run_batch.sh stop-now      abort the current contract now (it is re-queued)
#   bash run_batch.sh status        running?, progress, functions, ETA, last log lines
#   bash run_batch.sh log           follow progress.log live (Ctrl+C only stops watching)
#   bash run_batch.sh export        refresh the full CSV/JSONL now
#   bash run_batch.sh export 100    write the snapshot of the first 100 finished contracts now
#   bash run_batch.sh retry-failed  put failed contracts back in the queue
#
# Settings: edit below, or override per call, e.g.  ACTIVE_HOURS=20:00-07:00 bash run_batch.sh start
set -euo pipefail
cd "$(dirname "$0")"

BATCH="${BATCH:-main}"                             # folder under E2E/batch_runs/
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100}"            # fixed CSV of the first 100, 200, ... contracts; the run continues
TARGET="${TARGET:-}"                               # optional: stop for good after N done contracts; empty = never
INPUT_DIR="${INPUT_DIR:-E2E/data/contracts/smartbugs_wild}"
ACTIVE_HOURS="${ACTIVE_HOURS:-}"                   # e.g. 19:00-07:30; empty = whenever the server is up
EXPORT_EVERY="${EXPORT_EVERY:-5}"                  # refresh the CSV every N finished contracts
POLL_SECONDS="${POLL_SECONDS:-60}"                 # how often to re-check server / time window
EXTRA_ARGS="${EXTRA_ARGS:-}"                       # e.g. "--max-target-units 20"

PY=.venv/bin/python
DIR="E2E/batch_runs/$BATCH"
PID_FILE="$DIR/runner.pid"
OUT_FILE="$DIR/runner.out"
PROGRESS="$DIR/progress.log"

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then echo "$pid"; return 0; fi
  rm -f "$PID_FILE"
  return 1
}

start() {
  local pid
  if pid="$(running_pid)"; then
    echo "Already running (pid $pid). Use: bash run_batch.sh status"
    return 0
  fi
  mkdir -p "$DIR"
  local args=(-u -m E2E.tools.batch_audit run --batch "$BATCH" --input-dir "$INPUT_DIR" --shuffle
              --wait --poll-seconds "$POLL_SECONDS" --export-every "$EXPORT_EVERY"
              --snapshot-every "$SNAPSHOT_EVERY")
  [[ -n "$TARGET" ]] && args+=(--target-done "$TARGET")
  [[ -n "$ACTIVE_HOURS" ]] && args+=(--active-hours "$ACTIVE_HOURS")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
  nohup "$PY" "${args[@]}" >>"$OUT_FILE" 2>&1 &
  pid=$!
  echo "$pid" >"$PID_FILE"
  # Keep the Mac awake while the runner is alive (-s needs AC power; the display may still sleep).
  # Closing the lid still sleeps it unless `sudo pmset -a disablesleep 1` is set.
  command -v caffeinate >/dev/null && nohup caffeinate -is -w "$pid" >/dev/null 2>&1 &
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Runner exited immediately. Last output:"
    tail -n 20 "$OUT_FILE"
    return 1
  fi
  echo "Started batch '$BATCH' (pid $pid): stop target: ${TARGET:-none}, active hours: ${ACTIVE_HOURS:-any}."
  echo "  progress log : $PROGRESS"
  echo "  snapshots    : $DIR/${BATCH}_first${SNAPSHOT_EVERY}_functions.csv, _first$((SNAPSHOT_EVERY * 2))_..., (run continues)"
  echo "  live results : $DIR/${BATCH}_functions.csv (refreshed every $EXPORT_EVERY contracts)"
  echo "  watch        : bash run_batch.sh log"
}

stop() {
  local pid
  if ! pid="$(running_pid)"; then echo "Not running."; return 0; fi
  touch "$DIR/STOP"
  echo "Stop requested (pid $pid): it exits after the current contract (or within ${POLL_SECONDS}s if waiting)."
}

stop_now() {
  local pid
  if ! pid="$(running_pid)"; then echo "Not running."; return 0; fi
  kill -INT "$pid" 2>/dev/null || true
  sleep 1
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$PID_FILE"; echo "Stopped. The interrupted contract was re-queued."; return 0; }
    sleep 1
  done
  echo "Still running after 30s; force kill with: kill -9 $pid (the contract is re-queued on next start)."
}

status() {
  local pid
  if pid="$(running_pid)"; then echo "Runner: RUNNING (pid $pid)"; else echo "Runner: stopped"; fi
  if [[ ! -f "$DIR/state.sqlite" ]]; then echo "Batch '$BATCH' has not been started yet."; return 0; fi
  local args=(status --batch "$BATCH" --snapshot-every "$SNAPSHOT_EVERY")
  [[ -n "$TARGET" ]] && args+=(--target-done "$TARGET")
  "$PY" -m E2E.tools.batch_audit "${args[@]}"
  if [[ -f "$PROGRESS" ]]; then
    echo "--- last lines of $PROGRESS"
    tail -n 6 "$PROGRESS"
  fi
}

case "${1:-}" in
  start) start ;;
  continue)
    [[ -f "$DIR/state.sqlite" ]] || { echo "Batch '$BATCH' does not exist yet: use start."; exit 1; }
    start ;;
  stop) stop ;;
  stop-now) stop_now ;;
  status) status ;;
  log) touch "$PROGRESS" 2>/dev/null || { mkdir -p "$DIR"; touch "$PROGRESS"; }; tail -n 30 -f "$PROGRESS" ;;
  export)
    if [[ -n "${2:-}" ]]; then "$PY" -m E2E.tools.batch_audit export --batch "$BATCH" --first "$2"
    else "$PY" -m E2E.tools.batch_audit export --batch "$BATCH"; fi ;;
  retry-failed) "$PY" -m E2E.tools.batch_audit retry-failed --batch "$BATCH" ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
