#!/usr/bin/env bash
# ROUND 2: repair of the functions in data/results/round1_judge/ham_can_sua_lan_2.csv (Advisor + Fixer, no
# verifier, resumable). Results: data/results/<BATCH>/ham_sua_lan_2_ket_qua.csv. Code: E2E/experiments/repair_round2.py
# The 18 functions whose round-1 patch introduced a new problem run as their own group, in their own batch
# (round2_regress), with the self-check on: prefix any command below with GROUP=regress, e.g.
#   GROUP=regress bash scripts/run_round2.sh start
#
#   bash scripts/run_round2.sh prepare        re-scan the round-1 patches locally (no GPU; resumable, Ctrl+C is safe)
#   bash scripts/run_round2.sh preview 1      print the Advisor + Fixer prompts of stt_goc=1 (no LLM call)
#   bash scripts/run_round2.sh start          start (or resume) in the background; waits while the server is down
#   bash scripts/run_round2.sh continue       same as start; only works on an existing batch
#   bash scripts/run_round2.sh stop           finish the current function, then exit
#   bash scripts/run_round2.sh stop-now       abort the current function now (it is re-queued)
#   bash scripts/run_round2.sh status         running?, progress, last log lines
#   bash scripts/run_round2.sh log            follow progress.log live (Ctrl+C only stops watching)
#   bash scripts/run_round2.sh export         write the full CSV now (add --blind to drop the round-1 judge columns)
#   bash scripts/run_round2.sh retry-failed   put failed functions back in the queue
#
# Settings: edit below, or override per call, e.g.  EXTRA_ARGS="--score-detector" bash scripts/run_round2.sh start
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root

GROUP="${GROUP:-main}"                             # main (the 614) | regress (the 18)
if [[ "$GROUP" == regress ]]; then
  BATCH="${BATCH:-round2_regress}"
  SELF_CHECK="${SELF_CHECK:-1}"
else
  BATCH="${BATCH:-round2}"                         # folder under data/results/
  SELF_CHECK="${SELF_CHECK:-0}"                    # 1 = compile + re-scan + guard the reply, one retry
fi
INPUT="${INPUT:-data/results/round1_judge/ham_can_sua_lan_2.csv}"
ACTIVE_HOURS="${ACTIVE_HOURS:-}"                   # e.g. 19:00-07:30; empty = whenever the server is up
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPORT_EVERY="${EXPORT_EVERY:-10}"                 # refresh the CSV every N finished functions
EXTRA_ARGS="${EXTRA_ARGS:-}"                       # --score-detector | --on-all-dismissed keep_raw

PY=.venv/bin/python
DIR="data/results/$BATCH"
PID_FILE="$DIR/runner.pid"
OUT_FILE="$DIR/runner.out"
PROGRESS="$DIR/progress.log"
COMMON=(--batch "$BATCH" --input "$INPUT" --group "$GROUP")

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
    echo "Already running (pid $pid). Use: bash scripts/run_round2.sh status"
    return 0
  fi
  mkdir -p "$DIR"
  local args=(-u -m E2E.experiments.repair_round2 run "${COMMON[@]}" --wait --poll-seconds "$POLL_SECONDS"
              --export-every "$EXPORT_EVERY")
  [[ -n "$ACTIVE_HOURS" ]] && args+=(--active-hours "$ACTIVE_HOURS")
  [[ "$SELF_CHECK" == 1 ]] && args+=(--self-check)
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
  nohup "$PY" "${args[@]}" >>"$OUT_FILE" 2>&1 &
  pid=$!
  echo "$pid" >"$PID_FILE"
  command -v caffeinate >/dev/null && nohup caffeinate -is -w "$pid" >/dev/null 2>&1 &
  sleep 3
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Runner exited immediately. Last output:"
    tail -n 20 "$OUT_FILE"
    return 1
  fi
  echo "Started round 2 '$BATCH' [group $GROUP, self-check $SELF_CHECK] (pid $pid), active hours: ${ACTIVE_HOURS:-any}, extra: ${EXTRA_ARGS:-none}."
  echo "  progress log : $PROGRESS"
  echo "  results      : $DIR/ham_sua_lan_2_ket_qua.csv (refreshed every $EXPORT_EVERY functions)"
}

stop() {
  local pid
  if ! pid="$(running_pid)"; then echo "Not running."; return 0; fi
  touch "$DIR/STOP"
  echo "Stop requested (pid $pid): it exits after the current function (or within ${POLL_SECONDS}s if waiting)."
}

stop_now() {
  local pid
  if ! pid="$(running_pid)"; then echo "Not running."; return 0; fi
  kill -INT "$pid" 2>/dev/null || true
  sleep 1
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$PID_FILE"; echo "Stopped. The interrupted function was re-queued."; return 0; }
    sleep 1
  done
  echo "Still running after 30s; force kill with: kill -9 $pid (the function is re-queued on next start)."
}

status() {
  local pid
  if pid="$(running_pid)"; then echo "Runner: RUNNING (pid $pid)"; else echo "Runner: stopped"; fi
  "$PY" -m E2E.experiments.repair_round2 status "${COMMON[@]}"
  if [[ -f "$PROGRESS" ]]; then
    echo "--- last lines of $PROGRESS"
    tail -n 6 "$PROGRESS"
  fi
}

case "${1:-}" in
  prepare) "$PY" -u -m E2E.experiments.repair_round2 prepare "${COMMON[@]}" ;;
  preview) [[ -n "${2:-}" ]] || { echo "Usage: bash scripts/run_round2.sh preview <stt_goc>"; exit 1; }
    # shellcheck disable=SC2086
    "$PY" -m E2E.experiments.repair_round2 preview "${COMMON[@]}" --stt "$2" $EXTRA_ARGS ;;
  start) start ;;
  continue)
    [[ -f "$DIR/state.sqlite" ]] || { echo "Batch '$BATCH' does not exist yet: use start."; exit 1; }
    start ;;
  stop) stop ;;
  stop-now) stop_now ;;
  status) status ;;
  log) mkdir -p "$DIR"; touch "$PROGRESS"; tail -n 30 -f "$PROGRESS" ;;
  export) shift; "$PY" -m E2E.experiments.repair_round2 export "${COMMON[@]}" "$@" ;;
  retry-failed) "$PY" -m E2E.experiments.repair_round2 retry-failed "${COMMON[@]}" ;;
  *) sed -n '2,19p' "$0"; exit 1 ;;
esac
