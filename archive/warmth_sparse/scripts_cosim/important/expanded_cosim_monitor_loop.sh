#!/usr/bin/env bash
# Foreground 30m loop: rsync if needed → train+sweep pipeline until ALL_COMPLETE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"
PHASE_DIR="${ROOT}/logs/expanded_cosim_pipeline"
PID_FILE="${PHASE_DIR}/pipeline.pid"
MONITOR_LOG="${ROOT}/logs/expanded_cosim_monitor_loop.log"

mkdir -p "$PHASE_DIR" logs

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MONITOR_LOG"; }

start_pipeline() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      log "Pipeline already running pid=${pid}"
      return 0
    fi
  fi
  log "Starting expanded cosim train+sweep pipeline"
  nohup bash scripts_cosim/important/run_expanded_cosim_train_sweep.sh \
    >> "${PHASE_DIR}/pipeline_nohup.log" 2>&1 &
  echo $! > "$PID_FILE"
  log "Pipeline pid=$(cat "$PID_FILE")"
}

start_rsync() {
  if [[ -f "${PHASE_DIR}/rsync.pid" ]]; then
    local pid
    pid=$(cat "${PHASE_DIR}/rsync.pid")
    if kill -0 "$pid" 2>/dev/null; then
      log "Rsync already running pid=${pid}"
      return 0
    fi
  fi
  log "Starting datalab rsync"
  nohup bash scripts_cosim/transfer_expanded_cosim_from_datalab.sh \
    >> logs/transfer_expanded_cosim.log 2>&1 &
  echo $! > "${PHASE_DIR}/rsync.pid"
}

run_tick() {
  local rc=0
  log "========== health tick =========="
  bash scripts_cosim/important/expanded_cosim_health_report.sh 2>&1 | tee -a "$MONITOR_LOG" || rc=$?

  case "$rc" in
    0)
      log "ALL_COMPLETE — monitor exiting"
      echo 'AGENT_LOOP_TICK_TRAIN {"prompt":"Expanded co-sim train+sweep ALL COMPLETE. Summarize wssm + contention live gate results for user.","status":"complete"}'
      return 0
      ;;
    3)
      log "NEED_RSYNC — starting pull from datalab"
      start_rsync
      ;;
    2)
      log "UNHEALTHY — restarting pipeline (clear stale pid, resume from phase markers)"
      rm -f "$PID_FILE"
      start_pipeline
      ;;
    1)
      if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log "IN_PROGRESS but pipeline not running — starting"
        start_pipeline
      fi
      ;;
  esac
  return 1
}

log "expanded cosim monitor started (interval=${INTERVAL_SEC}s)"
log "Monitor log: ${MONITOR_LOG}"

run_tick && exit 0

while true; do
  log "sleeping ${INTERVAL_SEC}s..."
  sleep "$INTERVAL_SEC"
  run_tick && exit 0
  echo 'AGENT_LOOP_TICK_TRAIN {"prompt":"Expanded co-sim 30m tick: check expanded_cosim_health_report.sh; ensure rsync+pipeline running; post phase progress + sweep counts.","status":"in_progress"}'
done
