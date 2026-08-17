#!/usr/bin/env bash
# Foreground loop: health report every 30m until all datalab co-sim work is done.
# Emits AGENT_LOOP_TICK_COSIM for Cursor agent wake notifications.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
MONITOR_LOG="${LOG_DIR}/cosim_monitor_loop.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MONITOR_LOG"; }

run_tick() {
  local rc=0
  log "========== health tick =========="
  bash scripts_cosim/datalab/cosim_health_report.sh 2>&1 | tee -a "$MONITOR_LOG" || rc=$?

  if [[ "$rc" -eq 2 ]]; then
    log "UNHEALTHY — attempting remediation"
    bash scripts_cosim/datalab/cosim_health_remediate.sh 2>&1 | tee -a "$MONITOR_LOG" || true
    bash scripts_cosim/datalab/cosim_health_report.sh 2>&1 | tee -a "$MONITOR_LOG" || rc=$?
  fi

  if [[ "$rc" -eq 0 ]]; then
    log "ALL_COSIM_COMPLETE — monitor exiting"
    echo 'AGENT_LOOP_TICK_COSIM {"prompt":"Datalab co-sim monitor: ALL COMPLETE. Summarize final warmth/sparse non-unique + contention_v2 counts for the user.","status":"complete"}'
    return 0
  fi

  if [[ "$rc" -eq 3 ]]; then
    log "WARN: datalab unreachable (will retry next tick)"
  fi

  return 1
}

log "cosim monitor loop started (interval=${INTERVAL_SEC}s)"
log "Log file: ${MONITOR_LOG}"

# First tick immediately
run_tick && exit 0

while true; do
  log "sleeping ${INTERVAL_SEC}s until next tick..."
  sleep "$INTERVAL_SEC"
  run_tick && exit 0
  echo 'AGENT_LOOP_TICK_COSIM {"prompt":"Datalab co-sim 30m health tick: run scripts_cosim/datalab/cosim_health_report.sh; if UNHEALTHY run cosim_health_remediate.sh and fix; post concise progress (warmth/sparse SUCCESS counts, shard states, cont_jsonl) to chat. Continue monitoring until ALL_COSIM_COMPLETE.","status":"in_progress"}'
done
