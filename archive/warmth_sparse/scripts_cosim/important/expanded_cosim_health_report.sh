#!/usr/bin/env bash
# Health snapshot for expanded co-sim train+sweep pipeline.
# Exit: 0=all done, 1=in-progress, 2=failed/stalled, 3=needs rsync
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE_DIR="${ROOT}/logs/expanded_cosim_pipeline"
PID_FILE="${PHASE_DIR}/pipeline.pid"
STATE="${PHASE_DIR}/state.env"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

phase_ok() { [[ -f "${PHASE_DIR}/phase_${1}.done" ]]; }

echo "[$(timestamp)] === expanded cosim train+sweep health ==="

# Corpus check
warmth_jsonl=$(find "${ROOT}/simulation_data/gnn_datasets_4tasks_1060_warmth_v2" -name placements.jsonl 2>/dev/null | wc -l)
cont_jsonl=$(find "${ROOT}/simulation_data/gnn_datasets_4tasks_contention_v2" -name placements.jsonl 2>/dev/null | wc -l)
nu_lines=$(pipenv run python3 -c "
from pathlib import Path
p = Path('${ROOT}/simulation_data/gnn_datasets_4tasks_1060_warmth_v2/ds_00000/placements/placements.jsonl')
print(sum(1 for _ in open(p)) if p.exists() else 0)
" 2>/dev/null || echo 0)

echo "corpus: warmth_jsonl=${warmth_jsonl} cont_jsonl=${cont_jsonl} ds_00000_lines=${nu_lines}"

needs_rsync=0
if [[ "$nu_lines" -lt 2000 || "$cont_jsonl" -lt 850 ]]; then
  echo "STATUS: NEED_RSYNC (warmth cartesian or contention incomplete)"
  needs_rsync=1
fi

for ph in rsync ssc recache train sweep_wssm sweep_contention all; do
  if phase_ok "$ph"; then
    echo "phase_${ph}: DONE"
  else
    echo "phase_${ph}: pending"
  fi
done

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "pipeline_pid: ${pid} RUNNING"
  else
    echo "pipeline_pid: ${pid} DEAD (stale pid file)"
  fi
else
  echo "pipeline_pid: none"
fi

# Sweep progress
for sweep in "${ROOT}"/simulation_data/normal_sim_sweeps/wssm_expanded_nu_live_gate_*; do
  [[ -d "$sweep/results" ]] || continue
  n=$(find "$sweep/results" -name '*.json' 2>/dev/null | wc -l)
  echo "wssm_sweep: ${sweep##*/} results=${n}/9"
done
for sweep in "${ROOT}"/simulation_data/normal_sim_sweeps/contention_v2_expanded_live_gate_*; do
  [[ -d "$sweep/results" ]] || continue
  n=$(find "$sweep/results" -name '*.json' 2>/dev/null | wc -l)
  echo "contention_sweep: ${sweep##*/} results=${n}/9"
done

if phase_ok all; then
  echo "STATUS: ALL_COMPLETE"
  exit 0
fi

if [[ "$needs_rsync" -eq 1 ]]; then
  exit 3
fi

# Check for pipeline failure in log
latest_log=$(ls -t "${PHASE_DIR}"/pipeline_*.log 2>/dev/null | head -1 || true)
if [[ -n "$latest_log" ]] && grep -q "ERROR:" "$latest_log" 2>/dev/null; then
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    kill -0 "$pid" 2>/dev/null || {
      echo "STATUS: FAILED (pipeline dead with ERROR in log)"
      tail -5 "$latest_log" 2>/dev/null || true
      exit 2
    }
  fi
fi

echo "STATUS: IN_PROGRESS"
exit 1
