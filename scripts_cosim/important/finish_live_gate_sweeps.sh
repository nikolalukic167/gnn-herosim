#!/usr/bin/env bash
# Idempotent closure: validate live-gate results, run compares, write phase markers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/finish_live_gate_sweeps_${TS}.log"
PHASE_DIR="logs/strategic_merge_pipeline"
WEIGHTED_PHASE_DIR="logs/merged_contention_pipeline"

STRATEGIC_WSSM="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
STRATEGIC_CONT="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
WEIGHTED_CONT="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

count_sim_results() {
  local dir="$1"
  find "$dir" -maxdepth 1 -name '*.json' ! -name '*.decode_stats.json' -type f 2>/dev/null | wc -l
}

valid_rtt_count() {
  local dir="$1"
  local n=0
  while IFS= read -r f; do
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${f}')).get('total_rtt', 0))" 2>/dev/null || echo 0)
    if python3 -c "import sys; sys.exit(0 if float('${rtt}') > 0 else 1)" 2>/dev/null; then
      n=$((n + 1))
    fi
  done < <(find "$dir" -maxdepth 1 -name '*.json' ! -name '*.decode_stats.json' -type f 2>/dev/null | sort)
  echo "$n"
}

run_compare() {
  local script="$1" sweep="$2" out="$3"
  log "compare: ${script} -> ${out}"
  pipenv run python3 "$script" --sweep-dir "$sweep" | tee "$out" >> "$LOG"
}

mkdir -p "$PHASE_DIR" "$WEIGHTED_PHASE_DIR"

log "=== finish live gate sweeps ==="

wssm_n=$(valid_rtt_count "${STRATEGIC_WSSM}/results")
cont_n=$(valid_rtt_count "${STRATEGIC_CONT}/results")
weighted_n=$(valid_rtt_count "${WEIGHTED_CONT}/results")

log "strategic wssm: ${wssm_n}/9 valid"
log "strategic contention: ${cont_n}/9 valid"
log "weighted contention: ${weighted_n}/12 valid"

[[ "$wssm_n" -ge 9 ]] || { log "ERROR: strategic wssm incomplete"; exit 1; }
[[ "$cont_n" -ge 9 ]] || { log "ERROR: strategic contention incomplete"; exit 1; }
[[ "$weighted_n" -ge 12 ]] || { log "ERROR: weighted contention incomplete"; exit 1; }

run_compare scripts_cosim/important/compare_wssm_expanded_live_gate.py \
  "$STRATEGIC_WSSM" "${STRATEGIC_WSSM}/compare.txt"
run_compare scripts_cosim/important/compare_contention_v2_live_gate.py \
  "$STRATEGIC_CONT" "${STRATEGIC_CONT}/compare.txt"
run_compare scripts_cosim/important/compare_merged_contention_live_gate.py \
  "$WEIGHTED_CONT" "${WEIGHTED_CONT}/compare.txt"

touch "${PHASE_DIR}/phase_sweep_wssm.done"
touch "${PHASE_DIR}/phase_sweep_contention.done"
touch "${PHASE_DIR}/phase_all.done"
touch "${WEIGHTED_PHASE_DIR}/phase_live_gate.done"
touch "${WEIGHTED_PHASE_DIR}/phase_all.done"

log "phase markers: ${PHASE_DIR}/phase_{sweep_wssm,sweep_contention,all}.done"
log "phase markers: ${WEIGHTED_PHASE_DIR}/phase_{live_gate,all}.done"
log "=== finish complete === log=${LOG}"
