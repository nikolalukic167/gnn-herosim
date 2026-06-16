#!/usr/bin/env bash
# Resume learnable live-gate sims after timeout failures.
# Waits for in-flight strategic-merge WSSM, lets that pipeline finish contention,
# then runs weighted-merge contention (9 jobs) at limited parallelism.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/resume_live_gate_sweeps_${TS}.log"
PARALLEL="${PARALLEL:-2}"
export TIMEOUT="${TIMEOUT:-18000}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

STRATEGIC_CONT_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
WEIGHTED_SWEEP="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
STRATEGIC_GNN="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
STRATEGIC_MLP="models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt"
WEIGHTED_GNN="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
WEIGHTED_MLP="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

count_valid_results() {
  local dir="$1"
  local n=0
  [[ -d "$dir" ]] || { echo 0; return; }
  while IFS= read -r f; do
    [[ "$f" == *decode_stats* ]] && continue
    rtt=$(pipenv run python3 -c "import json; d=json.load(open('${f}')); print(d.get('total_rtt', 0))" 2>/dev/null || echo 0)
    if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
      n=$((n + 1))
    fi
  done < <(find "$dir" -maxdepth 1 -name '*.json' -type f 2>/dev/null | sort)
  echo "$n"
}

wait_for_sims() {
  local label="$1"
  log "Waiting for sims to finish (${label})..."
  while pgrep -f 'src\.executesimulation' >/dev/null 2>&1; do
    sleep 60
  done
  log "  idle (${label})"
}

run_job() {
  local cmd="$1"
  log "LAUNCH: ${cmd}"
  if ! eval "$cmd" >> "$LOG" 2>&1; then
    log "FAILED: ${cmd}"
    return 1
  fi
  return 0
}

run_jobs_limited() {
  local -a jobs=("$@")
  local failed=0
  local i=0
  local total=${#jobs[@]}
  while (( i < total )); do
    local batch=0
    while (( batch < PARALLEL && i < total )); do
      while (( $(pgrep -cf 'src\.executesimulation' 2>/dev/null || echo 0) >= PARALLEL )); do
        sleep 30
      done
      run_job "${jobs[$i]}" &
      i=$((i + 1))
      batch=$((batch + 1))
    done
    wait || failed=$((failed + 1))
  done
  return "$failed"
}

verify_sweep() {
  local dir="$1" expected="$2" label="$3"
  local n
  n=$(count_valid_results "$dir")
  if [[ "$n" -lt "$expected" ]]; then
    log "ERROR: ${label} incomplete (${n}/${expected} valid JSONs with total_rtt>0)"
    find "$dir" -maxdepth 1 -name '*.json' -type f 2>/dev/null | while read -r f; do
      rtt=$(pipenv run python3 -c "import json; d=json.load(open('${f}')); print(d.get('total_rtt', '?'))" 2>/dev/null || echo ERR)
      log "  $(basename "$f"): total_rtt=${rtt}"
    done
    return 1
  fi
  log "OK: ${label} ${n}/${expected} valid results"
  return 0
}

resume_strategic_contention() {
  local out="${STRATEGIC_CONT_SWEEP}/results"
  mkdir -p "$out"
  local n
  n=$(count_valid_results "$out")
  if [[ "$n" -ge 9 ]]; then
    log "Strategic contention already complete (${n}/9)"
    return 0
  fi

  log "=== Resume strategic-merge contention (${n}/9) ==="
  export SWEEP_DIR="$STRATEGIC_CONT_SWEEP"
  export GNN_MODEL="$STRATEGIC_GNN"
  export MLP_MODEL="$STRATEGIC_MLP"

  local -a jobs=()
  for policy in mlp gnn; do
    for entry in "${CONFIGS[@]}"; do
      name="${entry%%|*}"; path="${entry#*|}"
      jobs+=("bash scripts_cosim/important/run_contention_v2_live_gate_one.sh ${policy} ${name} ${path}")
    done
  done
  run_jobs_limited "${jobs[@]}" || true
  verify_sweep "$out" 9 "strategic contention"
}

resume_weighted_contention() {
  local out="${WEIGHTED_SWEEP}/results"
  mkdir -p "$out"
  local n
  n=$(count_valid_results "$out")
  if [[ "$n" -ge 12 ]]; then
    log "Weighted contention already complete (${n}/12)"
    return 0
  fi

  log "=== Resume weighted-merge contention (${n}/12) ==="
  export SWEEP_DIR="$WEIGHTED_SWEEP"
  export GNN_MODEL="$WEIGHTED_GNN"
  export MLP_MODEL="$WEIGHTED_MLP"

  local -a jobs=()
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    jobs+=("export GNN_DECODE_MODE=; bash scripts_cosim/important/run_merged_contention_live_gate_one.sh mlp ${name} ${path}")
  done
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    jobs+=("export GNN_DECODE_MODE=argmax_uniq; bash scripts_cosim/important/run_merged_contention_live_gate_one.sh gnn ${name} ${path} gnn_uniq")
  done
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    jobs+=("export GNN_DECODE_MODE=argmax; bash scripts_cosim/important/run_merged_contention_live_gate_one.sh gnn ${name} ${path} gnn_argmax")
  done
  run_jobs_limited "${jobs[@]}" || true
  verify_sweep "$out" 12 "weighted contention"
}

log "=== resume live gate sweeps (PARALLEL=${PARALLEL} TIMEOUT=${TIMEOUT}) ==="

# Let strategic-merge pipeline finish hub_k8 WSSM + contention first.
wait_for_sims "pre-strategic"

# Poll until strategic pipeline completes contention or we need to intervene.
deadline=$(( $(date +%s) + 86400 ))
while (( $(date +%s) < deadline )); do
  n=$(count_valid_results "${STRATEGIC_CONT_SWEEP}/results")
  if [[ -f logs/strategic_merge_pipeline/phase_sweep_contention.done && "$n" -ge 9 ]]; then
    log "Strategic pipeline contention phase done (${n}/9)"
    break
  fi
  if [[ "$n" -ge 9 ]]; then
    log "Strategic contention complete (${n}/9)"
    break
  fi
  if pgrep -f 'run_strategic_merge_train_sweep' >/dev/null 2>&1; then
    log "Strategic pipeline still running (contention ${n}/9)..."
    sleep 120
    continue
  fi
  log "Strategic pipeline not running; resuming contention (${n}/9)"
  resume_strategic_contention && break
  sleep 60
done

wait_for_sims "pre-weighted"
resume_weighted_contention

# Compare scripts
if verify_sweep "${STRATEGIC_CONT_SWEEP}/results" 9 "strategic contention (final)"; then
  pipenv run python3 scripts_cosim/important/compare_contention_v2_live_gate.py \
    --sweep-dir "$STRATEGIC_CONT_SWEEP" >> "$LOG" 2>&1 || log "ERROR: strategic compare failed"
fi

WSSM_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
if verify_sweep "${WSSM_SWEEP}/results" 9 "strategic wssm (final)"; then
  pipenv run python3 scripts_cosim/important/compare_wssm_expanded_live_gate.py \
    --sweep-dir "$WSSM_SWEEP" >> "$LOG" 2>&1 || log "ERROR: wssm compare failed"
fi

if verify_sweep "${WEIGHTED_SWEEP}/results" 12 "weighted contention (final)"; then
  pipenv run python3 scripts_cosim/important/compare_merged_contention_live_gate.py \
    --sweep-dir "$WEIGHTED_SWEEP" >> "$LOG" 2>&1 || log "ERROR: weighted compare failed"
fi

log "=== resume orchestrator complete === log=${LOG}"
