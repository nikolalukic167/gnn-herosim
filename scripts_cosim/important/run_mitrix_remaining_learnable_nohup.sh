#!/usr/bin/env bash
# Run remaining learnable live-gate jobs on mitrix (sequential, skip-if-valid-json).
# Waits for any in-flight executesimulation before each job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TIMEOUT="${TIMEOUT:-18000}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date -Is)] $*"; }

wait_for_idle() {
  while pgrep -f 'src\.executesimulation' >/dev/null 2>&1; do
    log "waiting for in-flight sim..."
    sleep 60
  done
}

P25="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
P35="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
SKEW="simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"

run_cont() {
  local policy="$1" name="$2" path="$3"
  export SWEEP_DIR="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
  export GNN_MODEL="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
  export MLP_MODEL="models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt"
  wait_for_idle
  log "strategic contention ${policy} ${name}"
  bash scripts_cosim/important/run_contention_v2_live_gate_one.sh "$policy" "$name" "$path"
}

run_weighted_mlp() {
  local name="$1" path="$2"
  export SWEEP_DIR="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
  export GNN_MODEL="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
  export MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"
  unset GNN_DECODE_MODE 2>/dev/null || true
  wait_for_idle
  log "weighted contention mlp ${name}"
  bash scripts_cosim/important/run_merged_contention_live_gate_one.sh mlp "$name" "$path"
}

run_weighted_gnn() {
  local name="$1" path="$2" decode="$3" tag="$4"
  export SWEEP_DIR="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
  export GNN_MODEL="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
  export MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"
  export GNN_DECODE_MODE="$decode"
  wait_for_idle
  log "weighted contention gnn ${name} ${tag} (${decode})"
  bash scripts_cosim/important/run_merged_contention_live_gate_one.sh gnn "$name" "$path" "$tag"
}

log "=== mitrix remaining learnable live gates ==="

run_cont mlp sparse_p25 "$P25"
run_cont mlp sparse_p35 "$P35"
run_cont mlp sparse_p25_skew "$SKEW"
run_cont gnn sparse_p25 "$P25"
run_cont gnn sparse_p35 "$P35"
run_cont gnn sparse_p25_skew "$SKEW"

run_weighted_mlp sparse_p25 "$P25"
run_weighted_mlp sparse_p35 "$P35"
run_weighted_mlp sparse_p25_skew "$SKEW"
run_weighted_gnn sparse_p25 "$P25" argmax_uniq gnn_uniq
run_weighted_gnn sparse_p35 "$P35" argmax_uniq gnn_uniq
run_weighted_gnn sparse_p25_skew "$SKEW" argmax_uniq gnn_uniq
run_weighted_gnn sparse_p25 "$P25" argmax gnn_argmax
run_weighted_gnn sparse_p35 "$P35" argmax gnn_argmax
run_weighted_gnn sparse_p25_skew "$SKEW" argmax gnn_argmax

log "=== ALL REMAINING LEARNABLE JOBS DONE ==="
pipenv run python3 scripts_cosim/important/compare_contention_v2_live_gate.py \
  --sweep-dir simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616 || true
pipenv run python3 scripts_cosim/important/compare_merged_contention_live_gate.py \
  --sweep-dir simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500 || true
