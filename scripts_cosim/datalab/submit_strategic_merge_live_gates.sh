#!/usr/bin/env bash
# Submit strategic-merge + weighted-merge live gate SLURM arrays on datalab.
# Auto-picks the GPU partition with the earliest schedulable start time.
#
# Usage:
#   bash scripts_cosim/datalab/submit_strategic_merge_live_gates.sh
#   PARTITION=GPU-h100 GRES=gpu:h100:1 bash scripts_cosim/datalab/submit_strategic_merge_live_gates.sh
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export TIMEOUT="${TIMEOUT:-18000}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
SLURM_TIME="${SLURM_TIME:-05:00:00}"

cancel_prior_live_gate_jobs() {
  echo "Cancelling prior pending live-gate arrays (if any)..."
  scancel -u "${USER}" --name=sm-wssm-gate 2>/dev/null || true
  scancel -u "${USER}" --name=sm-cont-gate 2>/dev/null || true
  scancel -u "${USER}" --name=wt-cont-gate 2>/dev/null || true
}

pick_earliest_partition() {
  if [[ -n "${PARTITION:-}" && -n "${GRES:-}" ]]; then
    SELECTED_PARTITION="$PARTITION"
    SELECTED_GRES="$GRES"
    echo "Using forced PARTITION=${SELECTED_PARTITION} GRES=${SELECTED_GRES}"
    return 0
  fi

  local best_part="" best_gres="" best_start="" best_epoch=9223372036854775807
  local candidates=(
    "GPU-a100s gpu:a100s:1"
    "GPU-a40 gpu:a40:1"
    "GPU-l40s gpu:l40s:1"
    "GPU-a100 gpu:a100:1"
  )

  for entry in "${candidates[@]}"; do
    local part="${entry%% *}"
    local gres="${entry#* }"
    echo "--- test-only ${part} (${gres}) ---"
    local out
    if ! out=$(sbatch --test-only \
      --partition="$part" \
      --gres="$gres" \
      --mem=32G \
      --cpus-per-task=4 \
      --time="$SLURM_TIME" \
      --wrap=/bin/true 2>&1); then
      echo "$out"
      continue
    fi
    echo "$out"
    local start
    start=$(echo "$out" | sed -n 's/.*to start at \([^ ]*\).*/\1/p')
    [[ -z "$start" ]] && continue
    local epoch
    epoch=$(date -d "$start" +%s 2>/dev/null || echo 9223372036854775807)
    if (( epoch < best_epoch )); then
      best_epoch=$epoch
      best_part="$part"
      best_gres="$gres"
      best_start="$start"
    fi
  done

  if [[ -z "$best_part" ]]; then
    echo "ERROR: no GPU partition could schedule a smoke job" >&2
    exit 1
  fi

  SELECTED_PARTITION="$best_part"
  SELECTED_GRES="$best_gres"
  echo "Selected ${SELECTED_PARTITION} (${SELECTED_GRES}) · earliest start ${best_start}"
}

submit_array() {
  local job_name="$1"
  local array_spec="$2"
  local sbatch_file="$3"
  sbatch \
    --job-name="$job_name" \
    --output="/home/nikola.lukic/gnn-herosim/logs/${job_name}-%A_%a.out" \
    --error="/home/nikola.lukic/gnn-herosim/logs/${job_name}-%A_%a.err" \
    --partition="$SELECTED_PARTITION" \
    --gres="$SELECTED_GRES" \
    --mem=32G \
    --cpus-per-task=4 \
    --time="$SLURM_TIME" \
    --array="$array_spec" \
    --export=ALL \
    "$sbatch_file" | awk '{print $NF}'
}

cancel_prior_live_gate_jobs
pick_earliest_partition

export SWEEP_DIR="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
wssm_job=$(submit_array "sm-wssm-gate" "8" \
  scripts_cosim/datalab/strategic_merge_wss_live_gate_gpu.sbatch)
echo "WSSM gate:        ${wssm_job} (array 8 only — hub_k8 GNN)"

export SWEEP_DIR="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
export GNN_MODEL="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
export MLP_MODEL="models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt"
cont_job=$(submit_array "sm-cont-gate" "3-8" \
  scripts_cosim/datalab/strategic_merge_contention_live_gate_gpu.sbatch)
echo "Strategic cont:   ${cont_job} (array 3-8 — mlp+gnn only)"

export SWEEP_DIR="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
export GNN_MODEL="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
export MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"
wt_job=$(submit_array "wt-cont-gate" "0-8" \
  scripts_cosim/datalab/merged_contention_weighted_live_gate_gpu.sbatch)
echo "Weighted cont:    ${wt_job} (array 0-8 — learnable only)"

echo ""
echo "PARTITION=${SELECTED_PARTITION} GRES=${SELECTED_GRES} TIME=${SLURM_TIME}"
echo "Monitor: squeue -u ${USER}"
echo "Predicted start: squeue --start -u ${USER}"
