#!/usr/bin/env bash
# Submit sealed multi-seed live holdout on datalab (GNN array by default).
# Run FROM datalab login node, or via: bash transfer_and_submit_sealed_holdout_datalab.sh
#
# Usage:
#   bash scripts_cosim/datalab/submit_sealed_holdout.sh
#   POLICY=gnn ARRAY=0-19 bash scripts_cosim/datalab/submit_sealed_holdout.sh
#   PARTITION=GPU-h100 GRES=gpu:h100:1 bash scripts_cosim/datalab/submit_sealed_holdout.sh
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export TIMEOUT="${TIMEOUT:-18000}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"
export POLICY="${POLICY:-gnn}"
ARRAY_SPEC="${ARRAY:-0-19}"
SLURM_TIME="${SLURM_TIME:-06:00:00}"
JOB_NAME="${JOB_NAME:-cv2-sealed}"

mkdir -p logs "$SWEEP_DIR/results"

[[ -f "$GNN_MODEL" ]] || { echo "ERROR: missing $GNN_MODEL — rsync from mitrix first" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "ERROR: missing $WORKLOAD" >&2; exit 1; }

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

scancel -u "${USER}" --name="$JOB_NAME" 2>/dev/null || true
pick_earliest_partition

job=$(sbatch \
  --job-name="$JOB_NAME" \
  --output="/home/nikola.lukic/gnn-herosim/logs/${JOB_NAME}-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/${JOB_NAME}-%A_%a.err" \
  --partition="$SELECTED_PARTITION" \
  --gres="$SELECTED_GRES" \
  --mem=32G \
  --cpus-per-task=4 \
  --time="$SLURM_TIME" \
  --array="$ARRAY_SPEC" \
  --export=ALL,POLICY,SWEEP_DIR,GNN_MODEL,MLP_MODEL,WORKLOAD,TIMEOUT \
  scripts_cosim/datalab/sealed_holdout_gpu.sbatch | awk '{print $NF}')

echo "Submitted ${JOB_NAME} job=${job} array=${ARRAY_SPEC} policy=${POLICY}"
echo "PARTITION=${SELECTED_PARTITION} GRES=${SELECTED_GRES} TIME=${SLURM_TIME}"
echo "SWEEP_DIR=${SWEEP_DIR}"
echo "Monitor: squeue -u \$USER -n ${JOB_NAME}"
