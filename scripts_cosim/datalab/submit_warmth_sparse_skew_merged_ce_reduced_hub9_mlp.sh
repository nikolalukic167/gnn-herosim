#!/usr/bin/env bash
# Submit 9-run skew-merged ce-reduced hub9 MLP-only sweep (FORCE_RERUN=1).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_skew_merged_ce_reduced_hub9_20260612}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-skew-merged-ce-reduced.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt}"
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export TIMEOUT="${TIMEOUT:-7200}"
export FORCE_RERUN=1
export JOBS_TSV="${JOBS_TSV:-${SWEEP_DIR}/configs/jobs_smcr_hub9_mlp.tsv}"

if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing ${JOBS_TSV}" >&2
  exit 1
fi

job_count=$(($(wc -l < "$JOBS_TSV") - 1))
last_idx=$((job_count - 1))
if [[ "$job_count" -ne 9 ]]; then
  echo "ERROR: expected 9 MLP jobs, got ${job_count}" >&2
  exit 1
fi

for f in "$MLP_MODEL" "$WORKLOAD" \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_hub9_gpu.sbatch \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_hub9_one.sh; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

mkdir -p logs "${SWEEP_DIR}/results"

pick_partition() {
  local part="$1"
  local gres="$2"
  local tag="$3"
  echo "--- test-only ${part} (${gres}) ---"
  if out=$(sbatch --test-only \
    --partition="$part" \
    --gres="$gres" \
    --array="0-${last_idx}" \
    --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS,GNN_MODEL,MLP_MODEL,GNN_BATCH_SIZE,GNN_BATCH_TIMEOUT,TIMEOUT,FORCE_RERUN,JOBS_TSV \
    scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_hub9_gpu.sbatch 2>&1); then
    echo "$out"
    if echo "$out" | grep -qiE 'PENDING|ReqNodeNotAvail|QOSMax|AssocMaxJobsLimit|Resources'; then
      return 1
    fi
    SELECTED_PARTITION="$part"
    SELECTED_GRES="$gres"
    SELECTED_TAG="$tag"
    return 0
  fi
  echo "$out" >&2
  return 1
}

SELECTED_PARTITION=""
SELECTED_GRES=""
SELECTED_TAG=""

echo "SWEEP_DIR=${SWEEP_DIR}"
echo "MLP_MODEL=${MLP_MODEL}"
echo "Jobs: ${job_count} (MLP only, FORCE_RERUN=1) · TIMEOUT=${TIMEOUT}s"

if ! pick_partition "GPU-a40" "gpu:a40:1" "a40"; then
  echo "GPU-a40 not immediately schedulable; trying GPU-l40s..."
  if ! pick_partition "GPU-l40s" "gpu:l40s:1" "l40s"; then
    echo "ERROR: neither GPU-a40 nor GPU-l40s can schedule ${job_count} jobs now" >&2
    exit 1
  fi
fi

gpu_job=$(sbatch \
  --job-name="wsmssk-hub9-mlp-${SELECTED_TAG}" \
  --output="/home/nikola.lukic/gnn-herosim/logs/wsmssk-hub9-mlp-${SELECTED_TAG}-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/wsmssk-hub9-mlp-${SELECTED_TAG}-%A_%a.err" \
  --partition="$SELECTED_PARTITION" \
  --gres="$SELECTED_GRES" \
  --array="0-${last_idx}" \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS,GNN_MODEL,MLP_MODEL,GNN_BATCH_SIZE,GNN_BATCH_TIMEOUT,TIMEOUT,FORCE_RERUN,JOBS_TSV \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_hub9_gpu.sbatch | awk '{print $NF}')

echo ""
echo "Submitted MLP array: ${gpu_job} (tasks 0-${last_idx} on ${SELECTED_PARTITION})"
echo "JOB_ID=${gpu_job}"
echo "PARTITION=${SELECTED_PARTITION}"
