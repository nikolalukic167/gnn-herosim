#!/bin/bash
#SBATCH --job-name=wsmcr-hub9-abl
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wsmcr-hub9-abl-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wsmcr-hub9-abl-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_hub9_20260611}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export MLP_CE_REDUCED="${MLP_CE_REDUCED:-models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt}"
export MLP_DIM22="${MLP_DIM22:-models/tabular/batch_edge_mlp.pt}"
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export TIMEOUT="${TIMEOUT:-7200}"

JOBS_TSV="${SWEEP_DIR}/configs/jobs_smcr_hub9_ablation.tsv"
if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing jobs manifest: $JOBS_TSV" >&2
  exit 1
fi

idx=0
target_line=""
while IFS= read -r entry; do
  [[ "$entry" == policy* ]] && continue
  if [[ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]]; then
    target_line="$entry"
    break
  fi
  idx=$((idx + 1))
done < "$JOBS_TSV"

if [[ -z "$target_line" ]]; then
  echo "ERROR: no job for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

policy=$(echo "$target_line" | awk -F'\t' '{print $1}')
name=$(echo "$target_line" | awk -F'\t' '{print $2}')
path=$(echo "$target_line" | awk -F'\t' '{print $3}')

bash scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_hub9_ablation_one.sh "$policy" "$name" "$path"
