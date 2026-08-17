#!/bin/bash
#SBATCH --job-name=bipartite-v1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bipartite-v1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bipartite-v1-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-7200}"

JOBS_TSV="${SWEEP_DIR}/configs/jobs_dim22.tsv"
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

bash scripts_cosim/datalab/run_tiered_hub_gnn_mlp_one.sh "$policy" "$name" "$path"
