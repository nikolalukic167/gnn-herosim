#!/bin/bash
#SBATCH --job-name=tiered-knat
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tiered-knat-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/tiered-knat-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-10

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export TIMEOUT="${TIMEOUT:-7200}"

JOBS_TSV="${SWEEP_DIR}/configs/knative_jobs.tsv"
if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing knative manifest: $JOBS_TSV" >&2
  exit 1
fi

idx=0
target_line=""
while IFS= read -r entry; do
  [[ "$entry" == config_name* ]] && continue
  if [[ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]]; then
    target_line="$entry"
    break
  fi
  idx=$((idx + 1))
done < "$JOBS_TSV"

if [[ -z "$target_line" ]]; then
  echo "ERROR: no knative job for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

name=$(echo "$target_line" | awk -F'\t' '{print $1}')
path=$(echo "$target_line" | awk -F'\t' '{print $2}')

bash scripts_cosim/datalab/run_tiered_hub_knative_one.sh "$name" "$path"
