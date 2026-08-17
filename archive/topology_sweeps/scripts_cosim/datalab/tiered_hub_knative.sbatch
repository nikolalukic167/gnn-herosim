#!/bin/bash
#SBATCH --job-name=tiered-knat
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tiered-knat-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/tiered-knat-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"

JOBS_TSV="${SWEEP_DIR}/configs/jobs.tsv"
if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing jobs manifest: $JOBS_TSV" >&2
  exit 1
fi

idx=0
target_line=""
while IFS= read -r entry; do
  [[ "$entry" == policy* ]] && continue
  partition=$(echo "$entry" | awk -F'\t' '{print $4}')
  if [[ "$partition" == "cpu" ]]; then
    if [[ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]]; then
      target_line="$entry"
      break
    fi
    idx=$((idx + 1))
  fi
done < "$JOBS_TSV"

if [[ -z "$target_line" ]]; then
  echo "ERROR: no cpu job for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

policy=$(echo "$target_line" | awk -F'\t' '{print $1}')
name=$(echo "$target_line" | awk -F'\t' '{print $2}')
path=$(echo "$target_line" | awk -F'\t' '{print $3}')

bash scripts_cosim/datalab/run_tiered_hub_gnn_mlp_one.sh "$policy" "$name" "$path"
