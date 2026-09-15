#!/bin/bash
#SBATCH --job-name=live-cpu
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/live-cpu-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/live-cpu-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-29

# One live-sim cell per array task. JOBS_TSV is policy\tname\tconfig\tseed.
# No --gres: inference is CPU-bound. Override array/job-name/output via sbatch CLI.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

JOBS_TSV="${JOBS_TSV:?JOBS_TSV required}"
[[ -f "$JOBS_TSV" ]] || { echo "ERROR: missing jobs file: $JOBS_TSV" >&2; exit 1; }

tid="${SLURM_ARRAY_TASK_ID:?}"
line_no=$((tid + 1))
line=$(sed -n "${line_no}p" "$JOBS_TSV" | tr -d '\r')
[[ -n "$line" ]] || { echo "ERROR: no jobs.tsv line ${line_no} (tid=${tid})" >&2; exit 1; }

IFS=$'\t' read -r policy name path seed <<<"$line"
[[ -n "${policy:-}" && -n "${name:-}" && -n "${path:-}" && -n "${seed:-}" ]] || {
  echo "ERROR: malformed jobs.tsv line ${line_no}: ${line}" >&2
  exit 1
}

export SWEEP_DIR="${SWEEP_DIR:?SWEEP_DIR required}"
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
# Declared explicitly rather than inherited: dim7/dim13 change meaning between contracts
# and a wrong value is silent (no clipping anywhere).
if [[ -n "${QUEUE_FEATURE_CONTRACT:-}" ]]; then
  export QUEUE_FEATURE_CONTRACT
fi

echo "=== live-cpu tid=${tid} ${policy} ${name} seed=${seed} ==="
echo "  SWEEP_DIR=${SWEEP_DIR}"
echo "  physics=${HEROSIM_WARMTH_PHYSICS} (strict)"
echo "  queue_feature_contract=${QUEUE_FEATURE_CONTRACT:-<from checkpoint>}"
echo "  config=${path}"

bash scripts_cosim/datalab/run_sealed_holdout_one.sh "$policy" "$name" "$path" "$seed"
