#!/bin/bash
#SBATCH --job-name=tier1-decode
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tier1-decode-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/tier1-decode-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-26

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc" 2>/dev/null || true
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn

JOBS_FILE="${JOBS_FILE:-scripts_cosim/datalab/tier1_decode_jobs.tsv}"
LINE=$((SLURM_ARRAY_TASK_ID + 1))
IFS=',' read -r CONFIG_NAME PROFILE < <(sed -n "${LINE}p" "$JOBS_FILE" | tr -d '\r')
CONFIG_NAME="${CONFIG_NAME// /}"
PROFILE="${PROFILE// /}"
if [[ -z "$CONFIG_NAME" || -z "$PROFILE" ]]; then
  echo "ERROR: invalid jobs line ${LINE}: config=${CONFIG_NAME} profile=${PROFILE}" >&2
  exit 1
fi

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export TIMEOUT="${TIMEOUT:-7200}"

bash scripts_cosim/run_bipartite_tier1_decode_one.sh "$CONFIG_NAME" "$PROFILE"
