#!/bin/bash
# serving_stability_v1 -- S1 + S2 read.
#
#   sbatch scripts_cosim/datalab/serving_stability_v1_s1_s2_read.sbatch
#
#SBATCH --job-name=ss-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ss-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ss-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
echo "[BARS] $(grep -E '^(S1_MIN_SEEDS|S1_MIN_CELLS|S2_REACTIVE_MAX_RATIO|S2_ARM_MULTIPLE|S2_MIN_CELLS|EARLY_DECILES|MIN_SEEDS_PER_CELL_ARM) ' scripts_cosim/serving_stability_v1_s1_s2_read.py | tr '\n' ' ')"

RES=simulation_data/peer_affinity_live_gate/results/ss_s1s2
n=$(ls "$RES"/*.summary.json 2>/dev/null | wc -l)
echo "[inputs] $n summaries (99 arms declared; gnn seed 3 livelocks on each of 3 cells, so 96 expected)"

${HEROSIM_PY} scripts_cosim/serving_stability_v1_s1_s2_read.py \
  --results "$RES" \
  --out simulation_data/serving_stability_v1/s1_s2.json

echo "=== S1/S2 read done ==="
