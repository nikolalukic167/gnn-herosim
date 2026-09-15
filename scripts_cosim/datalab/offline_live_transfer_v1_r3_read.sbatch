#!/bin/bash
# offline_live_transfer_v1 -- R3 step 2: decile the best and worst seed's queue time.
#
#   sbatch scripts_cosim/datalab/offline_live_transfer_v1_r3_read.sbatch
#
#SBATCH --job-name=olt-r3read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r3read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r3read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
echo "[BARS] $(grep -E '^(R3_DECILES|R3_START_MIN_FRAC|R3_COMPOUND_MAX_FRAC|R3_MIN_TASKS_PER_DECILE|R3_MIN_FULL_GAP_S) ' scripts_cosim/offline_live_transfer_v1_decile_read.py | tr '\n' ' ')"

CAP=simulation_data/peer_affinity_live_gate/results/olt_r3_capture
BEST="$CAP/best_gnn_s14.json"
WORST="$CAP/worst_gnn_s8.json"
for f in "$BEST" "$WORST"; do
  [[ -f "$f" ]] || { echo "FAIL LOUD: capture missing: $f" >&2; exit 1; }
done

${HEROSIM_PY} scripts_cosim/offline_live_transfer_v1_decile_read.py \
  --best "$BEST" --worst "$WORST" \
  --out simulation_data/offline_live_transfer_v1/r3_deciles.json

echo "=== R3 read done ==="
