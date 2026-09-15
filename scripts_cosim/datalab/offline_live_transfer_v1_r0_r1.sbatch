#!/bin/bash
# offline_live_transfer_v1 -- R0 (positive control) and R1 (the primary read).
#
# Reads only. Pairs every checkpoint in the three REGISTERED families with its own live
# result; the two burned families are refused by the tool itself.
#
#   sbatch scripts_cosim/datalab/offline_live_transfer_v1_r0_r1.sbatch
#
#SBATCH --job-name=olt-r0r1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r0r1-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r0r1-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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

# The bars must already be in the committed tool, not edited on the way in.
echo "[BARS] $(grep -E '^(R0_CONTROL_MIN_RHO|R1_WITHIN_MIN_ABS_RHO|R1_WITHIN_MIN_FAMILIES|R1_POOLED_Z_MIN_ABS_RHO|R1_ALPHA|MIN_SEEDS_PER_CELL) ' scripts_cosim/offline_live_transfer_v1_pairing_read.py | tr '\n' ' ')"

OUT_DIR=simulation_data/offline_live_transfer_v1
mkdir -p "$OUT_DIR"

${HEROSIM_PY} scripts_cosim/offline_live_transfer_v1_pairing_read.py \
  --gate-root simulation_data/peer_affinity_live_gate/results \
  --wandb-root wandb \
  --out "$OUT_DIR/r0_r1.json"

echo "=== R0/R1 done: $OUT_DIR/r0_r1.json ==="
