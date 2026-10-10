#!/bin/bash
# accel_replica_v1 W1: CD with and without the rule on the calibrated heavy cells (scripts_cosim/accel_w1.py).
#   sbatch --export=ALL,WT=<worktree>,ROOT=<calib heavy_root>,TAG=m27p6227,OUT=<new dir> accel_w1.sbatch
#SBATCH --job-name=accel-w1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-w1-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; ROOT=${ROOT:?}; TAG=${TAG:?}; OUT=${OUT:?}
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 HEROSIM_RAW_DIR=/tmp/accel_w1_$SLURM_JOB_ID; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
echo "HEAD $(git rev-parse HEAD) tag $TAG node $(hostname) $(date -Is)"
python3 scripts_cosim/accel_w1.py --root "$ROOT" --tag "$TAG" --out "$OUT" --wt "$WT" --seeds 9905 9906 9907 9908 --windows g0 g1
