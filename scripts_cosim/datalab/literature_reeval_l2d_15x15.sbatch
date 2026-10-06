#!/bin/bash
# literature_reeval_v1 -- L2D 15x15, 4 arms x 5 seeds = 20 CPU tasks (registered in
# docs/lineages/literature_reeval_v1.md; arms in scripts_reeval/l2d/arm_features.py).
# Upstream L2D is cloned at ~/L2D (commit 7b2efbb); nothing there is modified.
# Local timing: ~1.3 s/update at 15x15 on 2 threads -> ~4 h for 10 000 updates + validation.
#SBATCH --job-name=l2d-reeval-15
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-19

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export L2D_ROOT="${HOME}/L2D"

# Prove which interpreter actually serves (datalab-pitfalls #8).
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

ARMS=(gnn mpoff mlp_t1 mlp_t1x)
I="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$(( I % 4 ))]}"
SEED=$(( I / 4 ))
OUT=simulation_data/literature_reeval_v1/l2d/runs
mkdir -p "$OUT"
echo "[TASK] arm=$ARM seed=$SEED"
python3 scripts_reeval/l2d/train_arm.py --arm "$ARM" --n_j 15 --n_m 15 --seed "$SEED" \
  --max_updates 10000 --out "$OUT" --threads 2
