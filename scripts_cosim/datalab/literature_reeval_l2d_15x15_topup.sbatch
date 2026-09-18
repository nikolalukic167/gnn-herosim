#!/bin/bash
# literature_reeval_v1 Amendment A3b -- L2D 15x15 top-up: seeds 5,6,7 x 4 arms = 12 CPU tasks,
# authors' recipe (lr=2e-5, 10000 updates), to bring 15x15 from 5 to the registered 8 seeds/arm.
#SBATCH --job-name=l2d-reeval-15top
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15top-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15top-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-11

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export L2D_ROOT="${HOME}/L2D"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

ARMS=(gnn mpoff mlp_t1 mlp_t1x)
SEEDS=(5 6 7)
I="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$(( I % 4 ))]}"
SEED="${SEEDS[$(( I / 4 ))]}"
OUT=simulation_data/literature_reeval_v1/l2d/runs
mkdir -p "$OUT"
echo "[TASK] arm=$ARM seed=$SEED size=15x15 (top-up)"
python3 scripts_reeval/l2d/train_arm.py --arm "$ARM" --n_j 15 --n_m 15 --seed "$SEED" \
  --max_updates 10000 --out "$OUT" --threads 2
