#!/bin/bash
# literature_reeval_v1 Amendment A3a -- L2D 10x10 learning-rate sweep: 4 arms x lr{1e-4,5e-4} x 8 seeds = 64 CPU tasks.
# Mirrors literature_reeval_l2d_6x6_lrsweep.sbatch. See docs/lineages/literature_reeval_v1.md (Amendment A3).
#SBATCH --job-name=l2d-reeval-10lr
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-10lr-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-10lr-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-63

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
LRS=(1e-4 5e-4)
I="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$(( I % 4 ))]}"
LR="${LRS[$(( (I / 4) % 2 ))]}"
SEED=$(( I / 8 ))
OUT=simulation_data/literature_reeval_v1/l2d/runs
mkdir -p "$OUT"
echo "[TASK] arm=$ARM lr=$LR seed=$SEED size=10x10"
python3 scripts_reeval/l2d/train_arm.py --arm "$ARM" --n_j 10 --n_m 10 --seed "$SEED" --lr "$LR" \
  --max_updates 10000 --out "$OUT" --threads 2
