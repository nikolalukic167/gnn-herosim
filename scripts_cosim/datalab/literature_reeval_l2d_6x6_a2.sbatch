#!/bin/bash
# literature_reeval_v1 Amendment A2 -- L2D 6x6 extended budget (lr=5e-4, 30000 updates) to check
# validation-tail convergence: 4 arms x 8 seeds = 32 CPU tasks. Moved from local box to datalab
# 2026-09-09 to free the local host; completed seeds already synced from local are skipped.
# See docs/lineages/literature_reeval_v1.md (Amendment A2).
#SBATCH --job-name=l2d-reeval-6a2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-6a2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-6a2-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-31

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
I="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$(( I % 4 ))]}"
SEED=$(( I / 4 ))
OUT=simulation_data/literature_reeval_v1/l2d/runs
NAME="l2d_6x6_${ARM}_lr0.0005_up30000_s${SEED}"
mkdir -p "$OUT"
if [ -f "$OUT/$NAME/final.pth" ]; then
  echo "[TASK] $NAME already done (synced from local), skip"
  exit 0
fi
echo "[TASK] arm=$ARM seed=$SEED size=6x6 lr=5e-4 max_updates=30000"
python3 scripts_reeval/l2d/train_arm.py --arm "$ARM" --n_j 6 --n_m 6 --seed "$SEED" \
  --lr 5e-4 --max_updates 30000 --out "$OUT" --threads 2
