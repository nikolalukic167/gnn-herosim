#!/bin/bash
# literature_reeval_v1 -- full 15x15 re-evaluation (8 seeds/arm, best_val + final, test/val/tai)
# after the topup training brought mpoff/mlp_t1/mlp_t1x to 8 seeds. evaluate.py has no
# skip-if-output-exists logic, so this redoes all 32 runs + PDR baselines + upstream checkpoint.
# Moved from local box to datalab 2026-09-09 to free the local host.
#SBATCH --job-name=l2d-reeval-15eval
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15eval-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-15eval-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export L2D_ROOT="${HOME}/L2D"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

OUT=simulation_data/literature_reeval_v1/l2d/eval_15x15
mkdir -p "$OUT"
python3 scripts_reeval/l2d/evaluate.py --n_j 15 --n_m 15 \
  --runs simulation_data/literature_reeval_v1/l2d/runs --out "$OUT"
