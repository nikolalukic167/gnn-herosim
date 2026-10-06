#!/bin/bash
# literature_reeval_v1 -- 10x10 evaluation (authors' recipe + both lr-sweep checkpoints already
# on disk): best_val + final, test/val/tai splits, PDRs, upstream checkpoint. evaluate.py has no
# skip-if-output-exists logic, so this evaluates every l2d_10x10_* run dir present.
#SBATCH --job-name=l2d-reeval-10eval
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-10eval-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-10eval-%j.err
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

OUT=simulation_data/literature_reeval_v1/l2d/eval_10x10
mkdir -p "$OUT"
python3 scripts_reeval/l2d/evaluate.py --n_j 10 --n_m 10 \
  --runs simulation_data/literature_reeval_v1/l2d/runs --out "$OUT"
