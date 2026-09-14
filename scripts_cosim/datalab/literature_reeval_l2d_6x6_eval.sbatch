#!/bin/bash
# literature_reeval_v1 Amendment A2 -- 6x6 evaluation (authors' recipe + lr-sweep + the new
# up30000 extended-budget checkpoints): best_val + final, test/val/tai splits, PDRs, upstream
# checkpoint. evaluate.py has no skip-if-output-exists logic, so this evaluates every
# l2d_6x6_* run dir present.
#SBATCH --job-name=l2d-reeval-6eval
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-6eval-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-reeval-6eval-%j.err
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

OUT=simulation_data/literature_reeval_v1/l2d/eval_6x6
mkdir -p "$OUT"
python3 scripts_reeval/l2d/evaluate.py --n_j 6 --n_m 6 \
  --runs simulation_data/literature_reeval_v1/l2d/runs --out "$OUT"
