#!/bin/bash
# objective_pivot_v1 Phase 1 — precondition: trainer determinism, green in the PINNED tree.
#
# The registration (docs/lineages/objective_pivot_v1/phase1-registration-draft.md, signed
# off 2026-08-28) requires tests/test_trainer_determinism.py green on the GNN trainer
# BEFORE seed 9 is queued — and it must be green in the tree the draws will actually be
# trained from, which is the pinned worktree at c08aa7e, not the main checkout at HEAD.
# The train array is submitted with --dependency=afterok on this job, so a red test here
# stops the whole chain instead of producing eight unreproducible draws.
#
# CPU-only and ~12s locally; it goes through sbatch anyway (datalab-pitfalls #2: nothing
# resource-shaped runs on the login node).
#SBATCH --job-name=objp1-determinism
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/objp1-determinism-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/objp1-determinism-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-c08aa7e
PIN_COMMIT=c08aa7ee140fd51e3d384f97df3f31b126df96ab
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1

# The pin is the registration's binding condition; assert it before running anything.
HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
python3 -m pytest tests/test_trainer_determinism.py -q
echo "=== determinism GREEN in pinned tree ==="
