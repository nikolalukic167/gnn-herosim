#!/bin/bash
#SBATCH --job-name=rb8t-krank-verify
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-krank-verify-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-krank-verify-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# PP0' (ROUTE_B_STAGE2_PREREGISTRATION.md §10) — step 2 of 2 on the 8-task corpus:
# independent recomputation of the per-dataset and pooled krank repair fractions,
# tie bands included, by verify_route_b_scorer_agreement.py --check-krank (pure
# stdlib, own MGS QR, pooled intercepts projected out by within-dataset demeaning —
# a deliberately different algorithm from the transfer's numpy lstsq).
# 1e-9 agreement on every (dataset, arm) fraction or the lineage is VOID until
# resolved. This is a verification gate, NOT a kill test: no threshold reading is
# taken from the 8-task numbers (§10 PP0' row; user decision 2026-08-26 — the
# 8-task corpus plays no gate role).
#
# Submit with --dependency=afterok:<rb8t-transfer-pp0p job id>.
#
# Budget: pure-Python pass over 6.87M sweep rows (33 firing datasets), two LS arms;
# estimated 2-6 h single-process. 24 h walltime is margin, not an expectation.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

python3 -u scripts_cosim/verify_route_b_scorer_agreement.py \
  --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_8task \
  --check-krank simulation_data/route_b_8task_coefficient_transfer_pp0prime.json
