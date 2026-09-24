#!/bin/bash
#SBATCH --job-name=rb8t-transfer
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-transfer-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-transfer-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# route_b_v1 item 4, Step 5 — pooled krank coefficient transfer on the 8-task corpus.
# Submit with --dependency=afterok:<rb8t-score jobid>: the transfer re-verifies each
# cell's R_exact against the frozen report and refuses to fit on disagreement (its line
# ~280), so the report must exist and be final before this starts.
#
# --alpha 4.0 is the a-priori equal-tightness double of the 4-task TIGHT_ALPHA=2.0
# (cap_node has no task-count term; see route_b_8task_score.sbatch header). Budgeted
# 8-13 h single-process; 24 h walltime is margin, not an expectation.
#
# READ-OUT FRAMING (must travel with the numbers into LINEAGES.md §9c/§9d):
# krank_pooled_exploratory is flagged NOT REGISTERED, NO VERDICT READ FROM IT in both
# the code and §9c. The 8-task result inherits that — evidence about the layout
# hypothesis, not a gate. Report median_fraction, median_mean_tied, AND the firing rate
# (4-task baseline: 35/204 = 17.2%, pooled statistic resting on 20 of them; a
# materially different 8-task rate changes the power and must be said out loud).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

python3 -u scripts_cosim/route_b_coefficient_transfer.py \
  --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_8task \
  --report simulation_data/route_b_8task_rtt.json --alpha 4.0 \
  --out simulation_data/route_b_8task_coefficient_transfer.json
