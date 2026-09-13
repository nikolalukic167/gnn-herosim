#!/bin/bash
#SBATCH --job-name=rb8t-score
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-score-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-score-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# route_b_v1 item 4, Step 4 — score the 8-task Arm S corpus (job 713673, 204/204
# complete, 0 skips / 0 errors / all sweep_complete=true; cross-venue identity of
# ds_00002 vs the local smoke re-verified byte-identical on best/workload and as a
# sorted set on placements.jsonl).
#
# ALPHA CORRESPONDENCE — registered a priori, not searched. cap_node = alpha *
# max_single_demand has no task-count term, so 8 tasks against the same cap is ~2x
# tighter; the equal-tightness match to the 4-task ladder (2.0, 3.0) is the DOUBLE:
# primary 4.0 (= 2 * TIGHT_ALPHA), ladder 3.0,4.0,5.0,6.0 covering both doubled rungs
# plus the response curve. This is a reasoned deviation, recorded here and in
# LINEAGES.md §9c/§9d — do not "fix" the alphas back to the 4-task values.
#
# --include-per-dataset is MANDATORY: route_b_coefficient_transfer.py reads
# report["per_dataset"][str(alpha)] (its line 105) and raises without it.
#
# greedy_stuck and no_feasible_rows must be reported explicitly when this is read —
# silent-bias risks at 8 tasks that did not bite at 4.
#
# Single-process CPU job; runs on a compute node because it streams ~23 GB of
# placements.jsonl (login-node watchdog kills exactly this kind of work).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

python3 -u scripts_cosim/score_route_b_contention.py \
  --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_8task \
  --objective rtt --alphas 3.0,4.0,5.0,6.0 --include-per-dataset \
  --out simulation_data/route_b_8task_rtt.json
