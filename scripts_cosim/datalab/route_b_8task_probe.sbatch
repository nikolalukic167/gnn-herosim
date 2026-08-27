#!/bin/bash
#SBATCH --job-name=rb8t-probe
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

# route_b_v1 8-task probe (LINEAGES.md, handover plan item 4, 2026-08-25): does pooled
# krank closure (0.790 at 4 tasks, §9c) survive doubling the joint decision to 8 tasks?
# 204 Arm S datasets (2 conn_probs x 2 replica_configs x 3 queue_dists x 17 seeds),
# 16 shards of ~13 datasets each.
#
# COST: the original ~3,800 CPU-h projection in this header was wrong by ~20x. It came
# from the unreachability-free upper bound on placement combinations (~580k/dataset);
# reachability filtering cuts real sweeps to 1,440-57,600 combinations. Measured 8-task
# rate: ~2,935 sims/min on 8 cores WITH print I/O on (job 713559). 204 datasets is a few
# CPU-hours, not thousands — cheap enough that the corpus is generated LOCALLY (matching
# the 4-task baseline's venue, LINEAGES.md:4021) and this sbatch is the fallback, not the
# default route. The 12h walltime is left as generous margin.
#
# Reuses run_contention_regen_shard.sh unchanged (generic sharded-generation runner) —
# only the env vars differ. Two are new here and are NOT read by that script directly;
# they pass through because it execs python3 as a plain subprocess:
#   MAX_PLACEMENT_COMBINATIONS_SKIP — raised from the 250,000 default. 8-task datasets
#     exceed 250k routinely; the default would SILENTLY SKIP them and bias the corpus
#     toward its easier (less contended) half — exactly the failure mode documented at
#     generate_gnn_datasets_fast.py:387-391 for the route_a sizing ladder.
#     2,000,000, and the bound is derived rather than guessed. THE THRESHOLD IS TESTED
#     AGAINST total_possible, WHICH IS COMPUTED BEFORE THE UNIQUENESS CONSTRAINT
#     (executecosimulation.py:1808-1817) — not against the post-uniqueness sweep size.
#     For 2 DAG instances total_possible = n4(client_0) * n4(client_1), where n4(c) is the
#     4-task combination count from client node c: the two instances are emitted from
#     independently drawn client nodes, and per-task feasible_platforms is filtered by
#     reachability FROM THAT CLIENT, so instance 1 sees a different candidate set even
#     though the infrastructure is byte-identical to the 4-task dataset at the same index
#     (verified). Since feasible_t <= n_t, the hard upper bound is (max_ds prod_t n_t)^2 =
#     1248^2 = 1,557,504, so 2,000,000 cannot be reached by any dataset in this grid.
#     1,000,000 was NOT enough: job 713656 skipped ds_00026 (n4=1152) and was cancelled.
#     Post-uniqueness sweep sizes are much smaller — measured ds_00002 161,280 /
#     ds_00000 175,104 / ds_00092 193,500 / ds_00189 483,840 — so raising the threshold
#     costs little wall time; it only stops the largest-candidate-space datasets from
#     being dropped, which is the biased half.
#   HEROSIM_DATA_LOCALITY / HEROSIM_OUTPUT_SIZE_BYTES — Arm S (primary) physics, matching
#     ROUTE_B_PILOT_V1_GRID's original arm-S env exactly (see its comment block).
#
# ALLOW_NON_UNIQUE is deliberately left at the runner's default (0): route_b's decode mask
# is "no replica reuse" (docs/lineages/route_b_v1/stage2-preregistration.md §4), and route_b_pilot_v1 is not
# among the grids marked "MUST --allow-non-unique-replicas" in this file's grid comments.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_dag4_route_b_pilot_v1_8task
export GRID=route_b_pilot_v1_8task
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=204
export NUM_SHARDS=16
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=0
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000

# These two were MISSING in the version that ran as job 713615, and that is why it was
# cancelled: 8 of the first 16 datasets came back with sweep_complete=false (up to 45,387
# failed placements of 57,600), every failure an rf task forced onto a replica the episode
# had already reclaimed. They are not optional — they are half of the Arm S generation
# recipe recorded at LINEAGES.md:4022-4023, and the 4-task baseline this probe is compared
# against was built with them. See cosim_keep_alive()'s docstring
# (src/executecosimulation.py:86-109): a DAG stage stretched past KEEP_ALIVE=30s by route
# B's 800 MB transfers gets the CHILD's forced replica scaled down before the child
# dispatches, killing the episode with "Invalid forced placement ... not in replicas".
# 4-task episodes are short enough to mostly survive it; 8-task ones are not.
# HEROSIM_RETAIN_TASK_TIMES is what makes rows carry task_times, without which
# score_route_b_contention.py refuses the makespan objective (its lines 103-107).
export HEROSIM_COSIM_KEEP_ALIVE=1000000
export HEROSIM_RETAIN_TASK_TIMES=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
