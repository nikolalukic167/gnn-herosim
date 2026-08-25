#!/bin/bash
#SBATCH --job-name=rb8t-probe
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

# route_b_v1 8-task probe (LINEAGES.md, handover plan item 4, 2026-08-25): does pooled
# krank closure (0.790 at 4 tasks, §9c) survive doubling the joint decision to 8 tasks?
# 204 Arm S datasets (2 conn_probs x 2 replica_configs x 3 queue_dists x 17 seeds),
# 16 shards of ~13 datasets each. Projected ~3,800 CPU-h / ~7.4h wall at 16x32c
# (scratchpad costing in the handover plan); --time gives ~1.6x margin per shard since
# per-shard dataset cost is uneven (worst single dataset projected ~45 CPU-h alone).
#
# Reuses run_contention_regen_shard.sh unchanged (generic sharded-generation runner) —
# only the env vars differ. Two are new here and are NOT read by that script directly;
# they pass through because it execs python3 as a plain subprocess:
#   MAX_PLACEMENT_COMBINATIONS_SKIP — raised from the 250,000 default. Confirmed by local
#     smoke test (2026-08-25) that 8-task datasets exceed 250k combinations routinely; the
#     default would SILENTLY SKIP them and bias the corpus toward its easier (less
#     contended) half — exactly the failure mode documented at
#     generate_gnn_datasets_fast.py:387-391 for the route_a sizing ladder.
#   HEROSIM_DATA_LOCALITY / HEROSIM_OUTPUT_SIZE_BYTES — Arm S (primary) physics, matching
#     ROUTE_B_PILOT_V1_GRID's original arm-S env exactly (see its comment block).
#
# ALLOW_NON_UNIQUE is deliberately left at the runner's default (0): route_b's decode mask
# is "no replica reuse" (ROUTE_B_STAGE2_PREREGISTRATION.md §4), and route_b_pilot_v1 is not
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

export MAX_PLACEMENT_COMBINATIONS_SKIP=1000000
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000

bash scripts_cosim/datalab/run_contention_regen_shard.sh
