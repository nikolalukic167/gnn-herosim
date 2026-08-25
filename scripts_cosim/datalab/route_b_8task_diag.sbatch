#!/bin/bash
#SBATCH --job-name=rb8t-diag
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-diag-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-diag-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-0

# DIAGNOSTIC ONLY — not the real probe. Job 713382 (route_b_8task_probe.sbatch) failed
# 204/204 on "System state capture FAILED" in <1s each, and COSIM_SUPPRESS_SIM_PRINTS=1
# ate the actual traceback (generate_gnn_datasets_fast.py:1198-1200 redirects stdout/
# stderr to a discarded StringIO whenever it's set, or whenever GNN_CAPTURE_DATASET_STATE
# != 1 — see the `suppress_sim_prints` gate at line 1176-1179). This job generates ONE
# dataset (ds_00000) of the exact same grid/env, with that gate defeated and error tracing
# on, so the real exception actually reaches a log file this time.
#
# Writes to a THROWAWAY output subdir, not the real one.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_dag4_route_b_8task_diag
export GRID=route_b_pilot_v1_8task
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=1
export NUM_SHARDS=1
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=0

export MAX_PLACEMENT_COMBINATIONS_SKIP=1000000
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000

# The diagnostic flip, relative to route_b_8task_probe.sbatch:
export GNN_CAPTURE_DATASET_STATE=1
export COSIM_SUPPRESS_SIM_PRINTS=0
export HEROSIM_TRACE_DATASET_ERRORS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
