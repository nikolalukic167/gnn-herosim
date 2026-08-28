#!/bin/bash
#SBATCH --job-name=rbh3-ctrl
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-ctrl-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-ctrl-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

# route_b_env_pivot_v1 rung H3, PAIRED SEPARABLE CONTROL (B0-analog) — GENERATION ONLY.
#
# The S0 gate's control arm for route_b_pivot_h3_arm_s.sbatch. Config-identical to Arm S —
# same grid, same registered fresh seed block 3301-3317, same 204 shape — differing ONLY in
# the physics env block below. S0 is a VOID gate: this corpus must reach
# r_exact.frac_gt_1pct <= 0.02 before any S1-S4 number is read on Arm S. THIS JOB GENERATES;
# it reads no bar.
#
# Control definition is AMENDMENT 1's (screen-amendment-1.md §4), which is what took the H0
# control's mean_tied to exactly 0:
#   HEROSIM_DATA_LOCALITY   unset  (Arm S: 1)
#   HEROSIM_OUTPUT_SIZE_BYTES unset (Arm S: 800000000)
#   HEROSIM_STORAGE_NEUTRAL   =1    (Arm S: unset — control-arm-only lever)
# The remaining two of the five-variable recipe are set on BOTH arms.
#
# Reuses run_contention_regen_shard.sh unchanged (`micromamba shell hook --shell bash`,
# plain `python3`, never `pipenv run` — datalab-pitfalls #1 and #8).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_route_b_pivot_h3_ctrl
export GRID=route_b_pivot_h3
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=204
export NUM_SHARDS=16
export ONLY_MISSING_JSONL=0
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

# Must match Arm S: the control is config-identical by construction, and the runner's
# default is 1. See the Arm S script for why uniqueness is load-bearing on this rung.
export ALLOW_NON_UNIQUE=0

# Same bound as Arm S — the pre-uniqueness product does not depend on the physics arm.
export MAX_PLACEMENT_COMBINATIONS_SKIP=30000000000

# CONTROL physics. HEROSIM_DATA_LOCALITY and HEROSIM_OUTPUT_SIZE_BYTES are deliberately
# absent; env -u is not needed because sbatch does not inherit them here, but the runner is
# also invoked with `set -u` and reads neither.
export HEROSIM_STORAGE_NEUTRAL=1

# Set on both arms — see the Arm S script's note. Omitting these is the VOID-GENERATION
# failure that still reports 204/204 SUCCESS.
export HEROSIM_COSIM_KEEP_ALIVE=1000000
export HEROSIM_RETAIN_TASK_TIMES=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
