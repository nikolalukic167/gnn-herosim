#!/bin/bash
#SBATCH --job-name=rc8t-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-c-8task-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-c-8task-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

# route_c_link_transfer_v1 SCREEN — the registered CONTINGENCY rung (LINEAGES.md,
# registration + 4-task ladder outcome, 2026-08-26). The 4-task ladder measured a
# STRUCTURAL ceiling on link-wait share (wait/(wait+transfer) median 4-6%, max 8.8%
# against the 0.10 manipulation bar): one client + a diamond DAG cap concurrent
# transfers at 2, so no bandwidth or payload value can make link waiting material.
# This rung widens the ladder along the only remaining lever — concurrency: two
# diamond4 instances from independently drawn clients (grid route_c_link_screen_8task),
# at the best 4-task configuration (n_core=4/attach=1/chords=0 from the preset's
# backbone_defaults; bw=25 MB/s via LINK_BANDWIDTH_MBPS; contended input payload
# 150 MB via HEROSIM_INPUT_SIZE_BYTES).
#
# 24 datasets, 8 shards of 3. Sweep sizes are the route_b 8-task family's
# (post-uniqueness 27,648-516,096 rows measured there); the measured rate is
# ~2,935 sims/min on 8 cores, so a shard is minutes-to-a-few-hours; 12 h is margin.
#
# MAX_PLACEMENT_COMBINATIONS_SKIP: same derivation as route_b_8task_probe.sbatch —
# the threshold tests total_possible, computed BEFORE the uniqueness constraint
# (executecosimulation.py), whose hard bound for this grid family is 1248^2 =
# 1,557,504; 2,000,000 cannot silently skip any dataset. 1,000,000 was proven too
# low there (job 713656).
#
# HEROSIM_COSIM_KEEP_ALIVE / HEROSIM_RETAIN_TASK_TIMES are half the Arm S recipe
# (job 713615 died without them — see route_b_8task_probe.sbatch's comment).
# HEROSIM_RETAIN_LINK_STATS=1 is what writes link_wait_total into every
# placements.jsonl row — without it the screen's manipulation check
# (scripts_cosim/route_c_screen_manipulation_check.py) refuses the corpus.
#
# ALLOW_NON_UNIQUE stays 0 (route_b decode-mask convention, unchanged).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_route_c_link_screen_8task
export GRID=route_c_link_screen_8task
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=24
export NUM_SHARDS=8
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=0
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export LINK_BANDWIDTH_MBPS=25
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000
export HEROSIM_INPUT_SIZE_BYTES=157286400

export HEROSIM_COSIM_KEEP_ALIVE=1000000
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_RETAIN_LINK_STATS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
