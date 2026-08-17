#!/bin/bash
#SBATCH --job-name=cv4-cosim
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv4-cosim-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv4-cosim-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# contention_v4_deepq co-sim: live-calibrated deep-queue grid under node_disk_v2.
#
# Why this grid exists: contention_v2's labels are all warm-path (0/900 datasets have any
# cold start or pull) with optimum averageQueueTime 0.27-5.47s, while the live regime that
# 873/v5.5 fails in is pure queueing — live seed-42 averageQueueTime is Knative 7.7/30.6/44.9s
# vs GNN 87.9/210.0/503.4s, with averagePullTime 0.017s. v4 pushes pre-seeded queue depth into
# the winning policy's band so co-locating a batch is genuinely expensive in the labels.
#
# No #SBATCH --array here on purpose: pass --array at submit time so pilot and full runs share
# one file. The shard runner fails loud without SLURM_ARRAY_TASK_ID.
#
# Pilot:  sbatch --array=0-2 --export=ALL,TOTAL_DATASETS=27,NUM_SHARDS=3,OUTPUT_SUBDIR=gnn_datasets_4tasks_contention_v4_pilot ...
# Full:   sbatch --array=0-9 ...

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_contention_v4_deepq}"
export GRID=contention_v4_deepq
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START="${REMAINING_START:-0}"
export TOTAL_DATASETS="${TOTAL_DATASETS:-900}"
export NUM_SHARDS="${NUM_SHARDS:-10}"
export ONLY_MISSING_JSONL="${ONLY_MISSING_JSONL:-0}"
export ALLOW_NON_UNIQUE=1
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
