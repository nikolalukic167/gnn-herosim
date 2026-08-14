#!/bin/bash
#SBATCH --job-name=cv5-quick
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv5-quick-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv5-quick-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# contention_v5_quick_test: Deep-queue coupling optimization quick test (216 datasets).
#
# Hypothesis: norm450/uniform200_800/pois500 (4-8x deeper than v2) + scarce warm resources
# pushes coupling rate from v2's 7.1% to 15-25%, giving GNN measurable advantage.
#
# Submit: sbatch scripts_cosim/datalab/contention_v5_quick_test.sbatch

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR="gnn_datasets_4tasks_contention_v5_quick_test"
export GRID=contention_v5_quick_test
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=216
export NUM_SHARDS=3
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=1
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
