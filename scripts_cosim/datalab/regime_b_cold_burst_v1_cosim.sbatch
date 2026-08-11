#!/bin/bash
#SBATCH --job-name=rb-cosim
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/regime-b-cosim-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/regime-b-cosim-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-9

# Regime B co-sim: platform_reuse_v1 scarce-warm grid (450 ds, 10×45).
# Fail-loud if warmth_physics ≠ platform_reuse_v1 (enforced in generator).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_regime_b_cold_burst_v1
export GRID=regime_b_cold_burst_v1
export WARMTH_PHYSICS=platform_reuse_v1
export REMAINING_START=0
export TOTAL_DATASETS=450
export NUM_SHARDS=10
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=1
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
