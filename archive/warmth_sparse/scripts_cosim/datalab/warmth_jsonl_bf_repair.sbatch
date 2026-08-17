#!/bin/bash
#SBATCH --job-name=warmth-jsonl-bf
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-jsonl-bf-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-jsonl-bf-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Re-run brute-force for warmth_v2 datasets with best.json but missing placements.jsonl.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GRID=warmth_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
export START_FROM=0
export MAX_DATASETS=500
export SHARD_LABEL=jsonl_bf_repair
export ONLY_MISSING_JSONL=1

echo "=== Recover JSONL from surviving .bf_scratch (if any) ==="
bash scripts_cosim/recover_placements_jsonl_from_scratch.sh \
  "simulation_data/${OUTPUT_SUBDIR}" || true

bash scripts_cosim/datalab/run_warmth_regen_range.sh

echo "=== JSONL repair done ==="
find "simulation_data/${OUTPUT_SUBDIR}" -path '*/placements/placements.jsonl' -size +0c 2>/dev/null | wc -l
