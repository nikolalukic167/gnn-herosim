#!/bin/bash
#SBATCH --job-name=merged-jsonl
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/merged-jsonl-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/merged-jsonl-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Merged warmth+sparse co-sim corpus: ensure every ds_* has best.json + placements.jsonl.
# (824-graph training cache = warmth_v2 + sparse_warmth_v2)

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

bash scripts_cosim/datalab/run_warmth_merged_jsonl_finish.sh
