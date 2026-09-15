#!/bin/bash
# peer_affinity_v1 -- denser-graph rung: the registered read, once every report is on disk
# (peer_affinity_v1_x800_score.sbatch wrote them). The read re-runs nothing: it finds each
# report cached, picks the lr per arm on VALIDATION only, and reports the paired contrasts.
#   sbatch --dependency=afterok:<score job> --export=ALL,RUNG=p3 <this>
#
#SBATCH --job-name=pa-x800-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-x800-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-x800-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""

RUNG="${RUNG:?set RUNG=p3 or RUNG=p2}"
TAG="x800${RUNG}"
python3 scripts_cosim/peer_affinity_t1_read.py \
  --tag "$TAG" --seeds 16 \
  --models-dir models \
  --cache-dir "simulation_data/graphs_cache_peer_affinity_v1_x800_${RUNG}" \
  --split "experiments/peer_affinity_v1_x800_${RUNG}_split.json" \
  --reports-dir "simulation_data/peer_affinity_${TAG}_reports" \
  --logs-dir logs \
  --out "simulation_data/peer_affinity_${TAG}_read.json"
