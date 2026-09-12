#!/bin/bash
# serving_gap_v2 H3 — group splitting, one array task per (corpus, arm, seed).
# Reads the slim traces from serving_gap_v1 (every batch's plan) and the workload's peer table,
# and measures how much of each batch lands together. O(k^2) per batch over 45,375 batches, so
# this is an array job, never a login-node command.
#
#SBATCH --job-name=sg2-h3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sg2-h3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sg2-h3-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-95

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2

T=${SLURM_ARRAY_TASK_ID}
CORPORA=(x200p2 x800p2 x800p3); ARMS=(gnn mpoff)
CORPUS=${CORPORA[$(( T / 32 ))]}; ARM=${ARMS[$(( (T % 32) / 16 ))]}; SEED=$(( T % 16 + 1 ))
TRACE=simulation_data/serving_gap_v1/traces_h1/$CORPUS/${ARM}_s${SEED}.pkl
OUT_DIR=simulation_data/serving_gap_v2/h3/$CORPUS
OUT=$OUT_DIR/${ARM}_s${SEED}.json
mkdir -p "$OUT_DIR" logs
[[ -f "$OUT" ]] && { echo "[skip] $OUT exists"; exit 0; }
[[ -s "$TRACE" ]] || { echo "FAIL LOUD: trace missing: $TRACE" >&2; exit 1; }
python3 scripts_cosim/serving_gap_trace_read.py h3-one \
  --corpus "$CORPUS" --arm "$ARM" --seed "$SEED" --trace "$TRACE" --output "$OUT"
echo "=== task $T done ==="
