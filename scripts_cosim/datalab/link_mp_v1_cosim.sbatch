#!/bin/bash
# link_mp_v1 — generate the binding-backbone training corpus, AT THE PINNED COMMIT.
#
# Registration: docs/lineages/link_mp_v1.md. Three fabric variants (one per live-gate
# backbone family), 600 target datasets each = 1,800 total:
#   grid link_mp_v1_core4_bw0p5 -> gnn_datasets_4tasks_link_mp_v1_core4_bw0p5
#   grid link_mp_v1_core4_bw1p5 -> gnn_datasets_4tasks_link_mp_v1_core4_bw1p5
#   grid link_mp_v1_core8_bw1p5 -> gnn_datasets_4tasks_link_mp_v1_core8_bw1p5
#
# Array layout: 30 tasks = 3 grids x 10 shards. The generic sharder
# (run_contention_regen_shard.sh) shards by SLURM_ARRAY_TASK_ID, so the within-grid shard
# index is re-exported over it — deliberate, and confined to this file.
#
# ALLOW_NON_UNIQUE stays at the sharder's default 1: the contention-series presets REQUIRE
# --allow-non-unique-replicas (see the preset comment in generate_gnn_datasets_fast.py).
#SBATCH --job-name=linkmp-cosim
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/linkmp-cosim-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/linkmp-cosim-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-29

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-8aef27a
PIN_COMMIT=8aef27a98fd636000008468d75a52d645f999969
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"

HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

I="$SLURM_ARRAY_TASK_ID"
GRIDS=(link_mp_v1_core4_bw0p5 link_mp_v1_core4_bw1p5 link_mp_v1_core8_bw1p5)
GRID_IDX=$(( I / 10 ))
SHARD=$(( I % 10 ))

export GRID="${GRIDS[$GRID_IDX]}"
export OUTPUT_SUBDIR="gnn_datasets_4tasks_${GRID}"
export TOTAL_DATASETS=600
export NUM_SHARDS=10
export WARMTH_PHYSICS=node_disk_v2
export PROJECT_ROOT="$PIN_DIR"

echo "=== task=${I} grid=${GRID} shard=${SHARD}/10 -> simulation_data/${OUTPUT_SUBDIR} ==="
# The sharder reads SLURM_ARRAY_TASK_ID for its shard arithmetic; hand it the
# within-grid index.
export SLURM_ARRAY_TASK_ID="$SHARD"
bash scripts_cosim/datalab/run_contention_regen_shard.sh
echo "=== task=${I} grid=${GRID} shard=${SHARD} COMPLETE ==="
