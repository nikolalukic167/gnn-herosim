#!/bin/bash
# backlog_corpus_v1 -- the joint_burst_v2 corpus replayed WITH synthetic backlog
# (docs/lineages/backlog_corpus_v1.md). Same snapshot shards, traces, cells, candidate draw
# (seed 9001) and filters as joint_burst_v2_corpus.sbatch, so each dataset's no-backlog twin is
# the jb2 dataset of the same snapshot. Two passes with independent backlog draws (seeds 7001,
# 7002): 96 units U -> pass U/48, cell/window U%48 exactly as jb2. Array task T runs units
# 8T..8T+7 in sequence (a jb2 unit took <= 7 min), so the corpus holds 12 of the account's 50
# submit slots instead of 96.
#
#   sbatch --array=0-11 scripts_cosim/datalab/backlog_corpus_v1_corpus.sbatch
#
#SBATCH --job-name=bc1-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bc1-corpus-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bc1-corpus-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/backlog_corpus
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
DRAIN=simulation_data/drainable_objective_v1/drain_table.json
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing" >&2; exit 1; }
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12
export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN"
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }

RUNGS="${RUNGS:-0 3 8 20}"; BUSY="${BUSY:-0.5}"; TASKSEC="${TASKSEC:-4.5}"
SEEDS=($(seq 9201 9224)); WINDOWS=(w0 w1); BSEEDS=(7001 7002)
UNITS="${UNITS:-$(seq $(( SLURM_ARRAY_TASK_ID * 8 )) $(( SLURM_ARRAY_TASK_ID * 8 + 7 )))}"
OUTROOT="${OUTROOT:-simulation_data}"   # a smoke run sets OUTROOT=simulation_data/backlog_corpus_v1/smoke
LIMIT=${LIMIT:-150}; TARGET=${TARGET:-20000}; WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
G=simulation_data/peer_affinity_live_gate
for U in $UNITS; do
  PASS=$(( U / 48 )); J=$(( U % 48 ))
  SEED=${SEEDS[$(( J / 2 ))]}; WINDOW=${WINDOWS[$(( J % 2 ))]}; BSEED=${BSEEDS[$PASS]}
  CELL="cc40s${SEED}"
  if (( SEED >= 9222 )); then OUT=$OUTROOT/gnn_datasets_backlog_corpus_v1_heldout; else OUT=$OUTROOT/gnn_datasets_backlog_corpus_v1_train; fi
  START=$(( PASS * 20000 + J * 200 ))
  SHARD=simulation_data/joint_burst_v2/shards/${CELL}_${WINDOW}_shard0.jsonl
  if [[ $WINDOW == w0 ]]; then TRACE=$G/workloads/burst_drainable_f4000_n50000.json; else TRACE=$G/workloads/burst_drainable_${WINDOW}_n50000.json; fi
  CFG=$G/configs/${CELL}.json
  if [[ ! -s "$SHARD" ]]; then echo "[UNIT $U] no jb2 shard for $CELL $WINDOW (jb2 skipped it too)"; continue; fi
  [[ -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: $CFG or $TRACE missing" >&2; exit 1; }
  mkdir -p "$OUT"
  echo "[UNIT $U] pass $PASS (backlog seed $BSEED, rungs $RUNGS, busy $BUSY, task-s $TASKSEC) $CELL $WINDOW -> $OUT from $START"
  python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
    --output-dir "$OUT" --source-tag "pgb_${CELL}_${WINDOW}_b${BSEED}" --start-index "$START" --limit "$LIMIT" \
    --target-combos "$TARGET" --workers "$WORKERS" --no-cap-filter --min-choice-fraction 0.2 --quiet \
    --synthetic-backlog-rungs $RUNGS --synthetic-backlog-busy-prob "$BUSY" \
    --synthetic-backlog-task-seconds "$TASKSEC" --synthetic-backlog-seed "$BSEED"
  n=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}_b${BSEED}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
  r=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}_b${BSEED}\".*\"status\": \"rejected\"" "$OUT/warm_manifest.jsonl" || true)
  echo "=== unit $U done: $n success, $r rejected for $CELL/$WINDOW pass $PASS ==="
  [[ "$n" -ge 4 ]] || { echo "FAIL LOUD: only $n success datasets (need >= 4), $r rejected" >&2; exit 1; }
done
