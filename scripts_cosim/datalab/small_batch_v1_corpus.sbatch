#!/bin/bash
# small_batch_v1 -- the backlog_corpus_v1 recipe on live-sized batches (docs/lineages/small_batch_v1.md).
# Same synthetic backlog (rungs 0 3 8 20, busy 0.5, 4.5 task-s, seed 7001), same cap-free candidate draw and
# the same filters as backlog_corpus_v1_corpus.sbatch; the ONLY change is the batch: whole peer groups of the grounded
# workload (2..~12 tasks) instead of aligned 10-task groups (make_warm_corpus.py --variable-group-min 2).
# One array task per (training cell, grounded window): 24 x 4 = 96.
#
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/small_batch scripts_cosim/datalab/small_batch_v1_corpus.sbatch
#
#SBATCH --job-name=sb1-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sb1-corpus-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sb1-corpus-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-95
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/small_batch}
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

SEEDS=($(seq 9201 9224)); WINDOWS=(g0 g1 g2 g3)
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( T / 4 ))]}; WINDOW=${WINDOWS[$(( T % 4 ))]}
CELL="cc40s${SEED}"
SB=simulation_data/small_batch_v1
if (( SEED >= 9222 )); then OUT=$SB/gnn_datasets_small_batch_v1_heldout; else OUT=$SB/gnn_datasets_small_batch_v1_train; fi
START=$(( T * 200 )); LIMIT=${LIMIT:-55}; TARGET=${TARGET:-20000}; WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
G=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate
TRACE=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs/grounded/wl/grounded_${WINDOW}_n50000.json
CFG=$G/configs/${CELL}.json
SNAP=$SB/snapshots/${CELL}_${WINDOW}.jsonl
RUNGS="${RUNGS:-0 3 8 20}"; BUSY="${BUSY:-0.5}"; TASKSEC="${TASKSEC:-4.5}"; BSEED=7001
mkdir -p "$OUT"
if [[ ! -s "$SNAP" ]]; then echo "[TASK $T] no snapshots for $CELL $WINDOW (the capture spun); skipped"; exit 0; fi
[[ -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: $CFG or $TRACE missing" >&2; exit 1; }
echo "[TASK $T] $CELL $WINDOW -> $OUT ids from $START (limit $LIMIT, rungs $RUNGS)"
python3 scripts_cosim/make_warm_corpus.py --snapshots "$SNAP" --trace "$TRACE" --cell-config "$CFG" \
  --output-dir "$OUT" --source-tag "sb_${CELL}_${WINDOW}" --start-index "$START" --limit "$LIMIT" \
  --target-combos "$TARGET" --workers "$WORKERS" --no-cap-filter --min-choice-fraction 0.2 --quiet \
  --variable-group-min 2 \
  --synthetic-backlog-rungs $RUNGS --synthetic-backlog-busy-prob "$BUSY" \
  --synthetic-backlog-task-seconds "$TASKSEC" --synthetic-backlog-seed "$BSEED"
n=$(grep -c "\"source_tag\": \"sb_${CELL}_${WINDOW}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
r=$(grep -c "\"source_tag\": \"sb_${CELL}_${WINDOW}\".*\"status\": \"rejected\"" "$OUT/warm_manifest.jsonl" || true)
echo "=== task $T done: $n success, $r rejected for $CELL/$WINDOW ==="
[[ "$n" -ge 4 ]] || { echo "FAIL LOUD: only $n success datasets (need >= 4), $r rejected" >&2; exit 1; }
