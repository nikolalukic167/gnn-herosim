#!/bin/bash
# joint_burst_v2 -- the burst corpus WITHOUT the cap filter: every aligned snapshot of every
# capture run is offered, the loaded (cap-infeasible) states are kept, and a snapshot needs only
# 20 % of its tasks to have a choice. docs/lineages/joint_burst_v2.md.
#
# joint_burst_v1 offered 60 snapshots per run and rejected 634 of 993 (579 for "no plan under the
# alpha=2.0 cap" = the loaded states the served model meets and never trained on). Same captures
# (simulation_data/joint_burst_v1/snapshots), same trace, same cells; 48 tasks (24 cells x 2
# windows, 20 have snapshots), ids unique per task (START = T * 200).
#
#   sbatch --array=0-47 scripts_cosim/datalab/joint_burst_v2_corpus.sbatch
#
#SBATCH --job-name=jb2-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb2-corpus-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb2-corpus-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --array=0-47
set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing" >&2; exit 1; }
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12
export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
SEEDS=($(seq 9201 9224)); WINDOWS=(w0 w1)
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( T / 2 ))]}; WINDOW=${WINDOWS[$(( T % 2 ))]}
CELL="cc40s${SEED}"
if (( SEED >= 9222 )); then OUT=simulation_data/gnn_datasets_joint_burst_v2_heldout; else OUT=simulation_data/gnn_datasets_joint_burst_v2_train; fi
START=$(( T * 200 )); LIMIT=${LIMIT:-150}; TARGET=${TARGET:-20000}; WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
MINCHOICE=${MINCHOICE:-0.2}
G=simulation_data/peer_affinity_live_gate
JB=simulation_data/joint_burst_v1
JB2=simulation_data/joint_burst_v2
SNAP="$JB/snapshots/${CELL}_${WINDOW}.jsonl"
SHARD="$JB2/shards/${CELL}_${WINDOW}_shard0.jsonl"
if [[ $WINDOW == w0 ]]; then TRACE=$G/workloads/burst_drainable_f4000_n50000.json; else TRACE=$G/workloads/burst_drainable_${WINDOW}_n50000.json; fi
CFG=$G/configs/${CELL}.json
mkdir -p "$JB2/shards" "$OUT" logs
if [[ ! -s "$SNAP" ]]; then echo "[TASK $T] no snapshots for $CELL $WINDOW (the capture spun); skipped"; exit 0; fi
[[ -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: $CFG or $TRACE missing" >&2; exit 1; }
echo "[TASK $T] $CELL $WINDOW -> $OUT ids from $START (limit $LIMIT, target $TARGET, min-choice $MINCHOICE, workers $WORKERS)"
read -r PERN <<< "$(python3 - "$SNAP" <<'PY'
import json, sys
from scripts_cosim.make_warm_corpus import check_aligned_peer_group, SnapshotRejected
n = 0
for line in open(sys.argv[1]):
    try: check_aligned_peer_group(json.loads(line), 10); n += 1
    except SnapshotRejected: pass
print(n)
PY
)"
if (( PERN < 12 )); then echo "[TASK $T] only $PERN aligned snapshots in $SNAP (the capture spun); skipped"; exit 0; fi
echo "[TASK $T] offering ALL $PERN aligned snapshots"
python3 scripts_cosim/split_warm_snapshots.py --snapshots "$SNAP" --out-dir "$JB2/shards" \
  --prefix "${CELL}_${WINDOW}" --shards 1 --per "$PERN" --every 1
python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
  --output-dir "$OUT" --source-tag "pgb_${CELL}_${WINDOW}" --start-index "$START" --limit "$LIMIT" \
  --target-combos "$TARGET" --workers "$WORKERS" --no-cap-filter --min-choice-fraction "$MINCHOICE" --quiet
n=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
r=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}\".*\"status\": \"rejected\"" "$OUT/warm_manifest.jsonl" || true)
echo "=== task $T done: $n success, $r rejected datasets for $CELL/$WINDOW ==="
[[ "$n" -ge 4 ]] || { echo "FAIL LOUD: only $n success datasets (need >= 4), $r rejected" >&2; exit 1; }
