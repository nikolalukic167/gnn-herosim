#!/bin/bash
# joint_burst_v1 -- the corpus: one task per capture file (training cell x window), 12 datasets
# each from its aligned snapshots, brute-force sweep on the measured clock. docs/lineages/joint_burst_v1.md.
#
# 48 tasks = 24 training cells x {w0, w1}. Cells 9222-9224 form the HELD-OUT corpus (their
# datasets go to ..._heldout and never train); everything else is training. A capture file that
# does not exist (the run spun) is skipped loudly and the corpus reader counts what is left.
#
#   sbatch --array=0-47 scripts_cosim/datalab/joint_burst_v1_corpus.sbatch
#
#SBATCH --job-name=jb1-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb1-corpus-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb1-corpus-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-47

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
# the W1 generation physics, verbatim, plus the measured clock the psv3 corpora were generated with
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing" >&2; exit 1; }
export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN"
echo "[CLOCK] measured: $DRAIN ($(md5sum "$DRAIN" | cut -c1-12))"

SEEDS=($(seq 9201 9224)); WINDOWS=(w0 w1)
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( T / 2 ))]}; WINDOW=${WINDOWS[$(( T % 2 ))]}
CELL="cc40s${SEED}"
if (( SEED >= 9222 )); then OUT=simulation_data/gnn_datasets_joint_burst_v1_heldout; else OUT=simulation_data/gnn_datasets_joint_burst_v1_train; fi
# under bursts about 2 in 3 aligned snapshots are rejected by the corpus rules (no choice, or the
# live slate over the alpha=2.0 cap -- measured locally on cc40s9001 w0: 4 of 12), so 40 snapshots
# are offered per capture file and the builder stops at LIMIT successes
START=$(( T * 12 )); PER=${PER:-40}; LIMIT=${LIMIT:-12}; TARGET=${TARGET:-20000}; WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
G=simulation_data/peer_affinity_live_gate
JB=simulation_data/joint_burst_v1
SNAP="$JB/snapshots/${CELL}_${WINDOW}.jsonl"
SHARD="$JB/shards/${CELL}_${WINDOW}_shard0.jsonl"
if [[ $WINDOW == w0 ]]; then TRACE=$G/workloads/burst_drainable_f4000_n50000.json; else TRACE=$G/workloads/burst_drainable_${WINDOW}_n50000.json; fi
CFG=$G/configs/${CELL}.json
mkdir -p "$JB/shards" "$OUT" logs
if [[ ! -s "$SNAP" ]]; then echo "[TASK $T] no snapshots for $CELL $WINDOW (the capture spun); skipped"; exit 0; fi
[[ -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: $CFG or $TRACE missing" >&2; exit 1; }
echo "[TASK $T] $CELL $WINDOW -> $OUT ids from $START (per $PER, target $TARGET, workers $WORKERS)"
EVERY=$(python3 - "$SNAP" "$PER" <<'PY'
import json, sys
from scripts_cosim.make_warm_corpus import check_aligned_peer_group, SnapshotRejected
n = 0
for line in open(sys.argv[1]):
    try: check_aligned_peer_group(json.loads(line), 10); n += 1
    except SnapshotRejected: pass
per = int(sys.argv[2])
if n < 12: raise SystemExit(f"FAIL LOUD: only {n} aligned snapshots, need 12")
per = min(per, n)
print(max(1, n // per))
PY
)
echo "[TASK $T] every=$EVERY"
python3 scripts_cosim/split_warm_snapshots.py --snapshots "$SNAP" --out-dir "$JB/shards" \
  --prefix "${CELL}_${WINDOW}" --shards 1 --per "$PER" --every "$EVERY"
python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
  --output-dir "$OUT" --source-tag "pgb_${CELL}_${WINDOW}" --start-index "$START" --limit "$LIMIT" \
  --target-combos "$TARGET" --workers "$WORKERS" --quiet
n=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
r=$(grep -c "\"source_tag\": \"pgb_${CELL}_${WINDOW}\".*\"status\": \"rejected\"" "$OUT/warm_manifest.jsonl" || true)
echo "=== task $T done: $n success, $r rejected datasets for $CELL/$WINDOW ==="
[[ "$n" -ge $(( LIMIT - 3 )) ]] || { echo "FAIL LOUD: only $n success datasets (need >= $((LIMIT - 3))), $r rejected" >&2; exit 1; }
