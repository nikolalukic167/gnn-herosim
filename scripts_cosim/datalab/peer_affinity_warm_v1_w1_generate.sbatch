#!/bin/bash
# peer_affinity_warm_v1 -- W1 corpus: 12 brute-force-labelled datasets per (cell, source), spread over
# the warm-up run (every k-th aligned snapshot, k = aligned // 12), 20k-plan budget, cap-feasible draw.
# Train cells (s9001-s9024) -> gnn_datasets_peer_affinity_warm_v1_w1_train (48 tasks x 12 = 576);
# held-out cells (s9101-s9104) -> gnn_datasets_peer_affinity_warm_v1_w1_heldout (8 x 12 = 96).
#   task T: cell = CELLS[T / 2], source = (T % 2 == 0 ? knb : gnn); ids start at 12 * (index within corpus)
#   sbatch --array=0-55%20 scripts_cosim/datalab/peer_affinity_warm_v1_w1_generate.sbatch
#
#SBATCH --job-name=pa-warm-w1-gen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-gen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-gen-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python3 -c "import sys;print('[INTERP]',sys.executable)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
# physics env: identical to the W0 generation (t1b/x800 generation env + the tick knob)
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12

CELLS=($(seq 9001 9024) $(seq 9101 9104))
T=${SLURM_ARRAY_TASK_ID}
SEED=${CELLS[$(( T / 2 ))]}
if [[ $(( T % 2 )) -eq 0 ]]; then SRC=knb; else SRC=gnn; fi
if [[ $T -lt 48 ]]; then OUT=simulation_data/gnn_datasets_peer_affinity_warm_v1_w1_train; START=$(( T * 12 ));
else OUT=simulation_data/gnn_datasets_peer_affinity_warm_v1_w1_heldout; START=$(( (T - 48) * 12 )); fi
PER=${PER:-12}; TARGET=${TARGET:-20000}; WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
G=simulation_data/peer_affinity_live_gate
W1=simulation_data/peer_affinity_warm_v1/w1
SNAP="$W1/snapshots/cell_s${SEED}_${SRC}.jsonl"
SHARD="$W1/shards/cell_s${SEED}_${SRC}_shard0.jsonl"
TRACE=$G/workloads/workload-150-100-peer_p2_x200.json
CFG=$G/configs/cell_s${SEED}.json
mkdir -p "$W1/shards" "$OUT" logs
[[ -s "$SNAP" ]] || { echo "FAIL LOUD: snapshots missing: $SNAP (capture not finished?)" >&2; exit 1; }
[[ -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: $CFG or $TRACE missing" >&2; exit 1; }
echo "[TASK $T] cell_s$SEED source=$SRC -> $OUT ids from $START (per $PER, target $TARGET, workers $WORKERS)"

EVERY=$(python3 - "$SNAP" "$PER" <<'PY'
import json, sys
from scripts_cosim.make_warm_corpus import check_aligned_peer_group, SnapshotRejected
n = 0
for line in open(sys.argv[1]):
    try: check_aligned_peer_group(json.loads(line), 10); n += 1
    except SnapshotRejected: pass
per = int(sys.argv[2])
if n < per: raise SystemExit(f"FAIL LOUD: only {n} aligned snapshots, need {per}")
print(max(1, n // per))
PY
)
echo "[TASK $T] every=$EVERY"
python3 scripts_cosim/split_warm_snapshots.py --snapshots "$SNAP" --out-dir "$W1/shards" \
  --prefix "cell_s${SEED}_${SRC}" --shards 1 --per "$PER" --every "$EVERY"
python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
  --output-dir "$OUT" --source-tag "${SRC}_s${SEED}" --start-index "$START" --limit "$PER" \
  --target-combos "$TARGET" --workers "$WORKERS" --quiet
# A snapshot whose FULL live slate is cap-infeasible at alpha=2.0 is recorded `rejected` with its
# reason (no subset can carry a label; 2026-09-14, cell_s9007/gnn snapshot 133). That is a
# measurement, not a failure: allow up to 2 of PER, require every chosen snapshot accounted for.
# Anything other than success/rejected already raised inside make_warm_corpus.py.
n=$(grep -c "\"source_tag\": \"${SRC}_s${SEED}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
r=$(grep -c "\"source_tag\": \"${SRC}_s${SEED}\".*\"status\": \"rejected\"" "$OUT/warm_manifest.jsonl" || true)
echo "=== task $T done: $n success, $r rejected datasets for cell_s${SEED}/${SRC} ==="
[[ $(( n + r )) -eq "$PER" ]] || { echo "FAIL LOUD: manifest accounts for $((n + r)) of $PER chosen snapshots" >&2; exit 1; }
[[ "$n" -ge $(( PER - 2 )) ]] || { echo "FAIL LOUD: only $n success datasets (need >= $((PER - 2))), $r rejected" >&2; exit 1; }
