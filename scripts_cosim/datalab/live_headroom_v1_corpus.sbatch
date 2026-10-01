#!/bin/bash
# live_headroom_v1 -- sweep the CD-induced live snapshots (docs/lineages/live_headroom_v1.md). make_warm_corpus.py with NO synthetic
# backlog: the state is exactly what the capture saw (live standing queue), batches are whole peer groups of 2+ tasks.
# One array task per development topology; its x20/x30/x50 x g0-g3 units in sequence; evenly spaced snapshots (no RNG).
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/live_headroom scripts_cosim/datalab/live_headroom_v1_corpus.sbatch
#SBATCH --job-name=lh1-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh1-corpus-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lh1-corpus-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
CAP=/home/nikola.lukic/gnn-herosim-wt/small_batch/simulation_data/live_headroom_v1
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
IN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
TOPOS=($(python3 -c "import json;print(*json.load(open('$IN/selected.json'))['topologies'])"))
T=${SLURM_ARRAY_TASK_ID}; TOPO=${TOPOS[$T]}
OUT=simulation_data/live_headroom_v1/gnn_datasets_live_headroom_v1
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 )); LIMIT=${LIMIT:-30}; TARGET=${TARGET:-20000}
mkdir -p "$OUT" simulation_data/live_headroom_v1/shards
RI=0
for RUNG in x20 x30 x50; do
  WI=0
  for WINDOW in g0 g1 g2 g3; do
    TAG="cc40s${TOPO}_${RUNG}_${WINDOW}"; SNAP="$CAP/snapshots/${TAG}.jsonl"
    CFG="$IN/grounded_${RUNG}/cfg/cc40s${TOPO}.json"; TRACE="$IN/grounded/wl/grounded_${WINDOW}_n50000.json"
    START=$(( (T * 12 + RI * 4 + WI) * 100 )); WI=$(( WI + 1 ))
    if [[ ! -s "$SNAP" || ! -f "$CAP/capture_summaries/${TAG}.summary.json" ]]; then echo "[$TAG] no finished capture; skipped"; continue; fi
    SHARD=simulation_data/live_headroom_v1/shards/${TAG}.jsonl
    python3 - "$SNAP" "$SHARD" "$(( LIMIT * 3 / 2 ))" <<'PY'
import json, sys
from scripts_cosim.make_warm_corpus import SnapshotRejected, check_consecutive_batch
src, dst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = []
for line in open(src):
    if line.strip():
        try: check_consecutive_batch(json.loads(line), 2); ok.append(line)
        except SnapshotRejected: pass
step = max(1, len(ok) // want)
keep = ok[::step][:want]
open(dst, "w").writelines(keep)
print(f"[shard] {len(ok)} eligible snapshots, offering {len(keep)} (every {step})")
PY
    python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
      --output-dir "$OUT" --source-tag "lh_${TAG}" --start-index "$START" --limit "$LIMIT" \
      --target-combos "$TARGET" --workers "$WORKERS" --no-cap-filter --min-choice-fraction 0.2 --quiet \
      --variable-group-min 2 || echo "[$TAG] corpus step failed (rc $?)"
    n=$(grep -c "\"source_tag\": \"lh_${TAG}\".*\"status\": \"success\"" "$OUT/warm_manifest.jsonl" || true)
    echo "=== $TAG: $n success datasets ==="
  done
  RI=$(( RI + 1 ))
done
