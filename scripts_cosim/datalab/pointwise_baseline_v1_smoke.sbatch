#!/bin/bash
# pointwise_baseline_v1 -- ONE-ARM SERVING SMOKE TEST, not a gate arm.
#
# "An instrument is not on until a result file has its output" (docs/lessons.md): the
# always-on queue-feature probe produced nothing in any gate for weeks, and what caught it was
# a one-arm smoke test that failed loud. `mlp_batch` has never been served on these cells at
# all, so this runs exactly one arm on the R0 cell with the real gate's workload, physics and
# env, and then ASSERTS the result file carries what the read will need.
#
# Writes to a scratch directory, never to results/po_v1 -- a smoke arm must not be able to end
# up in a gate table.
#
#SBATCH --job-name=pb-smoke
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pb-smoke-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pb-smoke-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

GATE=simulation_data/peer_affinity_live_gate
OUT=simulation_data/pointwise_baseline_v1/smoke
mkdir -p "$OUT"
CELL=cs6s9001
CFG="$GATE/configs/${CELL}.json"
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"
CK="models/tabular/pointwise-baseline-v1-516-mlp-lr2e3_seed1.pt"

# Exactly the env every other arm on these cells is served under.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1
export PARTIAL_STATE_CONTRACT=partial_state_v3
export PARTIAL_STATE_PEER_MASS=1

for f in "$CFG" "$WORKLOAD" "$CK"; do
  [[ -f "$f" ]] || { echo "FAIL LOUD: missing $f" >&2; exit 1; }
done

RAW="$OUT/${CELL}__R0__pb516_mlp_s1.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy mlp_batch --mlp-model "$CK" --output "$RAW"
echo "[smoke] wall=$(( $(date +%s) - START ))s"

# The point of the exercise: prove the RESULT FILE carries what the read needs, rather than
# trusting that the run "worked" because it exited 0.
python3 - "$RAW" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
stats = r.get("stats") or r
def pick(*names):
    for n in names:
        for src in (stats, r):
            if isinstance(src, dict) and src.get(n) is not None:
                return n, src[n]
    return None, None
k, elapsed = pick("averageElapsedTime")
if elapsed is None:
    raise SystemExit("FAIL LOUD: no averageElapsedTime -- the gate read is keyed on it")
k2, n = pick("num_tasks", "numTasks", "totalTasks")
prov = r.get("run_provenance") or {}
print(f"[smoke] averageElapsedTime={elapsed}")
print(f"[smoke] tasks={n} ({k2})")
print(f"[smoke] python_env={prov.get('python_env')}")
print(f"[smoke] policy={r.get('policy') or stats.get('policy')}")
if not (isinstance(elapsed, (int, float)) and elapsed > 0):
    raise SystemExit(f"FAIL LOUD: averageElapsedTime={elapsed!r} is not a positive number")
if n is not None and int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n}, the gate's arms all run 50000")
# THE BAR THIS SMOKE TEST EXISTS FOR. The scheduler falls back to shortest-queue whenever the
# model returns no placement, and it counts that -- but nothing asserted the count, so a
# 100%-fallback arm produces a perfectly well-formed result file whose elapsed time is a
# heuristic's, not a model's. Measured on the first run of this arm: 50000/50000 fallbacks.
sc = (stats.get("schedulerCounters") or {})
pure, fb = sc.get("gnn_pure_decisions"), sc.get("fallback_decisions")
if pure is None or fb is None:
    raise SystemExit("FAIL LOUD: stats.schedulerCounters carries no decision counters; "
                     "without them a fallback-only arm is indistinguishable from a model arm")
print(f"[smoke] decisions: model={pure} fallback={fb}")
if fb != 0:
    raise SystemExit(f"FAIL LOUD: {fb} of {pure + fb} decisions were shortest-queue FALLBACKS. "
                     "This arm's elapsed time is not the model's and must never reach a gate table.")
print("[smoke] OK -- mlp_batch serves on a gate cell, the read's key is present, 0 fallbacks")
PY
echo "=== smoke complete ==="
