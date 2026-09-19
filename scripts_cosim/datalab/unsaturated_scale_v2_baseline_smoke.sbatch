#!/bin/bash
# unsaturated_scale_v2 -- BASELINE SMOKE TEST, not a study arm.
#
# `unsaturated_scale_v1` closed with exactly one baseline (`knative_network`), and its finding
# was that the entire margin is a 6.82 s batch-assembly wait the learned arms pay and reactive
# does not. `knative_network_batch` is reactive WITH that wait, so it is the control that
# separates "the model is better" from "batching is worse" -- the single most load-bearing new
# arm in v2. None of these three has ever been served on an 80-server cell.
#
# "An instrument is not on until a result file has its output" (docs/lessons.md): the arms run
# here first, one each, and the result file is ASSERTED, before any of them enters a design.
# Writes to a scratch directory -- a smoke arm must never reach a study table.
#
#   sbatch --array=0-2 scripts_cosim/datalab/unsaturated_scale_v2_baseline_smoke.sbatch
#
#SBATCH --job-name=us2-smoke
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/us2-smoke-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/us2-smoke-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-2

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
OUT=simulation_data/unsaturated_scale_v2/smoke
mkdir -p "$OUT"
CELL=cs80s9001
CFG="$GATE/configs/${CELL}.json"
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"

# Exactly the env every arm on these cells is served under.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1

POLICIES=(knative_network_batch knative_network_ect random_network)
POLICY=${POLICIES[$SLURM_ARRAY_TASK_ID]}
for f in "$CFG" "$WORKLOAD"; do
  [[ -f "$f" ]] || { echo "FAIL LOUD: missing $f" >&2; exit 1; }
done

RAW="$OUT/${CELL}__F4000__${POLICY}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))
echo "[smoke] $POLICY wall=${WALL}s"

python3 - "$RAW" "$POLICY" "$WALL" <<'PY'
import json, sys
raw, policy, wall = sys.argv[1], sys.argv[2], int(sys.argv[3])
doc = json.load(open(raw)); st = doc.get("stats") or doc
n = st.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: {policy} num_tasks={n!r}; every arm on this rung runs 50000. "
                     "A baseline that cannot finish the trace cannot be a baseline.")
e = st.get("averageElapsedTime"); q = st.get("averageQueueTime"); w = st.get("averageWaitTime")
if not (isinstance(e, (int, float)) and e > 0):
    raise SystemExit(f"FAIL LOUD: {policy} averageElapsedTime={e!r} is not a positive number")
share = float(q) / float(e)
print(f"[smoke] {policy}: elapsed={e:.3f}s queue={q:.3f}s batch_wait={w:.3f}s "
      f"queue_share={share:.3f} wall={wall}s")
# The number this smoke test exists for: reactive pays 0.00 s of batch wait and every learned
# arm pays 6.82 s. If knative_network_batch pays ~0 too, it is NOT a batching control and the
# v2 design must say so rather than quietly claiming one.
if policy == "knative_network_batch":
    print(f"[smoke] BATCHING CONTROL: knative_network_batch batch wait = {float(w):.3f} s "
          f"(reactive 0.000 s, learned arms 6.82 s)")
print(f"[smoke] OK -- {policy} serves an 80-server cell and the read's keys are present")
PY
rm -f "$RAW"
echo "=== smoke done: $POLICY ==="
