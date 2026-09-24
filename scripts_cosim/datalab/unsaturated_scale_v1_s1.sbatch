#!/bin/bash
# unsaturated_scale_v1 -- S1, the BASELINE SWEEP. docs/lineages/unsaturated_scale_v1.md.
#
# Reactive Knative alone, on the four 80-server cells the whole R3 record runs on, across the
# arrival ladder. Its ONLY output is one selected factor (`select_factor` in
# scripts_cosim/unsaturated_scale_v1_read.py, a rule signed before this ran); the learned arms
# are not touched here. f300 is NOT re-run: results/psv3_p3's reactive R3 arm IS this cell at
# that factor, and the read pulls it from there.
#
# Task table: 6 factors x 4 cells = 24 tasks, index = factor * 4 + cell.
#   sbatch --array=0-23 scripts_cosim/datalab/unsaturated_scale_v1_s1.sbatch
#
#SBATCH --job-name=us-s1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/us-s1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/us-s1-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-23

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
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
OUT_DIR="$GATE/results/us_v1"
mkdir -p "$OUT_DIR"
FACTORS=(500 700 1000 2000 4000 8000)     # 300 reused from psv3_p3 (K_FACTORS minus 300)
TSEEDS=(9001 9002 9003 9005)              # K_CELLS; 9004 hangs on every policy
NS=80

# Exactly the env every arm on these cells is served under (psv3_p3 / po_v1).
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}
F=${FACTORS[$(( T / 4 ))]}; TSEED=${TSEEDS[$(( T % 4 ))]}
CELL="cs${NS}s${TSEED}"
CFG="$GATE/configs/${CELL}.json"
WORKLOAD_NAME="drainable_f${F}_n50000"
WORKLOAD="$GATE/workloads/${WORKLOAD_NAME}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD (drainable_regime_v1_s0_traces.sbatch with DRAIN_FACTORS=$F)" >&2; exit 1; }
python3 - "$CFG" "$NS" "$WORKLOAD" "$F" <<'PY'
import json, sys
cfg, want, wl, f = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
if int(cfg["nodes"]["server_nodes"]["count"]) != want:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares {cfg['nodes']['server_nodes']['count']} servers, rung expects {want}")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
# The trace must be a rescale of THE gate trace at THIS factor, cut at 50,000 -- a trace with
# the right name and the wrong provenance would make this a different ladder.
d = json.load(open(wl)); r = d.get("arrival_rescale") or {}
if float(r.get("factor", -1)) != f:
    raise SystemExit(f"FAIL LOUD: {wl} declares factor {r.get('factor')!r}, expected {f}")
if len(d["events"]) != 50000 or int((r.get("truncation") or {}).get("max_events", -1)) != 50000:
    raise SystemExit(f"FAIL LOUD: {wl} has {len(d['events'])} events, the ladder is n50000")
print(f"[trace] f{f:.0f}: {r.get('rate_after_per_s'):.3f} arrivals/s, span {r.get('span_after_s'):.0f} s")
PY

ARM="${CELL}__F${F}__reactive_s0"
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] factor=$F cell=$CELL arm=$ARM -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARM}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy knative_network --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$F" "$NS" "$WORKLOAD_NAME" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, factor, servers, workload, wall = sys.argv[1:9]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, rung=f"F{factor}", factor=int(factor), servers=int(servers),
           workload=workload, arm_kind="reactive", checkpoint_seed=0,
           policy_name="knative_network", wallclock_s=int(wall), representation=None)
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; a reactive arm that did not finish the trace "
                     "cannot define a rung (the 5/10-client rungs were excluded for exactly this)")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} queue_share={out['queue_share']:.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
