#!/bin/bash
# unsaturated_scale_v2 -- M0, the SATURATION SCREEN. docs/lineages/unsaturated_scale_v2.md.
#
# Reactive `knative_network` alone on all 12 candidate topologies x 4 arrival windows = 48
# candidate environments. Its only outputs are (a) which environments are readable rungs and
# (b) the 4 study topologies, both decided by `read_m0` / `select_topologies` in
# scripts_cosim/unsaturated_scale_v2_read.py -- rules committed with their tests before this ran.
#
# This is BLOCKING. unsaturated_scale_v1's finding was that the whole 80-server record had been
# read against a baseline at 99 % queue share, so an environment enters v2 only after reactive
# has been measured healthy on it. Running the learned arms first and screening afterwards
# would be the same defect with extra steps.
#
#   sbatch --array=0-47 scripts_cosim/datalab/unsaturated_scale_v2_screen.sbatch
#
#SBATCH --job-name=us2-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/us2-screen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/us2-screen-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-47

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
OUT_DIR="$GATE/results/us_v2_screen"
mkdir -p "$OUT_DIR"
# M_TOPOLOGY_CANDIDATES and M_WINDOWS, in the reader's order. 9004 is absent: it hangs on
# every policy (peer_only_v1).
TOPOS=(9001 9002 9003 9005 9101 9102 9103 9104 9105 9106 9107 9108)
WINDOWS=(w0 w1 w2 w3)

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}
TOPO=${TOPOS[$(( T / 4 ))]}; W=${WINDOWS[$(( T % 4 ))]}
CELL="cs80s${TOPO}"
CFG="$GATE/configs/${CELL}.json"
if [[ $W == w0 ]]; then WL_NAME=drainable_f4000_n50000; else WL_NAME=drainable_${W}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG (run unsaturated_scale_v2_mint.sbatch)" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD (run unsaturated_scale_v2_mint.sbatch)" >&2; exit 1; }

python3 - "$CFG" "$WORKLOAD" "$W" <<'PY'
import json, sys
cfg, wl, w = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
if int(cfg["nodes"]["server_nodes"]["count"]) != 80:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares "
                     f"{cfg['nodes']['server_nodes']['count']} servers, the rung is 80")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
d = json.load(open(wl)); r = d["arrival_rescale"]; t = r["truncation"]
want_skip = 0 if w == "w0" else int(w[1:]) * 100000
if t.get("skip_events", 0) != want_skip:
    raise SystemExit(f"FAIL LOUD: {wl} skip_events={t.get('skip_events', 0)}, {w} expects {want_skip}")
if len(d["events"]) != 50000:
    raise SystemExit(f"FAIL LOUD: {wl} has {len(d['events'])} events, the design is n50000")
# The window axis is arrival PATTERN, not load: every window must deliver w0's rate.
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, "
                     "the design holds 0.460393")
PY

ARM="${CELL}__${W}__reactive_s0"
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] topo=$TOPO window=$W cell=$CELL workload=$WL_NAME -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARM}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy knative_network --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$TOPO" "$W" "$WL_NAME" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, topo, window, workload, wall = sys.argv[1:9]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, topology=int(topo), window=window, servers=80,
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
                     "cannot define an environment (the 5/10-client rungs were excluded for "
                     "exactly this)")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} queue_share={out['queue_share']:.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
