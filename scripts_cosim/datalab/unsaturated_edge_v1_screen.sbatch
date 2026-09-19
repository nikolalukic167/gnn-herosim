#!/bin/bash
# unsaturated_edge_v1 -- M0, the SATURATION SCREEN. docs/lineages/unsaturated_edge_v1.md.
#
# Reactive `knative_network` alone on 12 candidate topologies x 4 arrival windows x 2 client
# rungs (40, 80) = 96 candidate environments. Its only outputs are (a) which environments are
# readable rungs and (b) the 4 study topologies PER RUNG, both decided by `read_m0` /
# `select_topologies` in scripts_cosim/unsaturated_edge_v1_read.py -- rules committed with their
# tests before this is read. Same discipline as unsaturated_scale_v2's screen.
#
# Task table: T = rung_index*48 + topo_index*4 + window_index.
#   sbatch --array=0-47 scripts_cosim/datalab/unsaturated_edge_v1_screen.sbatch
#   sbatch --array=0-47 --export=ALL,UE1_TASK_OFFSET=48 scripts_cosim/datalab/unsaturated_edge_v1_screen.sbatch
#
#SBATCH --job-name=ue1-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ue1-screen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ue1-screen-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
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
OUT_DIR="$GATE/results/ue_v1_screen"
mkdir -p "$OUT_DIR"

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF || true

RUNGS=(40 80)
TOPOS=(9001 9002 9003 9005 9101 9102 9103 9104 9105 9106 9107 9108)
WINDOWS=(w0 w1 w2 w3)

OFFSET=${UE1_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if (( OFFSET != 0 )); then echo "[OFFSET] array ${SLURM_ARRAY_TASK_ID} + ${OFFSET} -> task $T"; fi
NTASKS=$(( ${#RUNGS[@]} * ${#TOPOS[@]} * ${#WINDOWS[@]} ))
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
NC=${RUNGS[$(( T / 48 ))]}; R=$(( T % 48 )); TOPO=${TOPOS[$(( R / 4 ))]}; WINDOW=${WINDOWS[$(( R % 4 ))]}
CELL="cc${NC}s${TOPO}"
CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=drainable_f4000_n50000; else WL_NAME=drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" "$NC" <<'PY'
import json, sys
cfg, wl, w, nc = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], int(sys.argv[4])
if int(cfg["nodes"]["server_nodes"]["count"]) != 6:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} is not the 6-server rung")
if int(cfg["nodes"]["client_nodes"]["count"]) != nc:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} is not the {nc}-client rung")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
if (r["truncation"] or {}).get("skip_events", 0) != want:
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, the design holds 0.460393")
PY

ARMNAME="${CELL}__${WINDOW}__reactive_s0"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] clients=$NC topo=$TOPO window=$WINDOW -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy knative_network --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$NC" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, topo, window, workload, nc, wall = sys.argv[1:10]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=int(nc), servers=6,
           rung=f"C{nc}", workload=workload, corpus="none", arm_kind="reactive",
           checkpoint_seed=0, policy_name="knative_network", wallclock_s=int(wall))
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
if e <= 0:
    raise SystemExit("FAIL LOUD: non-positive elapsed")
out["queue_share"] = q / e
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} share={q/e:.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
