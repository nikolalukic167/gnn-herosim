#!/bin/bash
# selfpredict_bar_v1 -- the LIVE GATE that sizes the new hand-rule bar. docs/lineages/selfpredict_bar_v1.md.
#
# lookahead_mp_v1 closed with peer_greedy_selfpredict_network beating peer_greedy_network -6.2/-8.5 %,
# but no reactive, random or CD greedy arm was in that gate. This gate runs the self-predict rule
# against all four, FRESH in one OUT_DIR so every comparison is paired at identical code/env, on
# the SAME 16 (topology, window) environments per client rung that unsaturated_edge_v1 selected.
# Task table (192 tasks): per rung (C40 = T<96, C80 = T>=96), 6 arms x 16 envs:
#     selfpredict: peer_greedy_selfpredict_network  (per arrival)             [primary]
#     rule       : peer_greedy_network              (per arrival)             [replication]
#     cd         : peer_greedy_network_cd           (CD greedy, batched)      [the ceiling]
#     reactive   : knative_network                  (reactive Knative -- peer_greedy_live_v1's baseline)
#     ect        : knative_network_ect              (disclosed: what rollout_imitation_v1 called "reactive")
#     random     : random_network
#
# Submit in blocks of <= 48 (MaxSubmit = 50 counts every array task):
#   sbatch --array=0-47 --export=ALL,SB1_TASK_OFFSET=0   scripts_cosim/datalab/selfpredict_bar_v1_gate.sbatch
#   sbatch --array=0-47 --export=ALL,SB1_TASK_OFFSET=48  scripts_cosim/datalab/selfpredict_bar_v1_gate.sbatch
#   sbatch --array=0-47 --export=ALL,SB1_TASK_OFFSET=96  scripts_cosim/datalab/selfpredict_bar_v1_gate.sbatch
#   sbatch --array=0-47 --export=ALL,SB1_TASK_OFFSET=144 scripts_cosim/datalab/selfpredict_bar_v1_gate.sbatch
#
#SBATCH --job-name=sb1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sb1-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sb1-gate-%A_%a.err
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
python3 -c "import sys;print('[INTERP]',sys.executable)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
OUT_DIR="$GATE/results/selfpredict_bar_v1_gate"
SELECTION=simulation_data/unsaturated_edge_v1/selected.json
mkdir -p "$OUT_DIR"
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
read_envs() {
python3 - "$SELECTION" "$1" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1])); nc = sys.argv[2]
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict is {sel.get('verdict')!r}, not DESIGN-READY")
envs = sel["rungs"][nc]["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: C{nc} has {len(envs)} environments, the registered design is 16")
for r, t, w in envs:
    if int(r) != int(nc):
        raise SystemExit(f"FAIL LOUD: environment {(r, t, w)} listed under rung C{nc}")
    print(f"{t} {w}")
PY
}
readarray -t ENVS40 < <(read_envs 40)
readarray -t ENVS80 < <(read_envs 80)
(( ${#ENVS40[@]} == 16 && ${#ENVS80[@]} == 16 )) || { echo "FAIL LOUD: environments read: ${#ENVS40[@]}/${#ENVS80[@]}" >&2; exit 1; }

# The ue1 study's serving environment, verbatim (same as rollout_imitation_v1_gate.sbatch).
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

ARMS=(peer_greedy_selfpredict_network peer_greedy_network peer_greedy_network_cd knative_network knative_network_ect random_network)
OFFSET=${SB1_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
NTASKS=192
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
NC=$(( T < 96 ? 40 : 80 )); R=$(( T % 96 )); KIND=${ARMS[$(( R / 16 ))]}; EI=$(( R % 16 ))
if (( NC == 40 )); then read -r TOPO WINDOW <<< "${ENVS40[$EI]}"; else read -r TOPO WINDOW <<< "${ENVS80[$EI]}"; fi
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
if float(cfg["scheduler"]["batch_timeout"]) != 16.0 or int(cfg["scheduler"]["batch_size"]) != 10:
    raise SystemExit("FAIL LOUD: scheduler block moved -- every policy constant is held fixed")
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
if (r["truncation"] or {}).get("skip_events", 0) != want:
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, the design holds 0.460393")
PY

unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF GNN_BATCH_TIMEOUT GNN_BATCH_SIZE HEROSIM_ROLLOUT_SCORER HEROSIM_PG_EXCHANGE_SCALE HEROSIM_PG_ORACLE_NODES HEROSIM_MAX_EVENTS HEROSIM_FORCED_PLACEMENTS || true
if [[ $KIND == peer_greedy_network_cd ]]; then
  # the coordinate-descent greedy refines over a peer-group batch: the learned arms' serving stack
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
else
  unset GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP || true
fi
ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] clients=$NC topo=$TOPO window=$WINDOW arm=$KIND -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy "$KIND" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$KIND" "$WALL" "$NC" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, kind, wall, nc = sys.argv[1:11]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=int(nc), servers=6,
           rung=f"C{nc}", workload=workload, arm_kind=kind, policy_name=kind, wallclock_s=int(wall))
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
if kind in ("peer_greedy_selfpredict_network", "peer_greedy_network") and int(c.get("pg_decisions") or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: {arm} pg_decisions={c.get('pg_decisions')!r} -- the rule's instrument is off")
if kind == "peer_greedy_selfpredict_network" and int(c.get("pg_lookahead_priced") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} priced no unarrived partner -- the self-predict term never fired")
if kind == "peer_greedy_network_cd" and int(c.get("pg_batches") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} decoded no batches")
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} exch/task={float(out['totalPeerExchangeTime'] or 0)/50000:.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
