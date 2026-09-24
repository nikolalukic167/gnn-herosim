#!/bin/bash
# lookahead_mp_v1 P0 -- is there live headroom in pricing partners that have NOT arrived?
# docs/lineages/lookahead_mp_v1.md.
#
# The physics charges each task the exchange to EVERY partner (waiting for an unarrived one,
# then paying against where it lands); the peer-greedy rule prices unarrived partners at 0.
# P0 serves, per arrival on the rule's own stack, on the SAME 16 (topology, window)
# environments per client rung that unsaturated_edge_v1 selected (6 servers):
#     rule      : peer_greedy_network            [the bar; also writes its per-task node map]
#     lookahead : peer_greedy_lookahead_network  [two-step hand estimate of an unarrived node]
#     oracle    : peer_greedy_oracle_network     [the node the partner ran on in the rule run]
#
# Two phases, because the oracle replays the rule run of the SAME environment:
#   LA_PHASE=1  64 tasks: per rung (C40 = T<32, C80 = T>=32), 2 arms x 16 envs (rule, lookahead)
#   LA_PHASE=2  32 tasks: per rung (C40 = T<16, C80 = T>=16), oracle x 16 envs -- after phase 1
#
# Submit (MaxSubmit = 50 counts every array task):
#   J1=$(sbatch --parsable --array=0-47 --export=ALL,LA_PHASE=1,LA_TASK_OFFSET=0  scripts_cosim/datalab/lookahead_mp_v1_p0.sbatch)
#   J2=$(sbatch --parsable --array=0-15 --export=ALL,LA_PHASE=1,LA_TASK_OFFSET=48 scripts_cosim/datalab/lookahead_mp_v1_p0.sbatch)
#   sbatch --dependency=afterok:${J1}:${J2} --array=0-31 --export=ALL,LA_PHASE=2,LA_TASK_OFFSET=0 scripts_cosim/datalab/lookahead_mp_v1_p0.sbatch
#
#SBATCH --job-name=la1-p0
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/la1-p0-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/la1-p0-%A_%a.err
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
OUT_DIR="$GATE/results/lookahead_mp_v1_p0"
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

PHASE=${LA_PHASE:?FAIL LOUD: set LA_PHASE=1 or 2}
OFFSET=${LA_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if [[ $PHASE == 1 ]]; then
  ARMS=(peer_greedy_network peer_greedy_lookahead_network)
  NTASKS=64; PER_RUNG=32
elif [[ $PHASE == 2 ]]; then
  ARMS=(peer_greedy_oracle_network)
  NTASKS=32; PER_RUNG=16
else
  echo "FAIL LOUD: LA_PHASE=$PHASE" >&2; exit 1
fi
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared phase-$PHASE tasks"; exit 0; fi
NC=$(( T < PER_RUNG ? 40 : 80 )); R=$(( T % PER_RUNG )); KIND=${ARMS[$(( R / 16 ))]}; EI=$(( R % 16 ))
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
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
if (r["truncation"] or {}).get("skip_events", 0) != want:
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, the design holds 0.460393")
PY

unset GNN_MODEL_PATH GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP HEROSIM_ROLLOUT_SCORER HEROSIM_PG_EXCHANGE_SCALE HEROSIM_PG_ORACLE_NODES HEROSIM_MAX_EVENTS HEROSIM_FORCED_PLACEMENTS || true
NODES="$OUT_DIR/${CELL}__${WINDOW}__peer_greedy_network_s0.nodes.json"
if [[ $KIND == peer_greedy_oracle_network ]]; then
  [[ -f "$NODES" ]] || { echo "FAIL LOUD: oracle needs the paired rule run's node map $NODES" >&2; exit 1; }
  export HEROSIM_PG_ORACLE_NODES="$NODES"
fi
ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS phase $PHASE] clients=$NC topo=$TOPO window=$WINDOW arm=$KIND -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy "$KIND" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$KIND" "$WALL" "$NC" "$NODES" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, kind, wall, nc, nodes_path = sys.argv[1:12]
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
if int(c.get("pg_decisions") or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: {arm} pg_decisions={c.get('pg_decisions')!r} -- the rule's instrument is off")
if kind != "peer_greedy_network" and int(c.get("pg_lookahead_priced") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} priced no unarrived partner -- the lookahead term never fired")
if kind == "peer_greedy_oracle_network" and int(c.get("pg_lookahead_blind") or 0) != 0:
    raise SystemExit(f"FAIL LOUD: {arm} left {c.get('pg_lookahead_blind')} partners unpriced; the oracle is never blind")
if kind == "peer_greedy_network":
    tr = st.get("taskResults") or []
    if not st.get("taskResultsIncluded", True) or len(tr) != 50000:
        raise SystemExit(f"FAIL LOUD: rule run carries {len(tr)} taskResults; the oracle needs all 50000")
    json.dump({int(t["taskId"]): str(t["executionNode"]) for t in tr}, open(nodes_path + ".partial", "w"))
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} exch/task={float(out['totalPeerExchangeTime'])/50000:.3f} "
      f"priced={c.get('pg_lookahead_priced')} blind={c.get('pg_lookahead_blind')} wall={wall}s")
PY
if [[ $KIND == peer_greedy_network ]]; then mv "$NODES.partial" "$NODES"; fi
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
