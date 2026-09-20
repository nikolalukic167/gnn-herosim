#!/bin/bash
# env_lever_v1 -- one lever, one step. docs/lineages/{burst_groups_v1,payload_scale_v1,
# backbone_sparsity_v1}.md. Inputs are minted by env_lever_v1_mint.sbatch first.
#
#   EL_LEVER  in burst | pk0.1 | pk10 | p04 | bw250
#   EL_STEP   screen : reactive knative_network on 12 candidate topologies x 4 windows at C40
#                      (48 tasks; T = topo_index*4 + window_index)
#             study  : on the 4 topologies the lever's screen selected
#                      (simulation_data/env_lever_v1/selected_<lever>.json):
#                        0-15    random_network
#                       16-31    peer_greedy_network
#                       32-47    peer_greedy_network_batch
#                       48-303   be1670 gnnedge0 x 16 env x 16 checkpoints
#                      304-559   1670 mpoff x 16 env x 16 checkpoints   (burst only)
#   EL_TASK_OFFSET  block base (MaxSubmit = 50 counts every array task)
#
#   sbatch --array=0-47 --export=ALL,EL_LEVER=burst,EL_STEP=screen scripts_cosim/datalab/env_lever_v1.sbatch
#   sbatch --array=0-47 --export=ALL,EL_LEVER=burst,EL_STEP=study,EL_TASK_OFFSET=48 scripts_cosim/datalab/env_lever_v1.sbatch
#
#SBATCH --job-name=el1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/el1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/el1-%A_%a.err
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

LEVER=${EL_LEVER:?FAIL LOUD: EL_LEVER unset}
STEP=${EL_STEP:?FAIL LOUD: EL_STEP unset}
case "$LEVER" in burst|pk0.1|pk10|p04|bw250) ;; *) echo "FAIL LOUD: unknown lever $LEVER" >&2; exit 1 ;; esac
GATE=simulation_data/peer_affinity_live_gate
OUT_DIR="$GATE/results/el_v1/$LEVER"
mkdir -p "$OUT_DIR"
NC=40
TOPOS=(9001 9002 9003 9005 9101 9102 9103 9104 9105 9106 9107 9108)
WINDOWS=(w0 w1 w2 w3)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

OFFSET=${EL_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if (( OFFSET != 0 )); then echo "[OFFSET] array ${SLURM_ARRAY_TASK_ID} + ${OFFSET} -> task $T"; fi

# the ue1 study's serving environment, verbatim
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF GNN_BATCH_TIMEOUT GNN_BATCH_SIZE GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP || true

CORPUS=none; SEED=0
if [[ $STEP == screen ]]; then
  NTASKS=48
  if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
  TOPO=${TOPOS[$(( T / 4 ))]}; WINDOW=${WINDOWS[$(( T % 4 ))]}
  KIND=reactive; POLICY=knative_network
elif [[ $STEP == study ]]; then
  SELECTION="simulation_data/env_lever_v1/selected_${LEVER}.json"
  [[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing -- read the lever's screen first" >&2; exit 1; }
  readarray -t ENVS < <(python3 - "$SELECTION" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1]))
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict is {sel.get('verdict')!r}, not DESIGN-READY")
envs = sel["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: {len(envs)} environments, the registered design is 16")
for r, t, w in envs:
    if int(r) != 40:
        raise SystemExit(f"FAIL LOUD: environment {(r, t, w)} is not C40")
    print(f"{t} {w}")
PY
)
  (( ${#ENVS[@]} == 16 )) || { echo "FAIL LOUD: environments read: ${#ENVS[@]}" >&2; exit 1; }
  if [[ $LEVER == burst ]]; then NTASKS=560; else NTASKS=304; fi
  if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
  if (( T < 48 )); then
    BASE=(random_network peer_greedy_network peer_greedy_network_batch)
    KIND=${BASE[$(( T / 16 ))]}; POLICY=$KIND; EI=$(( T % 16 ))
  else
    R=$(( T - 48 )); SI=$(( R / 256 )); REM=$(( R % 256 )); EI=$(( REM / 16 )); SEED=${SEEDS[$(( REM % 16 ))]}
    if (( SI == 0 )); then CORPUS=be1670; KIND=gnnedge0; else CORPUS=1670; KIND=mpoff; fi
    POLICY=gnn
  fi
  read -r TOPO WINDOW <<< "${ENVS[$EI]}"
else
  echo "FAIL LOUD: unknown step $STEP" >&2; exit 1
fi

# lever-specific inputs (exactly one of the two moves)
case "$LEVER" in
  burst|pk0.1|pk10) CELL="cc${NC}s${TOPO}"; WLP="${LEVER}_" ;;
  p04|bw250)        CELL="cc${NC}s${TOPO}_${LEVER}"; WLP="" ;;
esac
CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME="${WLP}drainable_f4000_n50000"; else WL_NAME="${WLP}drainable_${WINDOW}_n50000"; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG (mint first)" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD (mint first)" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" "$LEVER" <<'PY'
import json, sys
cfg, wl, w, lever = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
if int(cfg["nodes"]["server_nodes"]["count"]) != 6 or int(cfg["nodes"]["client_nodes"]["count"]) != 40:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} is not the 6-server / 40-client rung")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0 or int(cfg["scheduler"]["batch_size"]) != 10:
    raise SystemExit("FAIL LOUD: scheduler block moved -- every policy constant is held fixed")
net = cfg["network"]
want_p = 0.4 if lever == "p04" else 0.6
want_bw = 250.0 if lever == "bw250" else 1000.0
if float(net["topology"]["connection_probability"]) != want_p or float(net["backbone"]["bandwidth_mbps"]) != want_bw:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} network is p={net['topology']['connection_probability']} "
                     f"bw={net['backbone']['bandwidth_mbps']}, lever {lever} wants p={want_p} bw={want_bw}")
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
if (r["truncation"] or {}).get("skip_events", 0) != want:
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, the design holds 0.460393")
lv = (r.get("lever") or {}).get("name")
want_lv = lever if lever in ("burst", "pk0.1", "pk10") else None
if lv != want_lv:
    raise SystemExit(f"FAIL LOUD: {wl} carries lever {lv!r}, task wants {want_lv!r}")
if lever == "burst" and float(r["lever"]["span_after_s"]) != 0.0:
    raise SystemExit("FAIL LOUD: burst workload's group span after is not 0")
if lever.startswith("pk") and abs(float(r["lever"]["k"]) - float(lever[2:])) > 1e-9:
    raise SystemExit(f"FAIL LOUD: {wl} payload scale {r['lever']['k']} != {lever[2:]}")
PY

if [[ $POLICY == gnn ]]; then
  case "$CORPUS" in
    1670)   CK="models/peer-only-v1-1670-${KIND}-lr2e3-seed${SEED}.pt" ;;
    be1670) CK="models/bipartite-edge-v1-1670-${KIND}-lr2e3-seed${SEED}.pt" ;;
  esac
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  python3 - "$SC" "$KIND" "$CORPUS" <<'PY'
import hashlib, json, sys
sc, arm, corpus_tag = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
want = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22, "mp_residual": False}
if arm == "gnnedge0":
    want.update(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
    if "mp_bipartite_aggr" in sc:
        want["mp_bipartite_aggr"] = "mean"
elif "mp_bipartite_edge_conv" in sc:
    want.update(mp_bipartite_edge_conv=False, mp_bipartite_edge_attr_zero=False)
if "mp_platform_edges" in sc:
    want["mp_platform_edges"] = True
for k, v in want.items():
    if sc.get(k) != v:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}")
sha = hashlib.sha256(open("experiments/peer_only_v1_1670_split.json", "rb").read()).hexdigest()
got = (sc.get("split_artifact") or {}).get("sha256")
if got != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {str(got)[:12]} != corpus split {sha[:12]}")
PY
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
  ARMNAME="${CELL}__${WINDOW}__${CORPUS}_${KIND}_s${SEED}"
elif [[ $POLICY == peer_greedy_network_batch ]]; then
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
else
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
fi

SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] lever=$LEVER step=$STEP topo=$TOPO window=$WINDOW corpus=$CORPUS arm=$KIND seed=$SEED -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$CORPUS" "$KIND" "$SEED" "$WALL" "$POLICY" "$LEVER" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, topo, window, workload, corpus, kind, seed, wall, policy, lever = sys.argv[1:14]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=40, servers=6, rung="C40",
           lever=lever, workload=workload, corpus=corpus, arm_kind=kind, checkpoint_seed=int(seed),
           policy_name=policy, wallclock_s=int(wall), representation=(None if policy != "gnn" else "v3"))
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
if kind.endswith("greedy_network") or kind == "peer_greedy_network_batch":
    if "pg_decisions" not in c:
        raise SystemExit(f"FAIL LOUD: {arm} carries no pg_* counters -- the rule's instrument is off")
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} wait={float(out['averageWaitTime'] or 0):.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
