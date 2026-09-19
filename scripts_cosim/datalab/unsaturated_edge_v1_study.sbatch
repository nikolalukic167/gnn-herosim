#!/bin/bash
# unsaturated_edge_v1 -- E, the STUDY. docs/lineages/unsaturated_edge_v1.md.
#
# 16 (topology, window) environments PER CLIENT RUNG at 6 servers, the topologies chosen by M0's
# registered rule. Reactive is not re-run -- M0's arms are the reference and the read pulls them
# from results/ue_v1_screen.
#
# Task table (1344 tasks), most important arms first so a cut-short run still reads:
#     0-31      knative_network_ect  x {C40, C80} x 16 environments, seed 0
#    32-63      random_network       x {C40, C80} x 16 environments, seed 0
#    64-319     C40  be1670 gnnedge0 x 16 env x 16 checkpoints      (the headline arm)
#   320-575     C40  1670   mpoff    x 16 env x 16 checkpoints      (its corpus-matched twin)
#   576-831     C40  1670   peeronly x 16 env x 16 checkpoints
#   832-1087    C40  1670   gnn      x 16 env x 16 checkpoints      (GIN-sum bipartite)
#  1088-1343    C80  be1670 gnnedge0 x 16 env x 16 checkpoints      (the secondary rung)
#
# Submit in blocks of <= 48 (MaxSubmit = 50 counts every array task):
#   sbatch --array=0-47 --export=ALL,UE1_TASK_OFFSET=<base> scripts_cosim/datalab/unsaturated_edge_v1_study.sbatch
#
#SBATCH --job-name=ue1-study
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ue1-study-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ue1-study-%A_%a.err
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
OUT_DIR="$GATE/results/ue_v1"
SELECTION=simulation_data/unsaturated_edge_v1/selected.json
mkdir -p "$OUT_DIR"

# The study CANNOT run before M0 has been read: the environments are whatever the registered
# selection rule chose, and this file is the only place they exist.
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing -- run the M0 read first: "\
"python3 scripts_cosim/unsaturated_edge_v1_gate_read.py m0 $GATE/results/ue_v1_screen "\
"--write-selection $SELECTION" >&2; exit 1; }
readarray -t ENVS40 < <(python3 - "$SELECTION" 40 <<'PY'
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
)
readarray -t ENVS80 < <(python3 - "$SELECTION" 80 <<'PY'
import json, sys
sel = json.load(open(sys.argv[1])); nc = sys.argv[2]
envs = sel["rungs"][nc]["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: C{nc} has {len(envs)} environments, the registered design is 16")
for r, t, w in envs:
    print(f"{t} {w}")
PY
)
(( ${#ENVS40[@]} == 16 && ${#ENVS80[@]} == 16 )) || { echo "FAIL LOUD: environments read from $SELECTION: ${#ENVS40[@]}/${#ENVS80[@]}" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

BASELINES=(knative_network_ect random_network)
SPECS=("40 be1670 gnnedge0" "40 1670 mpoff" "40 1670 peeronly" "40 1670 gnn" "80 be1670 gnnedge0")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

OFFSET=${UE1_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if (( OFFSET != 0 )); then echo "[OFFSET] array ${SLURM_ARRAY_TASK_ID} + ${OFFSET} -> task $T"; fi
NB=$(( ${#BASELINES[@]} * 32 ))
NTASKS=$(( NB + ${#SPECS[@]} * 256 ))
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi

if (( T < NB )); then
  KIND=${BASELINES[$(( T / 32 ))]}; R=$(( T % 32 )); NC=$(( R < 16 ? 40 : 80 )); EI=$(( R % 16 ))
  CORPUS=none; SEED=0; POLICY=$KIND
else
  R=$(( T - NB )); SI=$(( R / 256 )); REM=$(( R % 256 )); EI=$(( REM / 16 )); SEED=${SEEDS[$(( REM % 16 ))]}
  read -r NC CORPUS KIND <<< "${SPECS[$SI]}"
  POLICY=gnn
fi
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
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
if (r["truncation"] or {}).get("skip_events", 0) != want:
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) > 0.005:
    raise SystemExit(f"FAIL LOUD: {wl} delivers {r['rate_after_per_s']:.4f} arrivals/s, the design holds 0.460393")
PY

if [[ $POLICY == gnn ]]; then
  case "$CORPUS" in
    1670)   CK="models/peer-only-v1-1670-${KIND}-lr2e3-seed${SEED}.pt" ;;
    be1670) CK="models/bipartite-edge-v1-1670-${KIND}-lr2e3-seed${SEED}.pt" ;;
    *) echo "FAIL LOUD: unknown corpus tag $CORPUS" >&2; exit 1 ;;
  esac
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  # The same pin unsaturated_scale_v2_study.sbatch applies, arm for arm: the flags that separate
  # these arms are weight-INVISIBLE and the model is BUILT from this sidecar.
  python3 - "$SC" "$KIND" "$CORPUS" <<'PY'
import hashlib, json, sys
sc, arm, corpus_tag = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
want = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
        "mp_residual": arm == "gnnres"}
BIP = ("gnnedge", "gnnedge0", "gnnedgesum")
want["mp_bipartite_edge_conv"] = arm in BIP
want["mp_bipartite_edge_attr_zero"] = arm in ("gnnedge0", "gnnedgesum")
if arm in BIP and "mp_bipartite_aggr" in sc:
    want["mp_bipartite_aggr"] = "sum" if arm == "gnnedgesum" else "mean"
if arm not in BIP and "mp_bipartite_edge_conv" not in sc:
    want.pop("mp_bipartite_edge_conv"); want.pop("mp_bipartite_edge_attr_zero")
if arm == "peeronly":
    want["mp_platform_edges"] = False
elif "mp_platform_edges" in sc:
    want["mp_platform_edges"] = True
for k, v in want.items():
    if sc.get(k) != v:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}")
SPLITS = {"1670": "experiments/peer_only_v1_1670_split.json",
          "be1670": "experiments/peer_only_v1_1670_split.json"}
sha = hashlib.sha256(open(SPLITS[corpus_tag], "rb").read()).hexdigest()
got = (sc.get("split_artifact") or {}).get("sha256")
if got != sha:
    raise SystemExit(f"FAIL LOUD: corpus tag {corpus_tag} expects split {SPLITS[corpus_tag]} "
                     f"(sha {sha[:12]}), sidecar carries {str(got)[:12]}")
PY
  unset PARTIAL_STATE_CONTRACT GNN_MP_PLATFORM_EDGES_OFF || true
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
  ARMNAME="${CELL}__${WINDOW}__${CORPUS}_${KIND}_s${SEED}"
else
  unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF || true
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
fi

SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] clients=$NC topo=$TOPO window=$WINDOW corpus=$CORPUS arm=$KIND seed=$SEED -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$CORPUS" "$KIND" "$SEED" "$WALL" "$POLICY" "$NC" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, topo, window, workload, corpus, kind, seed, wall, policy, nc = sys.argv[1:14]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=int(nc), servers=6,
           rung=f"C{nc}", workload=workload, corpus=corpus, arm_kind=kind,
           checkpoint_seed=int(seed), policy_name=policy, wallclock_s=int(wall),
           representation=(None if policy != "gnn" else "v3"))
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
po = env.get("GNN_MP_PLATFORM_EDGES_OFF")
if kind == "peeronly" and po not in (None, "1", "true", "yes"):
    raise SystemExit(f"FAIL LOUD: peeronly served with GNN_MP_PLATFORM_EDGES_OFF={po!r}")
if policy == "gnn" and kind != "peeronly" and po in ("1", "true", "yes"):
    raise SystemExit(f"FAIL LOUD: {kind} served with the GIN off")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} wait={float(out['averageWaitTime'] or 0):.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
