#!/bin/bash
# batch_window_edge_v1 -- the batch window, screened then confirmed held-out.
# docs/lineages/batch_window_edge_v1.md. Bars: scripts_cosim/batch_window_edge_v1_read.py.
#
# 6 servers / 40 clients, unsaturated_edge_v1's primary rung and its 16 study environments. The
# lowest-seed study topology SCREENS the window (4 environments); the other three CONFIRM it
# (12 environments the selection never saw). Reactive and random do not batch, so their study
# runs are the reference unchanged; the 16 s arms are the main study's.
#
# Task table (576 tasks):
#     0-191    SCREEN   gnnedge0 at {2, 4, 8} s x 4 windows (screen topology) x 16 checkpoints
#   192-383    CONFIRM  gnnedge0 at the CHOSEN window x 12 held-out environments x 16
#   384-575    CONFIRM  mpoff    at the CHOSEN window x 12 held-out environments x 16
# Confirm tasks refuse to run until the screen has been read and the window written.
#
#   sbatch --array=0-47 --export=ALL,BW1_TASK_OFFSET=<base> scripts_cosim/datalab/batch_window_edge_v1.sbatch
#
#SBATCH --job-name=bw1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bw1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bw1-%A_%a.err
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
OUT_DIR="$GATE/results/bw_v1"
SELECTION=simulation_data/unsaturated_edge_v1/selected.json
WINDOW_SEL=simulation_data/batch_window_edge_v1/selected_window.json
mkdir -p "$OUT_DIR" simulation_data/batch_window_edge_v1

[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing -- unsaturated_edge_v1's M0 has not been read" >&2; exit 1; }
# The split is the registered rule in batch_window_edge_v1_read.split_environments.
readarray -t SCREEN_ENVS < <(python3 -c "
import json, sys; sys.path.insert(0, '.')
from scripts_cosim.batch_window_edge_v1_read import split_environments
s = split_environments(json.load(open('$SELECTION'))['rungs']['40']['topologies'])
for _r, t, w in s['screen']: print(t, w)")
readarray -t HELD_ENVS < <(python3 -c "
import json, sys; sys.path.insert(0, '.')
from scripts_cosim.batch_window_edge_v1_read import split_environments
s = split_environments(json.load(open('$SELECTION'))['rungs']['40']['topologies'])
for _r, t, w in s['held_out']: print(t, w)")
(( ${#SCREEN_ENVS[@]} == 4 && ${#HELD_ENVS[@]} == 12 )) || { echo "FAIL LOUD: split read ${#SCREEN_ENVS[@]}/${#HELD_ENVS[@]} environments" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1
# The window comes from the CELL CONFIG and nothing else; an exported GNN_BATCH_TIMEOUT that
# disagreed would fail loud in the orchestrator, and one that agreed would be redundant.
unset GNN_BATCH_TIMEOUT || true

SCREEN_BT=(2 4 8)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
OFFSET=${BW1_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if (( OFFSET != 0 )); then echo "[OFFSET] array ${SLURM_ARRAY_TASK_ID} + ${OFFSET} -> task $T"; fi
NTASKS=576
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi

if (( T < 192 )); then
  PHASE=screen; BT=${SCREEN_BT[$(( T / 64 ))]}; REM=$(( T % 64 )); EI=$(( REM / 16 )); SEED=${SEEDS[$(( REM % 16 ))]}
  CORPUS=be1670; KIND=gnnedge0
  read -r TOPO WINDOW <<< "${SCREEN_ENVS[$EI]}"
else
  PHASE=confirm
  [[ -f "$WINDOW_SEL" ]] || { echo "FAIL LOUD: $WINDOW_SEL missing -- the screen has not been read: "\
"python3 scripts_cosim/batch_window_edge_v1_gate_read.py screen ... --write-selection $WINDOW_SEL" >&2; exit 1; }
  BT=$(python3 -c "
import json; s = json.load(open('$WINDOW_SEL'))
if s.get('verdict') != 'WINDOW-SELECTED': raise SystemExit('FAIL LOUD: screen verdict is %r, nothing to confirm' % s.get('verdict'))
w = float(s['window_s']); print(int(w) if w == int(w) else w)")
  R=$(( T - 192 )); EI=$(( (R % 192) / 16 )); SEED=${SEEDS[$(( R % 16 ))]}
  if (( R < 192 )); then CORPUS=be1670; KIND=gnnedge0; else CORPUS=1670; KIND=mpoff; fi
  read -r TOPO WINDOW <<< "${HELD_ENVS[$EI]}"
fi

# --- the cell: the study's cc40 cell with ONLY scheduler.batch_timeout moved ---------------
BASECFG="$GATE/configs/cc40s${TOPO}.json"
CFG="$GATE/configs/cc40s${TOPO}_bt${BT}.json"
[[ -f "$BASECFG" ]] || { echo "FAIL LOUD: base cell missing: $BASECFG" >&2; exit 1; }
python3 - "$BASECFG" "$CFG" "$BT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, os, sys
base, out, bt, tid = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
cfg = json.load(open(base))
if int(cfg["nodes"]["server_nodes"]["count"]) != 6 or int(cfg["nodes"]["client_nodes"]["count"]) != 40:
    raise SystemExit(f"FAIL LOUD: {base} is not the 6-server / 40-client cell")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit(f"FAIL LOUD: {base} batch_timeout is {cfg['scheduler']['batch_timeout']}, not the 16.0 incumbent")
cfg["scheduler"]["batch_timeout"] = bt
if os.path.exists(out):
    have = json.load(open(out))
    if have != cfg:
        raise SystemExit(f"FAIL LOUD: {out} exists and differs from base+window -- not overwriting")
else:
    # deterministic content, atomic replace: concurrent tasks minting the same cell are benign
    tmp = f"{out}.tmp.{tid}"
    json.dump(cfg, open(tmp, "w"), indent=1)
    os.replace(tmp, out)
d = json.load(open(out))
if float(d["scheduler"]["batch_timeout"]) != bt:
    raise SystemExit("FAIL LOUD: minted cell does not carry the window")
PY

if [[ $WINDOW == w0 ]]; then WL_NAME=drainable_f4000_n50000; else WL_NAME=drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD" >&2; exit 1; }

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
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
        "mp_residual": False}
want["mp_bipartite_edge_conv"] = arm == "gnnedge0"
want["mp_bipartite_edge_attr_zero"] = arm == "gnnedge0"
if arm == "gnnedge0" and "mp_bipartite_aggr" in sc:
    want["mp_bipartite_aggr"] = "mean"
if arm != "gnnedge0" and "mp_bipartite_edge_conv" not in sc:
    want.pop("mp_bipartite_edge_conv"); want.pop("mp_bipartite_edge_attr_zero")
if "mp_platform_edges" in sc:
    want["mp_platform_edges"] = True
for k, v in want.items():
    if sc.get(k) != v:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}")
sha = hashlib.sha256(open("experiments/peer_only_v1_1670_split.json", "rb").read()).hexdigest()
got = (sc.get("split_artifact") or {}).get("sha256")
if got != sha:
    raise SystemExit(f"FAIL LOUD: corpus tag {corpus_tag} expects the peer_only_v1 1670 split (sha {sha[:12]}), sidecar carries {str(got)[:12]}")
PY
unset PARTIAL_STATE_CONTRACT GNN_MP_PLATFORM_EDGES_OFF GNN_DISABLE_MESSAGE_PASSING || true
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

ARMNAME="cc40s${TOPO}__${WINDOW}__bt${BT}__${CORPUS}_${KIND}_s${SEED}"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] $PHASE bt=${BT}s topo=$TOPO window=$WINDOW arm=$KIND seed=$SEED -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$TOPO" "$WINDOW" "$WL_NAME" "$CORPUS" "$KIND" "$SEED" "$WALL" "$BT" "$PHASE" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, topo, window, workload, corpus, kind, seed, wall, bt, phase = sys.argv[1:13]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=f"cc40s{topo}", topology=int(topo), window=window, clients=40, servers=6,
           rung="C40", workload=workload, corpus=corpus, arm_kind=kind, checkpoint_seed=int(seed),
           policy_name="gnn", wallclock_s=int(wall), batch_timeout_s=float(bt), phase=phase,
           representation="v3")
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
if env.get("GNN_MP_PLATFORM_EDGES_OFF") in ("1", "true", "yes"):
    raise SystemExit(f"FAIL LOUD: {kind} served with the GIN off")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} wait={float(out['averageWaitTime'] or 0):.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
