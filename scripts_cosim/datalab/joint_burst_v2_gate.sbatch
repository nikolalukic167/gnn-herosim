#!/bin/bash
# joint_burst_v2 -- the gate under bursts on the SAME 16 study environments as joint_burst_v1,
# with checkpoints trained UNCAPPED (alpha=inf) on the corpus that keeps the loaded/cap-
# infeasible states v1's corpus rejected, served uncapped (the greedy's own seat), plus the
# coordinate-descent greedy as the honest bar. docs/lineages/joint_burst_v2.md.
#
# Task table (848 tasks), read from simulation_data/joint_burst_v1/selected.json (same screen):
#     0-15     random_network
#    16-31     peer_greedy_network          (immediate rule)
#    32-47     peer_greedy_network_batch    (batched greedy; reused from the v1 screen)
#    48-63     peer_greedy_network_cd       (coordinate-descent greedy, the honest bar)
#    64-319    jb2 gnnedge0 UNCAPPED  x 16 env x 16 seeds   (this lineage's checkpoints)
#   320-575    jb2 mpoff    UNCAPPED  x 16 env x 16 seeds   (its twin)
#   576-831    jb1 gnnedge0 served UNCAPPED (same weights, alpha=inf) x 16 x 16 -- isolates
#              corpus from serving cap: does relaxing the CAP ALONE move the v1 checkpoint?
#   832-847    peer_greedy_network_batch with HEROSIM_PG_EXCHANGE_SCALE=2 (disclosed probe,
#              16 env -- does MORE co-location than the rule read live-better?)
# Reactive and the batched greedy come from the v1 screen (identical cells). Every learned arm
# is served identically (masked_topo, peer-group batching, the cell's 16 s window).
#
#   sbatch --array=0-XX --export=ALL,JB2_TASK_OFFSET=<base> scripts_cosim/datalab/joint_burst_v2_gate.sbatch
#
#SBATCH --job-name=jb2-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb2-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb2-gate-%A_%a.err
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
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
GATE=simulation_data/peer_affinity_live_gate
OUT_DIR="$GATE/results/jb_v2/gate"
SELECTION=simulation_data/joint_burst_v1/selected.json
mkdir -p "$OUT_DIR"
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
readarray -t ENVS < <(python3 - "$SELECTION" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1]))
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict {sel.get('verdict')!r}")
envs = sel["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: {len(envs)} environments, the registered design is 16")
for r, t, w in envs:
    print(f"{t} {w}")
PY
)
(( ${#ENVS[@]} == 16 )) || { echo "FAIL LOUD: environments read: ${#ENVS[@]}" >&2; exit 1; }
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
OFFSET=${JB2_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
NTASKS=848
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF GNN_BATCH_TIMEOUT GNN_BATCH_SIZE GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP LIVE_AUDIT_SNAPSHOT_PATH || true
CORPUS=none; SEED=0; EXTRA_ENV=()
if (( T < 48 )); then
  BASE=(random_network peer_greedy_network peer_greedy_network_batch); KIND=${BASE[$(( T / 16 ))]}; POLICY=$KIND; EI=$(( T % 16 ))
elif (( T < 64 )); then
  KIND=peer_greedy_network_cd; POLICY=$KIND; EI=$(( T - 48 ))
elif (( T < 832 )); then
  R=$(( T - 64 )); SI=$(( R / 256 )); REM=$(( R % 256 )); EI=$(( REM / 16 )); SEED=${SEEDS[$(( REM % 16 ))]}
  case $SI in
    0) CORPUS=jb2; KIND=gnnedge0 ;;
    1) CORPUS=jb2; KIND=mpoff ;;
    2) CORPUS=jb1uncapped; KIND=gnnedge0 ;;
  esac
  POLICY=gnn
else
  R=$(( T - 832 )); EI=$R; KIND=peer_greedy_network_batch; POLICY=$KIND
  EXTRA_ENV=(HEROSIM_PG_EXCHANGE_SCALE=2.0)
fi
read -r TOPO WINDOW <<< "${ENVS[$EI]}"
CELL="cc40s${TOPO}"; CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=burst_drainable_f4000_n50000; else WL_NAME=burst_drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" && -f "$WORKLOAD" ]] || { echo "FAIL LOUD: $CFG or $WORKLOAD missing" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" <<'PY'
import json, sys
cfg, wl, w = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
assert int(cfg["nodes"]["server_nodes"]["count"]) == 6 and int(cfg["nodes"]["client_nodes"]["count"]) == 40
assert float(cfg["scheduler"]["batch_timeout"]) == 16.0 and int(cfg["scheduler"]["batch_size"]) == 10
d = json.load(open(wl)); r = d["arrival_rescale"]
assert (r["truncation"] or {}).get("skip_events", 0) == (0 if w == "w0" else int(w[1:]) * 100000)
assert (r.get("lever") or {}).get("name") == "burst" and float(r["lever"]["span_after_s"]) == 0.0
PY
if [[ $POLICY == gnn ]]; then
  case "$CORPUS" in
    jb2)         CK="models/joint-burst-v2-${KIND}-lr2e3-seed${SEED}.pt"; SPLIT=experiments/joint_burst_v2_split.json ;;
    jb1uncapped) CK="models/joint-burst-v1-uncapped-${KIND}-lr2e3-seed${SEED}.pt"; SPLIT=experiments/joint_burst_v1_split.json ;;
  esac
  [[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing" >&2; exit 1; }
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "$KIND" "$SPLIT" inf
  export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 GNN_PREFIX_ALPHA_KEY=inf
  if [[ $KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
  ARMNAME="${CELL}__${WINDOW}__${CORPUS}_${KIND}_s${SEED}"
elif [[ $POLICY == peer_greedy_network_cd ]]; then
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
elif [[ ${#EXTRA_ENV[@]} -gt 0 ]]; then
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  export "${EXTRA_ENV[@]}"
  ARMNAME="${CELL}__${WINDOW}__${KIND}_x2scale_s0"
elif [[ $POLICY == peer_greedy_network_batch ]]; then
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
  # the v1 screen already ran this arm on every study cell: reuse it
  S="$GATE/results/jb_v1/screen/${CELL}__${WINDOW}__batched_s0.summary.json"
  if [[ -e "$S" && ! -e "$OUT_DIR/${ARMNAME}.summary.json" ]]; then
    python3 scripts_cosim/joint_burst_v2_reuse.py "$S" "$OUT_DIR/${ARMNAME}.summary.json" "$ARMNAME"
    echo "[TASK $T] reused the v1 screen's batched arm"; exit 0
  fi
else
  ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
fi
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] topo=$TOPO window=$WINDOW corpus=$CORPUS arm=$KIND seed=$SEED -> $SUMMARY"
if [[ -e "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))
python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$CORPUS" "$KIND" "$SEED" "$WALL" "$POLICY" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, corpus, kind, seed, wall, policy = sys.argv[1:13]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
                              "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=40, servers=6, rung="C40", lever="burst",
           workload=workload, corpus=corpus, arm_kind=kind, checkpoint_seed=int(seed), policy_name=policy,
           wallclock_s=int(wall), representation=(None if policy != "gnn" else "v3"))
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} wait={float(out['averageWaitTime'] or 0):.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
