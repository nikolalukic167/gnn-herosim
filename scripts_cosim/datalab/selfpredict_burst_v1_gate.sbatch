#!/bin/bash
# selfpredict_burst_v1 -- the LIVE GATE: the self-predict rule vs gnnedge0 in the burst seat.
# docs/lineages/selfpredict_burst_v1.md.
#
# joint_burst_v2's uncapped gnnedge0 beat the 1-pass greedy in its own seat (-11.9 %, 13/13) and lost
# to the CD greedy (+12.5 %); selfpredict_bar_v1 made the self-predict rule the bar on the unburst
# rungs. This gate runs it on the SAME 16 burst environments (joint_burst_v1/selected.json, C40,
# 6 servers), FRESH in one OUT_DIR with every arm it is compared against, gnnedge0 included.
# Task table (304 tasks):
#     0- 15  peer_greedy_selfpredict_network   (per arrival)                [primary rule]
#    16- 31  peer_greedy_network               (immediate rule)
#    32- 47  peer_greedy_network_batch         (1-pass greedy, batched seat)
#    48- 63  peer_greedy_network_cd            (CD greedy, batched seat)
#    64- 79  knative_network                   (reactive)
#    80- 95  random_network
#    96-303  jb2 gnnedge0 UNCAPPED x 13 checkpoints x 16 env (seeds 1-5,7-10,12,14-16; 6/11/13 never trained)
# gnnedge0 is served exactly as joint_burst_v2 served it (masked_topo, peer-group batching, the
# cell's 16 s window, GNN_PREFIX_ALPHA_KEY=inf).
#
# Submit in blocks of <= 48 (MaxSubmit = 50 counts every array task):
#   sbatch --array=0-47 --export=ALL,SBU_TASK_OFFSET=<0|48|96|144|192|240|288> scripts_cosim/datalab/selfpredict_burst_v1_gate.sbatch
#
#SBATCH --job-name=sbu1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sbu1-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sbu1-gate-%A_%a.err
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
OUT_DIR="$GATE/results/selfpredict_burst_v1_gate"
SELECTION=simulation_data/joint_burst_v1/selected.json
mkdir -p "$OUT_DIR"
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
readarray -t ENVS < <(python3 - "$SELECTION" <<'PY2'
import json, sys
sel = json.load(open(sys.argv[1]))
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict {sel.get('verdict')!r}")
envs = sel["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: {len(envs)} environments, the registered design is 16")
for r, t, w in envs:
    if int(r) != 40:
        raise SystemExit(f"FAIL LOUD: environment {(r, t, w)} is not the 40-client rung")
    print(f"{t} {w}")
PY2
)
(( ${#ENVS[@]} == 16 )) || { echo "FAIL LOUD: environments read: ${#ENVS[@]}" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1

RULES=(peer_greedy_selfpredict_network peer_greedy_network peer_greedy_network_batch peer_greedy_network_cd knative_network random_network)
SEEDS=(1 2 3 4 5 7 8 9 10 12 14 15 16)
OFFSET=${SBU_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
NTASKS=304
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
SEED=0
if (( T < 96 )); then
  KIND=${RULES[$(( T / 16 ))]}; POLICY=$KIND; EI=$(( T % 16 ))
else
  R=$(( T - 96 )); EI=$(( R / 13 )); SEED=${SEEDS[$(( R % 13 ))]}; KIND=gnnedge0; POLICY=gnn
fi
read -r TOPO WINDOW <<< "${ENVS[$EI]}"
CELL="cc40s${TOPO}"; CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=burst_drainable_f4000_n50000; else WL_NAME=burst_drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" && -f "$WORKLOAD" ]] || { echo "FAIL LOUD: $CFG or $WORKLOAD missing" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" <<'PY2'
import json, sys
cfg, wl, w = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
if int(cfg["nodes"]["server_nodes"]["count"]) != 6 or int(cfg["nodes"]["client_nodes"]["count"]) != 40:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} is not the 6-server / 40-client cell")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0 or int(cfg["scheduler"]["batch_size"]) != 10:
    raise SystemExit("FAIL LOUD: scheduler block moved -- every policy constant is held fixed")
d = json.load(open(wl)); r = d["arrival_rescale"]
if (r["truncation"] or {}).get("skip_events", 0) != (0 if w == "w0" else int(w[1:]) * 100000):
    raise SystemExit(f"FAIL LOUD: {wl} is not window {w}")
if (r.get("lever") or {}).get("name") != "burst" or float(r["lever"]["span_after_s"]) != 0.0:
    raise SystemExit(f"FAIL LOUD: {wl} is not the burst lever")
PY2

unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF GNN_BATCH_TIMEOUT GNN_BATCH_SIZE GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP GNN_PREFIX_ALPHA_KEY LIVE_AUDIT_SNAPSHOT_PATH HEROSIM_ROLLOUT_SCORER HEROSIM_PG_EXCHANGE_SCALE HEROSIM_PG_ORACLE_NODES HEROSIM_PG_CD_PASSES HEROSIM_MAX_EVENTS HEROSIM_FORCED_PLACEMENTS || true
if [[ $POLICY == gnn ]]; then
  CK="models/joint-burst-v2-gnnedge0-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing" >&2; exit 1; }
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" gnnedge0 experiments/joint_burst_v2_split.json inf
  export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 GNN_PREFIX_ALPHA_KEY=inf
  CORPUS=jb2
elif [[ $POLICY == peer_greedy_network_batch || $POLICY == peer_greedy_network_cd ]]; then
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  CORPUS=none
else
  CORPUS=none
fi
ARMNAME="${CELL}__${WINDOW}__${KIND}_s${SEED}"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] topo=$TOPO window=$WINDOW arm=$KIND seed=$SEED -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$CORPUS" "$KIND" "$SEED" "$WALL" "$POLICY" <<'PY2'
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
           wallclock_s=int(wall))
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}; every arm in this design runs 50000")
if kind in ("peer_greedy_selfpredict_network", "peer_greedy_network") and int(c.get("pg_decisions") or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: {arm} pg_decisions={c.get('pg_decisions')!r} -- the rule's instrument is off")
if kind == "peer_greedy_selfpredict_network" and int(c.get("pg_lookahead_priced") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} priced no unarrived partner -- the self-predict term never fired")
if kind in ("peer_greedy_network_batch", "peer_greedy_network_cd") and int(c.get("pg_batches") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} decoded no batches")
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} wait={float(out['averageWaitTime'] or 0):.3f} wall={wall}s")
PY2
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
