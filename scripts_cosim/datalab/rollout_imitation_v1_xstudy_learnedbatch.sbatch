#!/bin/bash
# rollout_imitation_v1 CROSS-STUDY arm B: the rollout LEARNED scorer served in the BATCHED seat on
# the BURST joint_burst_v2 environments, so the rollout arm and gnnedge0 sit in ONE paired table in
# the seat gnnedge0 wins in. Same 16 C40 burst envs, same batched serving (masked_topo, peer-group
# batching) as jb2; the decision is the learned MLP over the rule's terms instead of the greedy sum
# or the GNN. Output lands in the jb2 gate dir to pair against its existing arms.
# docs/lineages/rollout_imitation_v1.md.
#
# 16 tasks (C40 x 16 burst envs, deterministic s0):
#   sbatch --array=0-15 --export=ALL scripts_cosim/datalab/rollout_imitation_v1_xstudy_learnedbatch.sbatch
#
#SBATCH --job-name=ri1-xlb
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ri1-xlb-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ri1-xlb-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-15

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
SCORER=${RI1_SCORER:-models/rollout_imitation_v1/scorer.pt}
mkdir -p "$OUT_DIR"
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
[[ -f "$SCORER" && -f "${SCORER%.pt}.contract.json" ]] || { echo "FAIL LOUD: scorer $SCORER or sidecar missing" >&2; exit 1; }
readarray -t ENVS < <(python3 - "$SELECTION" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1]))
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict {sel.get('verdict')!r}")
envs = sel["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: {len(envs)} environments, design is 16")
for r, t, w in envs:
    print(f"{t} {w}")
PY
)
(( ${#ENVS[@]} == 16 )) || { echo "FAIL LOUD: envs ${#ENVS[@]}" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1
T=$SLURM_ARRAY_TASK_ID
read -r TOPO WINDOW <<< "${ENVS[$T]}"
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

export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
export HEROSIM_ROLLOUT_SCORER="$SCORER"
ARMNAME="${CELL}__${WINDOW}__learnedbatch_s0"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/16] topo=$TOPO window=$WINDOW learnedbatch -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy peer_greedy_learned_network_batch --output "$RAW"
WALL=$(( $(date +%s) - START ))
python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$WALL" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, wall = sys.argv[1:9]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime",
       "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=40, servers=6, rung="C40",
           lever="burst", workload=workload, corpus="none", arm_kind="learnedbatch", checkpoint_seed=0,
           policy_name="peer_greedy_learned_network_batch", wallclock_s=int(wall), representation=None)
c = out.get("schedulerCounters") or {}
if "pg_decisions" not in c and int(c.get("pg_batches") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} carries no pg_* counters")
if int(c.get("pg_batches") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {arm} decoded no batches")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} total_rtt={float(out['total_rtt'] or 0):.1f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
