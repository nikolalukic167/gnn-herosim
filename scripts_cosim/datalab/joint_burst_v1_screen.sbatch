#!/bin/bash
# joint_burst_v1 -- the burst pool's screen. docs/lineages/joint_burst_v1.md.
#
# Two serving paths must finish on a study cell: reactive `knative_network` (admissibility,
# queue share <= 0.80) and the batch path (`peer_greedy_network_batch`, the learned arms'
# stack), because under bursts the batch path enters the starved-client spin on topologies
# reactive survives (burst_groups_v1 Amendment 2). 28 candidate topologies x 4 windows x 2 arms
# = 224 tasks; T = arm*112 + topo_index*4 + window_index. Existing summaries are skipped.
#
#   sbatch --array=0-47 --export=ALL,JB1_TASK_OFFSET=<base> scripts_cosim/datalab/joint_burst_v1_screen.sbatch
#
#SBATCH --job-name=jb1-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb1-screen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb1-screen-%A_%a.err
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
OUT_DIR="$GATE/results/jb_v1/screen"
mkdir -p "$OUT_DIR"
TOPOS=(9001 9002 9003 9005 9101 9102 9103 9104 9105 9106 9107 9108 $(seq 9109 9124))
WINDOWS=(w0 w1 w2 w3)
ARMS=(reactive batched)
OFFSET=${JB1_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
NTASKS=$(( ${#ARMS[@]} * ${#TOPOS[@]} * 4 ))
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
AI=$(( T / (${#TOPOS[@]} * 4) )); R=$(( T % (${#TOPOS[@]} * 4) )); TOPO=${TOPOS[$(( R / 4 ))]}; WINDOW=${WINDOWS[$(( R % 4 ))]}
KIND=${ARMS[$AI]}

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING GNN_MP_PLATFORM_EDGES_OFF GNN_BATCH_TIMEOUT GNN_BATCH_SIZE GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP LIVE_AUDIT_SNAPSHOT_PATH || true
if [[ $KIND == reactive ]]; then POLICY=knative_network; else POLICY=peer_greedy_network_batch; export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1; fi

CELL="cc40s${TOPO}"; CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=burst_drainable_f4000_n50000; else WL_NAME=burst_drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG (mint first)" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: trace missing: $WORKLOAD" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" <<'PY'
import json, sys
cfg, wl, w = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
assert int(cfg["nodes"]["server_nodes"]["count"]) == 6 and int(cfg["nodes"]["client_nodes"]["count"]) == 40
assert float(cfg["scheduler"]["batch_timeout"]) == 16.0 and int(cfg["scheduler"]["batch_size"]) == 10
d = json.load(open(wl)); r = d["arrival_rescale"]
want = 0 if w == "w0" else int(w[1:]) * 100000
assert (r["truncation"] or {}).get("skip_events", 0) == want, (wl, w)
assert abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) <= 0.005
assert (r.get("lever") or {}).get("name") == "burst" and float(r["lever"]["span_after_s"]) == 0.0
PY
ARMNAME="${CELL}__${WINDOW}__${KIND}_s0"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] topo=$TOPO window=$WINDOW arm=$KIND ($POLICY) -> $SUMMARY"
if [[ -e "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))
python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$KIND" "$WALL" "$POLICY" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, kind, wall, policy = sys.argv[1:11]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
                              "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=40, servers=6, rung="C40", lever="burst",
           workload=workload, corpus="none", arm_kind=kind, checkpoint_seed=0, policy_name=policy, wallclock_s=int(wall))
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
n = out.get("num_tasks")
if n is None or int(n) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={n!r}")
e, q = float(out["averageElapsedTime"]), float(out["averageQueueTime"])
out["queue_share"] = q / e if e > 0 else None
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={e:.3f} queue={q:.3f} share={out['queue_share']:.3f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
