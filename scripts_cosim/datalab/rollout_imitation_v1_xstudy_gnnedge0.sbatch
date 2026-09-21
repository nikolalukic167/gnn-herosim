#!/bin/bash
# rollout_imitation_v1 CROSS-STUDY arm A: joint_burst_v2 `gnnedge0` served on the UNBURST
# unsaturated_edge_v1 environments (the rollout gate's study), so gnnedge0 and the rollout learned
# arm sit in ONE paired table. Same batched learned seat gnnedge0 was gated in (masked_topo,
# peer-group batching, uncapped alpha=inf); the burst-trained checkpoint is now served on the
# unburst arrival structure -- a disclosed train/serve transfer. docs/lineages/rollout_imitation_v1.md.
#
# 512 tasks: 2 rungs (C40 = T<256, C80) x 16 envs x 16 seed-checkpoints. Output lands in the SAME dir
# as the rollout gate; the seed suffix (_s1.._s16) means the rollout reader (which matches only _s0)
# ignores these, and the cross-study reader aggregates them per env (median over seeds).
#
#   sbatch --array=0-47 --export=ALL,XG_TASK_OFFSET=<0,48,...,480> scripts_cosim/datalab/rollout_imitation_v1_xstudy_gnnedge0.sbatch
#
#SBATCH --job-name=ri1-xg
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ri1-xg-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ri1-xg-%A_%a.err
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
OUT_DIR="$GATE/results/rollout_imitation_v1_gate"
SELECTION=simulation_data/unsaturated_edge_v1/selected.json
mkdir -p "$OUT_DIR"
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
read_envs() {
python3 - "$SELECTION" "$1" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1])); nc = sys.argv[2]
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict is {sel.get('verdict')!r}")
envs = sel["rungs"][nc]["environments"]
if len(envs) != 16:
    raise SystemExit(f"FAIL LOUD: C{nc} has {len(envs)} environments")
for r, t, w in envs:
    print(f"{t} {w}")
PY
}
readarray -t ENVS40 < <(read_envs 40)
readarray -t ENVS80 < <(read_envs 80)
(( ${#ENVS40[@]} == 16 && ${#ENVS80[@]} == 16 )) || { echo "FAIL LOUD: envs ${#ENVS40[@]}/${#ENVS80[@]}" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
OFFSET=${XG_TASK_OFFSET:-0}
T=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
NTASKS=512
if (( T >= NTASKS )); then echo "[TASK $T] beyond $NTASKS"; exit 0; fi
NC=$(( T < 256 ? 40 : 80 )); R=$(( T % 256 )); EI=$(( R / 16 )); SEED=${SEEDS[$(( R % 16 ))]}
if (( NC == 40 )); then read -r TOPO WINDOW <<< "${ENVS40[$EI]}"; else read -r TOPO WINDOW <<< "${ENVS80[$EI]}"; fi
CELL="cc${NC}s${TOPO}"; CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=drainable_f4000_n50000; else WL_NAME=drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" && -f "$WORKLOAD" ]] || { echo "FAIL LOUD: $CFG or $WORKLOAD missing" >&2; exit 1; }
python3 - "$CFG" "$WORKLOAD" "$WINDOW" "$NC" <<'PY'
import json, sys
cfg, wl, w, nc = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], int(sys.argv[4])
assert int(cfg["nodes"]["server_nodes"]["count"]) == 6 and int(cfg["nodes"]["client_nodes"]["count"]) == nc
assert float(cfg["scheduler"]["batch_timeout"]) == 16.0 and int(cfg["scheduler"]["batch_size"]) == 10
d = json.load(open(wl)); r = d["arrival_rescale"]
assert (r["truncation"] or {}).get("skip_events", 0) == (0 if w == "w0" else int(w[1:]) * 100000)
assert abs(float(r["rate_after_per_s"]) / 0.460393 - 1.0) <= 0.005
PY

CK="models/joint-burst-v2-gnnedge0-lr2e3-seed${SEED}.pt"
[[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing" >&2; exit 1; }
python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" gnnedge0 experiments/joint_burst_v2_split.json inf
export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 GNN_PREFIX_ALPHA_KEY=inf
ARMNAME="${CELL}__${WINDOW}__gnnedge0unburst_s${SEED}"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T/$NTASKS] clients=$NC topo=$TOPO window=$WINDOW gnnedge0 seed=$SEED -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$RAW"
WALL=$(( $(date +%s) - START ))
python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$TOPO" "$WINDOW" "$WL_NAME" "$SEED" "$WALL" "$NC" <<'PY'
import json, sys
raw, summary, arm, cell, topo, window, workload, seed, wall, nc = sys.argv[1:11]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime",
       "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
out.update(arm=arm, cell=cell, topology=int(topo), window=window, clients=int(nc), servers=6,
           rung=f"C{nc}", workload=workload, corpus="jb2", arm_kind="gnnedge0_unburst",
           checkpoint_seed=int(seed), policy_name="gnn", wallclock_s=int(wall), representation="v3")
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
