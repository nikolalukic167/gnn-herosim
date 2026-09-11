#!/bin/bash
# peer_affinity_v1 stage 3 -- serving-knob probe on the production trace.
# The registered gate measured: both learned arms halve the peer-exchange service and give
# it all back to concurrency (effective parallel channels 15.3 -> 5.4), because the masked
# decoder's per-node load resets every batch, so the capacity cap is intra-batch and blind
# to the queue already standing on a node. Two serving knobs attack that, both default OFF:
#   GNN_PREFIX_LOAD_SEED=<s>            seed the decode's per-node load with the node's
#                                       queue IMBALANCE (depth above the least-loaded
#                                       candidate node) in demand units, x s
#   GNN_PREFIX_CONCURRENCY_PENALTY=<p>  soft score penalty proportional to a candidate's
#                                       share of the batch's worst candidate queue
# Arms (one checkpoint, gnn lr2e3 seed 1, so the ONLY difference is the knob):
#   0 baseline (both off -- reproduces the gate arm)   1 load seed 1.0
#   2 load seed 1.0 + penalty 1.0                      3 penalty 1.0 alone
#   4 load seed 4.0                                    5 load seed 0.25
#   6 platform cap 1        7 platform cap 2        8 platform cap 3
# Reference: knative_network on the same trace is 2.0131e10 with 15.3 channels.
#
#SBATCH --job-name=pa-s3-knob
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-s3-knob-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-s3-knob-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-8

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
WORKLOAD="$GATE/workloads/workload-150-150-peer_p2_x200.json"
CFG="$GATE/configs/cell_s7901.json"
OUT_DIR="$GATE/results/serving_knob_probe"
mkdir -p "$OUT_DIR" logs
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_MODEL_PATH=models/peer-affinity-v1-t1b-gnn-lr2e3-seed1.pt
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
[[ -f "$GNN_MODEL_PATH" ]] || { echo "FAIL LOUD: checkpoint missing" >&2; exit 1; }

case ${SLURM_ARRAY_TASK_ID} in
  0) ARM=baseline ;;
  1) ARM=seed1;    export GNN_PREFIX_LOAD_SEED=1.0 ;;
  2) ARM=seed1_pen1; export GNN_PREFIX_LOAD_SEED=1.0 GNN_PREFIX_CONCURRENCY_PENALTY=1.0 ;;
  3) ARM=pen1;     export GNN_PREFIX_CONCURRENCY_PENALTY=1.0 ;;
  4) ARM=seed4;    export GNN_PREFIX_LOAD_SEED=4.0 ;;
  5) ARM=seed025;  export GNN_PREFIX_LOAD_SEED=0.25 ;;
  # Round 2 (2026-09-11): rounds 1's node-level knobs both made it worse -- they scattered
  # peer groups ACROSS nodes (comms/task 2.82 -> 5.4, back to Knative's level) and bought no
  # concurrency at all (5.37 -> 4.96-5.81). The decoder's only unconstrained axis was the
  # platform: a batch may legally stack on one replica, and 12.4 % of live batches placed
  # their whole plan on a single platform. GNN_PREFIX_PLATFORM_CAP caps that. On the 3k smoke
  # it is the first knob to help (+3.5 % at cap 2, channels 3.67 -> 4.04).
  6) ARM=platcap1; export GNN_PREFIX_PLATFORM_CAP=1 ;;
  7) ARM=platcap2; export GNN_PREFIX_PLATFORM_CAP=2 ;;
  8) ARM=platcap3; export GNN_PREFIX_PLATFORM_CAP=3 ;;
  *) echo "FAIL LOUD: unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK ${SLURM_ARRAY_TASK_ID}] arm=$ARM load_seed=${GNN_PREFIX_LOAD_SEED:-off} penalty=${GNN_PREFIX_CONCURRENCY_PENALTY:-off}"
if [[ -f "$SUMMARY" ]]; then echo "[TASK] summary exists, not re-running"; exit 0; fi

OUT="$OUT_DIR/${ARM}.json"
${HEROSIM_PY:-pipenv run python3} src/executesimulation.py \
  --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$OUT.partial"
python3 - "$OUT.partial" "$SUMMARY" "$ARM" <<'PY'
import json, sys
raw, summary, arm = sys.argv[1:4]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in (
    "policy", "num_tasks", "total_rtt", "averageElapsedTime", "totalPeerExchangeTime",
    "totalPeerRendezvousWait", "endTime", "schedulerCounters", "averageQueueTime",
    "averageWaitTime", "averageExecutionTime", "averageCommunicationsTime",
    "averageColdStartTime", "averagePullTime", "averageInitializationTime",
    "averageOccupation", "coldStartProportion", "nodeCacheHitsProportion",
    "taskCacheHitsProportion", "unusedPlatforms")}
_se = st.get("scaleEvents"); s["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
s["env"] = {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v}
s["arm"] = arm
n, end = s["num_tasks"], s["endTime"]
svc = (s["averageCommunicationsTime"] or 0) + (s["averageExecutionTime"] or 0)
s["throughput_tasks_per_s"] = n / end if end else None
s["effective_parallel_channels"] = (n / end) * svc if end else None
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env",)}))
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$OUT.partial"
