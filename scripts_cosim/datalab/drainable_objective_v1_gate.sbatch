#!/bin/bash
# drainable_objective_v1 -- Phase C: the LIVE GATE that closes the lineage (CLAUDE.md
# rule 6). Same cell, trace, window and physics as drainable_regime_v1's S1 and
# drainable_serving_config_v1's config E, so the T1b numbers at this cell (gnn 53.45 s,
# mpoff 51.32 s, knative_network 25.95 s) are the standing baseline and are NOT re-run.
#
# Amendment 1 (2026-09-14): the V = 0 control is the EXISTING T1b lr2e3 checkpoints, which
# were gated at this exact cell (gnn 53.45 s, mpoff 51.32 s, knative_network 25.95 s) and
# are not re-run -- repricing the backlog on the measured clock leaves the argmin plan
# unmoved, so a separately trained V = 0 arm would be a copy.
#
# 34 arms:
#   task  0        knative_network        (reactive, per arrival)
#   task  1        knative_network_batch  (reactive, 10-task window)
#   tasks 2-17     v1 gnn    seeds 1-16   (SHAPED label: the treatment)
#   tasks 18-33    v1 mpoff  seeds 1-16
#
# UNCAPPED: GNN_PREFIX_PLATFORM_CAP=1 deadlocks 3/16 seeds in a drainable regime
# (hard-stops.md, drainable_regime_v1 Amendment 2). Q = 100, the rung D3 showed is real.
#
# Two seeds per learned arm keep their raw per-task records (KEEP_RAW) for bar C1, the
# behaviour control: if the V = 1 arms did not stop taking replicas deeper than the
# shallowest, a latency win is not evidence about the label.
#
#   sbatch --array=0-33%8 scripts_cosim/datalab/drainable_objective_v1_gate.sbatch
#
#SBATCH --job-name=dobj-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-33

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
WORKLOAD="${WORKLOAD:-$GATE/workloads/drainable_f4000_n50000.json}"
CFG="${CFG:-$GATE/configs/cell_s7901_f4000_pg16.json}"
OUT_DIR="${OUT_DIR:-$GATE/results/dobj_f4000_pg16}"
mkdir -p "$OUT_DIR" logs

[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# Physics + gate contract, identical for every arm.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
# NOT exported: HEROSIM_BACKLOG_DRAIN_TABLE. The table reprices a SEEDED co-sim backlog;
# the live trace has no seeded backlog, it builds its own queues from real arrivals.
# Exporting it here would change the gate's physics and make this a different cluster from
# every drainable_* reading in the record.
echo "[KNOBS] Q='${HEROSIM_QUEUE_LENGTH:-<unset: default 100>}' CAP='${GNN_PREFIX_PLATFORM_CAP:-<unset: uncapped>}' DRAIN_TABLE=<deliberately unset: live queues are not seeded>"

T=${SLURM_ARRAY_TASK_ID}
if [[ $T -eq 0 ]]; then
  POLICY=knative_network; ARM=knative_network
elif [[ $T -eq 1 ]]; then
  POLICY=knative_network_batch; ARM=knative_network_batch
  export KNATIVE_BATCH_SIZE=10 KNATIVE_BATCH_BY_PEER_GROUP=1
else
  POLICY=gnn
  I=$(( T - 2 ))
  GROUP=$(( I / 16 )); SEED=$(( I % 16 + 1 ))
  case $GROUP in
    0) TAG=v1; ARMNAME=gnn ;;
    1) TAG=v1; ARMNAME=mpoff ;;
    *) echo "FAIL LOUD: task $T is beyond the 34 declared arms" >&2; exit 1 ;;
  esac
  ARM="${TAG}_${ARMNAME}_s${SEED}"
  CK="models/drainable-objective-v1-${TAG}-${ARMNAME}-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: checkpoint missing: $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: sidecar missing for $CK" >&2; exit 1; }
  # The sidecar must say which label this arm was fitted to, and it must be the one this
  # task's name claims. Serving the V = 1 checkpoint in the V = 0 slot would produce a
  # perfectly clean gate of the wrong thing.
  python3 - "$SC" "$TAG" "$ARMNAME" <<'PY'
import json, sys
sc, tag, arm = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
want = {"v1": "rtt_drift:1"}[tag]
got = sc.get("label_objective") or "rtt"
if not got.startswith(want):
    raise SystemExit(f"FAIL LOUD: sidecar label_objective={got!r}, slot {tag} expects {want}")
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing={sc.get('disable_message_passing')!r} for arm {arm}")
print(f"[sidecar] {tag}/{arm}: label={got}")
PY
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
fi

# Bar C1 needs per-task records; two seeds per learned arm carry them (~350 MB each with
# the queue snapshots, so not all 64).
if [[ "$ARM" == *_s1 || "$ARM" == *_s2 ]]; then
  export SIM_FORCE_FULL_STATS=1
  export GNN_CAPTURE_DATASET_STATE=1
  KEEP_RAW=1
else
  KEEP_RAW=0
fi

OUT="$OUT_DIR/${ARM}.json"
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] policy=$POLICY arm=$ARM keep_raw=$KEEP_RAW -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] summary exists, not re-running"; exit 0; fi

python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$OUT.partial"

python3 - "$OUT.partial" "$SUMMARY" "$ARM" "$POLICY" <<'PY'
import json, os, sys
raw, summary, arm, policy = sys.argv[1:5]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in (
    "policy", "num_tasks", "total_rtt", "averageElapsedTime", "totalPeerExchangeTime",
    "totalPeerRendezvousWait", "endTime", "schedulerCounters",
    "averageQueueTime", "averageWaitTime", "averageExecutionTime", "averageCommunicationsTime",
    "averageColdStartTime", "averagePullTime", "averageInitializationTime", "averageOccupation",
    "coldStartProportion", "nodeCacheHitsProportion", "taskCacheHitsProportion",
    "localCommunicationsProportion", "penaltyProportion", "unusedNodes", "unusedPlatforms")}
_se = st.get("scaleEvents")
s["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
s["env"] = {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v}
s["code"] = (d.get("run_provenance") or {}).get("code")
s["arm"] = arm; s["policy_name"] = policy
s["queue_length"] = d.get("queue_length")
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env", "code")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"
if [[ "$KEEP_RAW" == "1" ]]; then
  mv "$OUT.partial" "$OUT"
  echo "[RAW KEPT] $OUT ($(du -h "$OUT" | cut -f1))"
else
  rm -f "$OUT.partial"
fi
echo "=== task $T done: $ARM ==="
