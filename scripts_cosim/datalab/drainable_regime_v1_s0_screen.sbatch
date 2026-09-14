#!/bin/bash
# drainable_regime_v1 -- S0, step 2: the live characterisation screen.
# 4 rungs x 2 reactive policies on cell_s7901. No model, no training, no checkpoint.
#   task T: rung = FACTORS[T / 2], policy = (T % 2 == 0 ? knative_network : knative_network_batch)
# The batch arm carries KNATIVE_BATCH_BY_PEER_GROUP=1 and LIVE_AUDIT snapshots -- it is the
# corpus bridge's own source, so S2 (queue column) and S4 (aligned batches) are read off it.
# The plain arm exists for one descriptive reading: if two reactive policies are
# indistinguishable on a rung, placement has no room there whatever S1 says.
#   sbatch --dependency=afterok:<traces job> --array=0-7 <this>
#
#SBATCH --job-name=drain-s0
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/drain-s0-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/drain-s0-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

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
S0="${DRAIN_S0_DIR:-simulation_data/drainable_regime_v1/s0}"
N=50000
# Amendment 1 (2026-09-14): overridable so the R1b/R1c rungs submit against the same script.
read -r -a FACTORS <<< "${DRAIN_FACTORS:-300 1000 2000 4000}"
T=${SLURM_ARRAY_TASK_ID}
F=${FACTORS[$(( T / 2 ))]}
WORKLOAD=$GATE/workloads/drainable_f${F}_n${N}.json
CFG=$GATE/configs/cell_s7901.json
mkdir -p "$S0/results" "$S0/snapshots" logs

[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: rung trace missing: $WORKLOAD (run the traces job)" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# physics + gate contract, verbatim from peer_affinity_v1_stage3_live_gate.sbatch. The whole
# point of the screen is that only the arrival rate differs from the landed gate, so nothing
# in this block may be touched. HEROSIM_REPLICA_PLATFORM_TYPES stays unset = the full cluster.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu

if [[ $(( T % 2 )) -eq 0 ]]; then
  POLICY=knative_network; ARM="knative_network_f${F}"
else
  POLICY=knative_network_batch; ARM="knative_network_batch_f${F}"
  # Amendment 2 (2026-09-14): the batch window is a POLICY time constant, not physics. Holding
  # it at 0.02 s while stretching arrivals up to 4,000x shrinks it by the same factor relative
  # to the workload, and S0 pass 1 measured the consequence: mean batch size 1.01-1.09 on every
  # rung, 0 aligned batches everywhere including the overloaded control, where the landed gate
  # at 2,659 arrivals/s aligns 149/150. DRAIN_SCALE_BATCH_TIMEOUT=1 scales it with the rung so
  # the batching policy is comparable across rungs; the loop still exits the moment the group is
  # complete, so a longer window costs nothing when the group arrives sooner.
  BT=0.02
  if [[ "${DRAIN_SCALE_BATCH_TIMEOUT:-0}" == "1" ]]; then
    BT=$(python3 -c "print(0.02 * $F)")
  fi
  echo "[BATCH] KNATIVE_BATCH_TIMEOUT=$BT (scale=${DRAIN_SCALE_BATCH_TIMEOUT:-0}, factor=$F)"
  export KNATIVE_BATCH_SIZE=10 KNATIVE_BATCH_TIMEOUT="$BT" KNATIVE_BATCH_BY_PEER_GROUP=1
  export LIVE_AUDIT_SNAPSHOT_PATH="$S0/snapshots/f${F}.jsonl"
  export LIVE_AUDIT_STRIDE=200 LIVE_AUDIT_MAX_SNAPSHOTS=150
  export LIVE_AUDIT_MIN_BATCH_SIZE=1 LIVE_AUDIT_MIN_CANDIDATES=1
  rm -f "$LIVE_AUDIT_SNAPSHOT_PATH"
fi
OUT="$S0/results/${ARM}.json"
SUMMARY="$S0/results/${ARM}.summary.json"
echo "[TASK $T] rung=x${F} policy=$POLICY arm=$ARM -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] summary exists, not re-running"; exit 0; fi

${HEROSIM_PY:-pipenv run python3} src/executesimulation.py \
  --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$OUT.partial"

python3 - "$OUT.partial" "$SUMMARY" "$ARM" "$POLICY" "$WORKLOAD" "$F" <<'PY'
import json, sys
raw, summary, arm, policy, workload, factor = sys.argv[1:7]
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
# the rung's own identity, so the read never has to infer the rate from a filename
r = (json.load(open(workload)) or {}).get("arrival_rescale") or {}
s["factor"] = float(factor)
s["arrival_span_s"] = r.get("span_after_s")
s["arrival_rate_per_s"] = r.get("rate_after_per_s")
s["env"] = {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v}
s["code"] = (d.get("run_provenance") or {}).get("code")
s["arm"] = arm; s["policy_name"] = policy; s["source"] = "raw json (deleted after summary)"
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env", "code", "schedulerCounters")}))
print("[COUNTERS]", json.dumps(s.get("schedulerCounters")))
print("[PROVENANCE env]", json.dumps(s["env"]))
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$OUT.partial"
if [[ -n "${LIVE_AUDIT_SNAPSHOT_PATH:-}" ]]; then
  echo "[AUDIT] $(wc -l < "$LIVE_AUDIT_SNAPSHOT_PATH") snapshots -> $LIVE_AUDIT_SNAPSHOT_PATH"
fi
echo "=== task $T done ==="
