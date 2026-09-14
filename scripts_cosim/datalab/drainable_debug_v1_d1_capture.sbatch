#!/bin/bash
# drainable_debug_v1 -- D1, step 1: capture live batch states at the drainable rung.
# Three sources on one trace and one cell, so every source sees identical physics and the
# only difference is which policy was driving when the state was captured:
#   task 0 = knative_network_batch (the reactive reference)
#   task 1 = gnn            (T1b lr2e3 seed 1)
#   task 2 = mpoff          (the same checkpoint with message passing off)
#   sbatch --array=0-2 scripts_cosim/datalab/drainable_debug_v1_d1_capture.sbatch
#
# The reactive source is the BATCH policy, not the per-arrival one: only a batch scheduler
# emits a peer-group-aligned snapshot, and `check_aligned_peer_group` rejects anything else,
# so a per-arrival capture would yield zero datasets. Its placement rule is unchanged --
# `knative_network_batch` batches collection only and still places per task by shortest queue
# (src/policy/knative_network_batch/scheduler.py:20-29).
#
#SBATCH --job-name=dbg-d1-cap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-cap-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-cap-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

G=simulation_data/peer_affinity_live_gate
D1=simulation_data/drainable_debug_v1/d1
WORKLOAD="${WORKLOAD:-$G/workloads/drainable_f4000_n50000.json}"
CFG="${CFG:-$G/configs/cell_s7901_f4000_pg16.json}"
mkdir -p "$D1/snapshots" "$D1/capture_summaries" logs
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

T=${SLURM_ARRAY_TASK_ID}
case $T in
  0) SRC=knb ;;
  1) SRC=gnn ;;
  2) SRC=mpoff ;;
  *) echo "FAIL LOUD: task $T is not one of the three sources" >&2; exit 1 ;;
esac
SNAP="$D1/snapshots/${SRC}.jsonl"
SUMMARY="$D1/capture_summaries/${SRC}.summary.json"
if [[ -f "$SUMMARY" && -s "$SNAP" ]]; then echo "[TASK $T] done already: $SUMMARY"; exit 0; fi
echo "[TASK $T] source=$SRC -> $SNAP"
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# physics + gate contract, identical for every source and identical to the parent's S1 gate
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
# Stride 10 is one snapshot per peer group at group size 10, so a captured batch is aligned by
# construction; MAX caps the file. MIN_CANDIDATES=1 keeps states where a task had one legal
# replica -- 25 % of tasks on the landed trace -- because excluding them would sample only the
# states where placement was a choice, which is the question, not a filter.
export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP"
export LIVE_AUDIT_STRIDE="${LIVE_AUDIT_STRIDE:-10}"
export LIVE_AUDIT_MAX_SNAPSHOTS="${LIVE_AUDIT_MAX_SNAPSHOTS:-120}"
export LIVE_AUDIT_MIN_BATCH_SIZE=1 LIVE_AUDIT_MIN_CANDIDATES=1
rm -f "$SNAP"

if [[ $SRC == knb ]]; then
  POLICY=knative_network_batch
  # The window must match the cell's GNN window or the sources see different batch policies.
  BT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['scheduler']['batch_timeout'])" "$CFG")
  export KNATIVE_BATCH_SIZE=10 KNATIVE_BATCH_TIMEOUT="$BT" KNATIVE_BATCH_BY_PEER_GROUP=1
  export KNATIVE_BATCH_POLL_INTERVAL=$(python3 -c "print(float('$BT')/20.0)")
  echo "[BATCH] KNATIVE_BATCH_TIMEOUT=$BT poll=$KNATIVE_BATCH_POLL_INTERVAL (from the cell config)"
else
  POLICY=gnn
  # Each arm has its OWN checkpoint, exactly as the stage-3 gate builds the path. `mpoff` is
  # not the gnn checkpoint served with message passing switched off: the sidecar records the
  # flag the weights were TRAINED under and `prefix_serving._adopt_or_verify_env` refuses the
  # mismatch as a train/serve error rather than an ablation (which is the correct behaviour --
  # it is the guard added after the 2026-09-06 MP-OFF loader hole).
  CK="models/peer-affinity-v1-t1b-${SRC}-lr2e3-seed1.pt"
  [[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing" >&2; exit 1; }
  export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  # Uncapped: GNN_PREFIX_PLATFORM_CAP=1 deadlocks 3/16 seeds in a drainable regime
  # (drainable_regime_v1 Amendment 2) and is not a serving default.
  [[ $SRC == mpoff ]] && export GNN_DISABLE_MESSAGE_PASSING=1
fi

RAW="$D1/capture_summaries/${SRC}.raw.json"
${HEROSIM_PY:-pipenv run python3} src/executesimulation.py \
  --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
python3 - "$RAW" "$SUMMARY" "$SNAP" "$SRC" <<'PY'
import json, sys
raw, summary, snap, src = sys.argv[1:5]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in ("policy", "num_tasks", "total_rtt", "averageElapsedTime",
                            "averageQueueTime", "averageWaitTime", "totalPeerExchangeTime",
                            "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
s["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
s.update({"source": src, "snapshots": sum(1 for _ in open(snap)),
          "queue_length": d.get("queue_length"),
          "env": {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v},
          "code": (d.get("run_provenance") or {}).get("code")})
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env", "code")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
[[ -s "$SNAP" ]] || { echo "FAIL LOUD: no snapshots captured in $SNAP" >&2; exit 1; }
echo "=== task $T done: $(wc -l < "$SNAP") snapshots ==="
