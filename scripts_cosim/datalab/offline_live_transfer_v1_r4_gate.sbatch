#!/bin/bash
# offline_live_transfer_v1 -- R4, the LIVE GATE that closes the lineage (rule 6).
#
# R1 measured that the offline score carries no per-checkpoint signal (pooled-z rho =
# -0.030, p = 0.772, n = 96). This gate measures what that costs, using weights that
# already exist: every training run saves BOTH the offline-selected checkpoint and the
# last epoch. The selected arms are already gated (results/dobj_f4000_pg16); this runs
# their last-epoch twins at the identical cell, trace, window and knobs.
#
#   SELECTOR-WORKS      selected beats last-epoch: median lower, >= 12/16, p < 0.05, BOTH arms
#   SELECTOR-IS-NOISE   neither direction clears
#   SELECTOR-IS-HARMFUL last-epoch beats selected in both arms at p < 0.05
#
# 32 arms:
#   tasks  0-15   v1 gnn   seeds 1-16, LAST EPOCH
#   tasks 16-31   v1 mpoff seeds 1-16, LAST EPOCH
#
# Declared in advance: the SELECTED v1 gnn seed 3 checkpoint deterministically livelocks
# the simulator (drainable_objective_v1 Phase C), so that arm is n = 15 against a >= 12
# bar. Whether the last-epoch seed 3 hangs is unknown and is an outcome of this gate, not
# a reason to exclude it. --time is deliberately SHORT so a livelocked arm dies in minutes
# instead of holding a core for 12 h: a completed arm at this cell takes ~3 min.
#
#   sbatch --array=0-31%8 scripts_cosim/datalab/offline_live_transfer_v1_r4_gate.sbatch
#
#SBATCH --job-name=olt-r4
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r4-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r4-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-31

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
OUT_DIR="${OUT_DIR:-$GATE/results/olt_r4_lastepoch_f4000_pg16}"
SELECTED_DIR="$GATE/results/dobj_f4000_pg16"
mkdir -p "$OUT_DIR" logs

[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
# The comparison is only meaningful against the arms this gate mirrors.
[[ -d "$SELECTED_DIR" ]] || { echo "FAIL LOUD: selected-arm results missing: $SELECTED_DIR" >&2; exit 1; }
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# Physics + gate contract, byte-identical to the selected arms' gate.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
# NOT exported: HEROSIM_BACKLOG_DRAIN_TABLE -- live queues are not seeded. Same as the
# gate this one is compared against; exporting it here would change the cluster.
echo "[KNOBS] Q='${HEROSIM_QUEUE_LENGTH:-<unset: default 100>}' CAP='${GNN_PREFIX_PLATFORM_CAP:-<unset: uncapped>}' DRAIN_TABLE=<deliberately unset>"

T=${SLURM_ARRAY_TASK_ID}
GROUP=$(( T / 16 )); SEED=$(( T % 16 + 1 ))
case $GROUP in
  0) ARMNAME=gnn ;;
  1) ARMNAME=mpoff ;;
  *) echo "FAIL LOUD: task $T is beyond the 32 declared arms" >&2; exit 1 ;;
esac
ARM="lastepoch_${ARMNAME}_s${SEED}"
CK="models/drainable-objective-v1-v1-${ARMNAME}-seed${SEED}-final.pt"
[[ -f "$CK" ]] || { echo "FAIL LOUD: last-epoch checkpoint missing: $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: sidecar missing for $CK" >&2; exit 1; }

# The last-epoch weights must be the same label and the same architecture as the selected
# twin they are compared against. Serving the wrong one would give a clean gate of nothing.
python3 - "$SC" "$ARMNAME" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
got = sc.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1"):
    raise SystemExit(f"FAIL LOUD: sidecar label_objective={got!r}, expected rtt_drift:1")
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing="
                     f"{sc.get('disable_message_passing')!r} for arm {arm}")
print(f"[sidecar] last-epoch {arm}: label={got}")
PY

# The selected twin must already be on disk, or this seed has nothing to be compared with.
TWIN="$SELECTED_DIR/v1_${ARMNAME}_s${SEED}.summary.json"
if [[ ! -f "$TWIN" ]]; then
  echo "[TASK $T] NOTE: selected twin missing ($TWIN) -- this seed is unpaired and will be"
  echo "          reported as such by the read, not silently dropped."
fi

export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

OUT="$OUT_DIR/${ARM}.json"
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] arm=$ARM ckpt=$CK -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] summary exists, not re-running"; exit 0; fi

python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$OUT.partial"

python3 - "$OUT.partial" "$SUMMARY" "$ARM" <<'PY'
import json, sys
raw, summary, arm = sys.argv[1:4]
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
s["arm"] = arm; s["policy_name"] = "gnn"; s["checkpoint_kind"] = "last_epoch"
s["queue_length"] = d.get("queue_length")
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env", "code")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$OUT.partial"
echo "=== task $T done: $ARM ==="
