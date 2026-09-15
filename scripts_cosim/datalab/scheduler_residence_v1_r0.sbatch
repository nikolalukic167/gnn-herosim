#!/bin/bash
# scheduler_residence_v1 -- R0: split `averageWaitTime` into its parts.
#
# Registered in docs/lineages/scheduler_residence_v1.md. `averageWaitTime` is
# `scheduled_time - dispatched_time` -- the WHOLE scheduler-side residence -- and
# queue_range_v1 wrote its 6.87 s up as "batch wait". It contains head-of-line blocking
# behind a batch the task is not part of, the collection window itself, re-queue after
# deferral, and store/mutex serialization. The split has never been measured because the
# collector's own `actual_wait_time` was printed and thrown away (scheduler.py:474).
#
# R0's blocking sub-bar is arithmetic: head_of_line + collection + placement must reconstruct
# `averageWaitTime` to within 1 %, and `residence_unstamped` must be 0. This job's own summary
# step refuses to write an arm that fails either -- a decomposition that cannot be reconciled
# against the number it decomposes is a story.
#
# NOTHING about the serving configuration changes here. Same cells, same 16 s window, same
# checkpoints, same knobs as queue_range_v1's `plain` arms, and GNN_BATCH_POLL_INTERVAL stays
# unset exactly as it was -- the control must be the control. The instrument is required to be
# INERT: these arms must reproduce 54.819 / 22.350 / 65.401 to three decimals.
#
# 96 arms = 3 cells x (16 gnn + 16 mpoff). `mpoff` is the pointwise control; a scheduling
# change is policy-agnostic and no GNN-vs-pointwise claim may be founded here.
#
# DECLARED IN ADVANCE: `gnn` seed 3's checkpoint deterministically livelocks the simulator on
# cell_s7901 (drainable_objective_v1 Phase C, reproduced byte-for-byte). --time is 30 min.
#
#   sbatch --array=0-95%12 scripts_cosim/datalab/scheduler_residence_v1_r0.sbatch
#
#SBATCH --job-name=sr-r0
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sr-r0-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sr-r0-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-95

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
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"
OUT_DIR="$GATE/results/sr_r0"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }

CELLS=(cell_s7901_f4000_pg16 cell_s9001_f4000_pg16 cell_s9002_f4000_pg16)

# Physics contract, identical to every drainable_* reading in the record.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1    # load-bearing: without it there are no per-task records

T=${SLURM_ARRAY_TASK_ID}
C=$(( T / 32 )); I=$(( T % 32 ))
CELL="${CELLS[$C]}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

POLICY=gnn
if [[ $I -lt 16 ]]; then ARMNAME=gnn; SEED=$(( I + 1 )); else ARMNAME=mpoff; SEED=$(( I - 15 )); fi
ARM="${CELL}__residence_${ARMNAME}_s${SEED}"
CK="models/drainable-objective-v1-v1-${ARMNAME}-lr2e3-seed${SEED}.pt"
[[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
python3 - "$SC" "$ARMNAME" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing="
                     f"{sc.get('disable_message_passing')!r} for arm {arm}")
PY
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] cell=$CELL arm=$ARM -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

RAW="$OUT_DIR/${ARM}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$POLICY" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
from scripts_cosim.queue_range_v1_read import queue_range_decile_summary
from scripts_cosim.scheduler_residence_v1_read import residence_summary

raw, summary, arm, cell, policy = sys.argv[1:6]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out["arm"], out["cell"], out["policy_name"] = arm, cell, policy

c = out.get("schedulerCounters") or {}
tasks = c.pop("residence_tasks", None)
batches = c.pop("residence_batches", None)
qr = c.pop("queue_range_records", None)
if "residence_unstamped" not in c:
    raise SystemExit("FAIL LOUD: residence_unstamped missing from schedulerCounters -- the "
                     "orchestrator whitelist dropped the instrument")
if not tasks:
    raise SystemExit("FAIL LOUD: no per-task residence rows; nothing was scheduled through "
                     "the instrumented path and this arm cannot be read against R0")
out["residence_summary"] = residence_summary(
    tasks, batches or [], out["decile_summary"]["bounds"],
    float(out["averageWaitTime"]), int(c["residence_unstamped"]))
if qr:
    out["queue_range_summary"] = queue_range_decile_summary(
        qr, out["decile_summary"]["bounds"])

# R0's blocking sub-bar, applied here so an unreconcilable decomposition is never written out.
r = out["residence_summary"]
if r["unstamped"]:
    raise SystemExit(f"FAIL LOUD: {r['unstamped']} task(s) were scheduled without passing "
                     "through the collector -- the decomposition does not cover every task")
if r["reconstruction_error"] is None or r["reconstruction_error"] > 0.01:
    raise SystemExit(
        f"FAIL LOUD: parts sum to {r['sum_of_parts_s']:.4f} s against averageWaitTime "
        f"{r['average_wait_time_s']:.4f} s (error {r['reconstruction_error']}) -- the "
        "decomposition does not reconcile with the number it decomposes")

out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
json.dump(out, open(summary + ".partial", "w"), indent=1)
d = out["decile_summary"]
print(f"[RESULT] {arm}  elapsed={out['averageElapsedTime']:.3f}  queue={d['mean_queue_s']:.3f}"
      f"  | waitTime {r['average_wait_time_s']:.3f} = HOL {r['head_of_line_s']:.3f}"
      f" + collect {r['collection_s']:.3f} + place {r['placement_s']:.3f}"
      f"  (err {r['reconstruction_error']:.6f}, batches {r['n_batches']},"
      f" mean size {r['mean_batch_size']})")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
