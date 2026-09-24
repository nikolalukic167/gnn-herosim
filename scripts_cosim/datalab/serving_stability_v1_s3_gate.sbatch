#!/bin/bash
# serving_stability_v1 -- S3: the GUARDED arms, the live gate that closes the lineage.
#
# 96 arms = 3 cells x (16 gnn + 16 mpoff), each with GNN_QUEUE_GUARD_K=3. Reactive and the
# unguarded arms are the ones results/ss_s1s2 already holds and are NOT re-run.
#
# S3-a (blocking): 0/16 hangs per cell. GNN_PREFIX_PLATFORM_CAP deadlocks 3/16 here, and
#   this guardrail is one letter away from it, so nothing is assumed -- --time is 30 min.
# S3-b (blocking): the arm's own counters must show the mask bound on >= 5 % of decisions.
#   A knob that did nothing reads exactly like one that did not help.
#
# Per-decile queue is computed in this job's OWN summary step and the ~780 MB raw is thrown
# away -- 99 arms cannot each keep one, and a read that can only afford two captures is how
# offline_live_transfer_v1 R3 ended at n = 2.
#
# DECLARED IN ADVANCE: `gnn` seed 3's V = 1 checkpoint deterministically livelocks the
# simulator (drainable_objective_v1 Phase C, reproduced byte-for-byte). It will TIMEOUT on
# every cell; the gnn arm is therefore n = 15 against a >= 12 seed bar, the same declaration
# offline_live_transfer_v1 R4 carried. --time is 30 min so a livelocked arm dies in minutes.
#
#   sbatch --array=0-98%12 scripts_cosim/datalab/serving_stability_v1_s1_s2_gate.sbatch
#
#SBATCH --job-name=ss-s3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ss-s3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ss-s3-%A_%a.err
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
OUT_DIR="$GATE/results/ss_s3"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }

CELLS=(cell_s7901_f4000_pg16 cell_s9001_f4000_pg16 cell_s9002_f4000_pg16)
# The guardrail under test. Registered in the node as S3_K = 3 and NOT tuned.
export GNN_QUEUE_GUARD_K=3

# Physics contract, identical to every drainable_* reading in the record.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1    # load-bearing: without it there are no per-task records
echo "[KNOBS] Q='${HEROSIM_QUEUE_LENGTH:-<unset: default 100>}' CAP='${GNN_PREFIX_PLATFORM_CAP:-<unset: uncapped>}' GUARDRAIL_K='${GNN_QUEUE_GUARD_K}'"

T=${SLURM_ARRAY_TASK_ID}
C=$(( T / 32 )); I=$(( T % 32 ))
CELL="${CELLS[$C]}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

POLICY=gnn
if [[ $I -lt 16 ]]; then ARMNAME=gnn; SEED=$(( I + 1 )); else ARMNAME=mpoff; SEED=$(( I - 15 )); fi
ARM="${CELL}__guarded_${ARMNAME}_s${SEED}"
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
echo "[TASK $T] cell=$CELL arm=$ARM policy=$POLICY -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

RAW="$OUT_DIR/${ARM}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$POLICY" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary

raw, summary, arm, cell, policy = sys.argv[1:6]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
# Ten numbers instead of 780 MB. Fails loud on empty records rather than reporting a
# perfectly flat queue, which is exactly the statistic S2 reads.
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out["arm"], out["cell"], out["policy_name"] = arm, cell, policy
# S3-b: the guardrail's counters must have survived the orchestrator whitelist. If they are
# absent the gate would silently measure an unguarded arm under a guarded name.
c = out.get("schedulerCounters") or {}
missing = [k for k in ("queue_guard_decisions", "queue_guard_steps_active",
                       "queue_guard_masked") if k not in c]
if missing:
    raise SystemExit(f"FAIL LOUD: guardrail counters missing from schedulerCounters: {missing}")
if not c["queue_guard_decisions"]:
    raise SystemExit("FAIL LOUD: the guardrail saw zero decisions -- GNN_QUEUE_GUARD_K did "
                     "not reach the decoder, so this arm is unguarded under a guarded name")
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
json.dump(out, open(summary + ".partial", "w"), indent=1)
d = out["decile_summary"]
print(f"[RESULT] {arm}  elapsed={out['averageElapsedTime']:.3f}  "
      f"queue={d['mean_queue_s']:.3f}  max/min decile={d['queue_max_over_min']}")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
