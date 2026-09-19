#!/bin/bash
# queue_range_v1 -- the live gate. 288 arms, 3 serving arms x 2 policies x 16 seeds x 3 cells.
#
# Registered in docs/lineages/queue_range_v1.md. The claim: the learned arms lose their early
# advantage over reactive Knative because the queue column they rank candidates with leaves
# the contract it was trained under, and leaves it as the trace gets busier. On the arms' own
# corpus the divisor is 1.0 in all 516 datasets, so dim7 IS raw queue depth, candidate p50 12
# and max 42.
#
#   plain    no knobs        -- the control, re-run rather than reused so that all three arms
#                              come from ONE commit (serving_stability_v1's arms predate the
#                              mechanism instrument and carry no queue_range_summary).
#   pinned   divisor = 1.0   -- training semantics restored: dim7 = raw depth.
#   inrange  + dim7 <= 42,   -- the column can never leave the range it was trained on.
#            dim13 <= 0.44
#
# Q1 (blocking): the knob must have reached the feature builder. This job's own summary step
# refuses to write an arm whose per-batch records are missing -- a knob that did nothing reads
# exactly like one that did not help (drainable_regime_v1's registered general rule).
#
# Per-decile queue AND the per-batch queue-column trace are computed in this job's summary
# step and the ~780 MB raw is thrown away; 288 arms cannot each keep one.
#
# DECLARED IN ADVANCE: `gnn` seed 3's checkpoint deterministically livelocks the simulator
# (drainable_objective_v1 Phase C, reproduced byte-for-byte). It TIMEOUTs on cell_s7901; the
# gnn arm is read at n <= 15, and n = 13 on cell_s7901 where the jam_probe burn also removes
# seeds 8 and 14. --time is 30 min so a livelocked arm dies in minutes.
#
#   sbatch --array=0-287%12 scripts_cosim/datalab/queue_range_v1_gate.sbatch
#
#SBATCH --job-name=qr-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/qr-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/qr-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-287

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
OUT_DIR="$GATE/results/qr_v1"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }

CELLS=(cell_s7901_f4000_pg16 cell_s9001_f4000_pg16 cell_s9002_f4000_pg16)
SERVE_ARMS=(plain pinned inrange)

# Physics contract, identical to every drainable_* reading in the record.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1    # load-bearing: without it there are no per-task records

T=${SLURM_ARRAY_TASK_ID}
C=$(( T / 96 )); R=$(( T % 96 )); A=$(( R / 32 )); I=$(( R % 32 ))
CELL="${CELLS[$C]}"
SERVE="${SERVE_ARMS[$A]}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

# The intervention. Registered values, NOT tuned: 42.0 and 0.44 are the measured corpus
# candidate maxima for dim7 and dim13 (516 datasets, graphs_cache_drainable_objective_v1_v1).
case "$SERVE" in
  plain)   ;;
  pinned)  export GNN_QUEUE_SERVE_DIVISOR=1.0 ;;
  inrange) export GNN_QUEUE_SERVE_DIVISOR=1.0
           export GNN_QUEUE_SERVE_DIM7_CLAMP=42.0
           export GNN_QUEUE_SERVE_DIM13_CLAMP=0.44 ;;
  *) echo "FAIL LOUD: unknown serving arm $SERVE" >&2; exit 1 ;;
esac
echo "[KNOBS] serve=$SERVE DIVISOR='${GNN_QUEUE_SERVE_DIVISOR:-<unset>}' "\
"DIM7_CLAMP='${GNN_QUEUE_SERVE_DIM7_CLAMP:-<unset>}' DIM13_CLAMP='${GNN_QUEUE_SERVE_DIM13_CLAMP:-<unset>}'"

POLICY=gnn
if [[ $I -lt 16 ]]; then ARMNAME=gnn; SEED=$(( I + 1 )); else ARMNAME=mpoff; SEED=$(( I - 15 )); fi
ARM="${CELL}__${SERVE}_${ARMNAME}_s${SEED}"
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
# The whole lineage is about legacy_v0's magnitudes. Serving a checkpoint trained under the
# other contract would make every number here uninterpretable.
qc = sc.get("queue_feature_contract")
if qc not in (None, "legacy_v0"):
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} was trained under queue contract {qc!r}")
PY
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] cell=$CELL serve=$SERVE arm=$ARM -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

RAW="$OUT_DIR/${ARM}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$POLICY" "$SERVE" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
from scripts_cosim.queue_range_v1_read import queue_range_decile_summary

raw, summary, arm, cell, policy, serve = sys.argv[1:7]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out["arm"], out["cell"], out["policy_name"], out["serve_arm"] = arm, cell, policy, serve

# Q1's control, applied here so a knob that never reached the feature builder cannot be
# written out as an arm that reached it and did not help.
c = out.get("schedulerCounters") or {}
records = c.pop("queue_range_records", None)
missing = [k for k in ("qr_batches", "qr_blind_batches", "qr_divisor_above_one_batches",
                       "qr_dim7_over_corpus_batches") if k not in c]
if missing:
    raise SystemExit(f"FAIL LOUD: queue-range counters missing from schedulerCounters: "
                     f"{missing} -- the orchestrator whitelist dropped the instrument")
if not records:
    raise SystemExit("FAIL LOUD: no per-batch queue-column records; this arm served no graph "
                     "through the feature builder and cannot be read against Q0/Q1")
out["queue_range_summary"] = queue_range_decile_summary(
    records, out["decile_summary"]["bounds"])
qr = out["queue_range_summary"]
if serve in ("pinned", "inrange") and qr["pinned_frac"] < 0.99:
    raise SystemExit(f"FAIL LOUD: serve={serve} but only {qr['pinned_frac']:.4f} of batches "
                     "served a divisor of 1.0 -- GNN_QUEUE_SERVE_DIVISOR did not bind")
if serve == "inrange" and qr["max_dim7"] > 42.0:
    raise SystemExit(f"FAIL LOUD: serve=inrange but max dim7 is {qr['max_dim7']} > 42 -- "
                     "GNN_QUEUE_SERVE_DIM7_CLAMP did not bind")
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
json.dump(out, open(summary + ".partial", "w"), indent=1)
d = out["decile_summary"]
print(f"[RESULT] {arm}  elapsed={out['averageElapsedTime']:.3f}  "
      f"queue={d['mean_queue_s']:.3f}  max/min decile={d['queue_max_over_min']}  "
      f"| batches={qr['n_batches']} pinned={qr['pinned_frac']:.3f} "
      f"blind={qr['blind_frac']:.3f} oor={qr['out_of_range_frac']:.3f} "
      f"max_dim7={qr['max_dim7']:.1f} max_div={qr['max_divisor']:.1f}")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
