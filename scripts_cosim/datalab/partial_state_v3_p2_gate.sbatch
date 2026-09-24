#!/bin/bash
# partial_state_v3 -- P2: the live instrument at 6 servers. docs/lineages/partial_state_v3.md.
#
# Does a checkpoint trained under the size-free contract serve as well as its v2 twin where
# the v2 twin works? Three cells this family is always gated on, 16 training seeds per arm,
# v3 and v2 paired by training seed:
#
#   3 cells x 2 representations (v3, v2) x 2 policies (gnn, mpoff) x 16 seeds = 192 arms
#
# The v2 arms are RE-RUN rather than reused so both representations come from one commit,
# and P2-a (blocking) reads the v2 gnn arms' per-cell median against queue_range_v1's plain
# medians 54.819 / 22.350 / 65.401 to three decimals -- the proof that the code which adds v3
# left the v2 serving path untouched. Bars in scripts_cosim/partial_state_v3_read.py.
#
# DECLARED IN ADVANCE: gnn seed 3 is excluded by name on BOTH representations (the v2 draw
# deterministically livelocks the simulator; the v3 draw is new but the seed sets must match),
# so gnn is read at n = 15 and mpoff at n = 16. Every arm name carries cell, representation,
# policy and seed. Template: queue_range_v1_gate.sbatch (285/288 clean).
#
#   sbatch --array=0-191%16 scripts_cosim/datalab/partial_state_v3_p2_gate.sbatch
#
#SBATCH --job-name=psv3-p2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/psv3-p2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/psv3-p2-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-191

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
OUT_DIR="$GATE/results/psv3_p2"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }

CELLS=(cell_s7901_f4000_pg16 cell_s9001_f4000_pg16 cell_s9002_f4000_pg16)
REPS=(v3 v2)

# Physics contract, identical to every drainable_* reading in the record.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}
C=$(( T / 64 )); R=$(( (T % 64) / 32 )); I=$(( T % 32 ))
CELL="${CELLS[$C]}"; REP="${REPS[$R]}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
python3 - "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
bt = float(cfg["scheduler"]["batch_timeout"])
if bt != 16.0:
    raise SystemExit(f"FAIL LOUD: batch_timeout {bt} -- every policy constant is held fixed")
PY

POLICY=gnn
if [[ $I -lt 16 ]]; then ARMNAME=gnn; SEED=$(( I + 1 )); else ARMNAME=mpoff; SEED=$(( I - 15 )); fi
ARM="${CELL}__${REP}_${ARMNAME}_s${SEED}"
if [[ $ARMNAME == gnn && $SEED -eq 3 ]]; then
  echo "[TASK $T] $ARM: gnn seed 3 excluded by name (declared in the registration)"; exit 0
fi
case "$REP" in
  v3) CK="models/partial-state-v3-${ARMNAME}-lr2e3-seed${SEED}.pt"; WANT_CONTRACT=partial_state_v3; WANT_DIM=22 ;;
  v2) CK="models/drainable-objective-v1-v1-${ARMNAME}-lr2e3-seed${SEED}.pt"; WANT_CONTRACT=partial_state_v2; WANT_DIM=38 ;;
esac
[[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
python3 - "$SC" "$ARMNAME" "$WANT_CONTRACT" "$WANT_DIM" <<'PY'
import json, sys
sc, arm, contract, dim = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], int(sys.argv[4])
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing={sc.get('disable_message_passing')!r} for arm {arm}")
if sc.get("partial_state_contract") != contract or int(sc.get("partial_state_feature_dim") or 0) != dim:
    raise SystemExit(f"FAIL LOUD: sidecar contract/dim = {sc.get('partial_state_contract')!r}/"
                     f"{sc.get('partial_state_feature_dim')!r}, this arm expects {contract}/{dim}")
qc = sc.get("queue_feature_contract")
if qc not in (None, "legacy_v0"):
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} was trained under queue contract {qc!r}")
PY
# The serving path ADOPTS the partial-state contract from the sidecar (prefix_serving
# _adopt_or_verify_env); nothing is exported here so a v2 and a v3 arm in one array cannot
# inherit each other's contract from the shell.
unset PARTIAL_STATE_CONTRACT || true
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] cell=$CELL rep=$REP arm=$ARM ck=$CK -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARM}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))
python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$REP" "$ARMNAME" "$SEED" "$WALL" "$WANT_CONTRACT" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, rep, armname, seed, wall, contract = sys.argv[1:10]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
c = out.get("schedulerCounters") or {}
for k in ("queue_range_records", "residence_tasks", "residence_batches"):
    c.pop(k, None)
out.update(arm=arm, cell=cell, representation=rep, arm_kind=armname, training_seed=int(seed),
           policy_name="gnn", wallclock_s=int(wall))
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
# The served contract must be the one the arm claims. The sidecar was verified above and the
# loader adopts it into the environment; provenance is captured early in the run, so a
# missing value is "not yet adopted at capture time", while a DIFFERENT value is a defect.
served = env.get("PARTIAL_STATE_CONTRACT")
if served and served != contract:
    raise SystemExit(f"FAIL LOUD: run_provenance says PARTIAL_STATE_CONTRACT={served!r}, arm is {contract}")
out["served_contract"] = contract
json.dump(out, open(summary + ".partial", "w"), indent=1)
d = out["decile_summary"]
print(f"[RESULT] {arm}  elapsed={out['averageElapsedTime']:.3f}  queue={d['mean_queue_s']:.3f}  "
      f"wait={out['averageWaitTime']:.3f}  served={served}  wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
